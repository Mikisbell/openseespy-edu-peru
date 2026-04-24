#!/usr/bin/env python3
"""
PEER NGA-West2 Automated Downloader
====================================
Downloads seismic records (.AT2) from PEER NGA-West2 using curl as backend.
curl handles TLS reliably on WSL2 (where Python urllib3 has handshake issues).
Credentials are read from .env (PEER_EMAIL / PEER_PASSWORD).

Usage:
    python tools/peer_downloader.py --rsn 766
    python tools/peer_downloader.py --rsn 766 1158 4517
    python tools/peer_downloader.py --rsn 766 --out db/excitation/records/

Credentials setup (add to .env, gitignored):
    PEER_EMAIL=your@email.com
    PEER_PASSWORD=yourpassword

Register free at: https://ngawest2.berkeley.edu/members/sign_up
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "db" / "excitation" / "records"

PEER_BASE = "https://ngawest2.berkeley.edu"
PEER_SIGN_IN = f"{PEER_BASE}/members/sign_in"
PEER_SIGN_OUT = f"{PEER_BASE}/members/sign_out"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def load_credentials() -> tuple[str, str]:
    """Read PEER_EMAIL and PEER_PASSWORD from environment or .env file."""
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    email = os.environ.get("PEER_EMAIL", "")
    password = os.environ.get("PEER_PASSWORD", "")

    if not email or not password:
        print(
            "ERROR: PEER credentials not found.\n"
            "Add to .env (gitignored):\n"
            "  PEER_EMAIL=your@email.com\n"
            "  PEER_PASSWORD=yourpassword\n"
            "Register free at: https://ngawest2.berkeley.edu/members/sign_up",
            file=sys.stderr,
        )
        sys.exit(1)

    return email, password


# ---------------------------------------------------------------------------
# curl helpers
# ---------------------------------------------------------------------------

def _curl(
    url: str,
    cookie_jar: Path,
    *,
    data: dict | None = None,
    referer: str | None = None,
    output: Path | None = None,
    follow: bool = True,
    verbose: bool = False,
) -> tuple[int, str]:
    """
    Run a curl request.  Returns (http_status_code, response_body_or_path).
    Always saves/loads cookies from cookie_jar.
    """
    cmd = [
        "curl",
        "--silent", "--show-error",   # silent progress but show errors
        "--globoff",                  # disable [ ] glob patterns in URLs (Rails params use brackets)
        "--max-time", "120",
        "--cookie", str(cookie_jar),
        "--cookie-jar", str(cookie_jar),
        "--user-agent", UA,
        "--header", "Accept-Language: en-US,en;q=0.9",
        "--header", "Accept: text/html,application/xhtml+xml,*/*",
        "--write-out", "\n__STATUS__:%{http_code}",
        "--ipv4",  # force IPv4 (WSL2 has no IPv6 routing)
    ]
    if follow:
        cmd += ["--location"]
    if referer:
        cmd += ["--referer", referer]
    if output:
        cmd += ["--output", str(output)]
    # Write POST body to a temp file to avoid shell-escaping issues
    # with CSRF tokens (contain +, =, /) and field names with [ ]
    _post_file = None
    if data:
        body = urllib.parse.urlencode(data)
        _post_file = Path(tempfile.mktemp(suffix=".post"))
        _post_file.write_text(body, encoding="utf-8")
        cmd += ["--data", f"@{_post_file}",
                "--header", "Content-Type: application/x-www-form-urlencoded"]
    cmd.append(url)

    if verbose:
        print(f"  [curl] {'POST' if data else 'GET'} {url}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
    if _post_file and _post_file.exists():
        _post_file.unlink()

    body = result.stdout
    status = 0
    if "__STATUS__:" in body:
        parts = body.rsplit("__STATUS__:", 1)
        body = parts[0].rstrip("\n")
        try:
            status = int(parts[1].strip())
        except ValueError:
            status = 0

    if result.returncode != 0 and not output:
        # curl error (not HTTP error) — include exit code for diagnosis
        err = result.stderr.strip() or f"(no stderr, curl exit={result.returncode})"
        return -1, err

    if output:
        return status, str(output)
    return status, body


def _extract_csrf(html: str) -> str:
    """Extract Rails authenticity_token from HTML."""
    m = re.search(
        r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']', html
    )
    if m:
        return m.group(1)
    m = re.search(
        r'<input[^>]+name=["\']authenticity_token["\'][^>]+value=["\']([^"\']+)["\']', html
    )
    if m:
        return m.group(1)
    raise ValueError("Could not extract CSRF token from page")


# ---------------------------------------------------------------------------
# PEER session
# ---------------------------------------------------------------------------

class PeerSession:
    """Authenticated curl-backed session to PEER NGA-West2."""

    def __init__(self, email: str, password: str, verbose: bool = True):
        self.email = email
        self.password = password
        self.verbose = verbose
        # Temporary cookie jar (cleared when object is garbage-collected)
        self._tmpdir = tempfile.mkdtemp(prefix="peer_")
        self.cookie_jar = Path(self._tmpdir) / "cookies.txt"
        self._logged_in = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[PEER] {msg}")

    def _get(self, url: str, referer: str | None = None) -> tuple[int, str]:
        return _curl(url, self.cookie_jar, referer=referer, verbose=self.verbose)

    def _post(self, url: str, data: dict, referer: str | None = None) -> tuple[int, str]:
        return _curl(url, self.cookie_jar, data=data, referer=referer, verbose=self.verbose)

    def login(self) -> bool:
        """Perform Rails session login. Returns True on success."""
        self._log("Fetching login page…")
        status, html = self._get(PEER_SIGN_IN)
        if status not in (200, 302) or not html:
            self._log(f"Login page failed: HTTP {status}")
            return False

        try:
            csrf = _extract_csrf(html)
        except ValueError as exc:
            self._log(f"CSRF extraction failed: {exc}")
            return False

        spinner_match = re.search(r'name="spinner"[^>]+value="([^"]+)"', html)
        spinner_value = spinner_match.group(1) if spinner_match else ""

        self._log("Posting credentials…")
        status2, body2 = self._post(
            PEER_SIGN_IN,
            data={
                "authenticity_token": csrf,
                "member[email]": self.email,
                "member[password]": self.password,
                "member[remember_me]": "0",
                "member[subtitle]": "",  # honeypot — empty for humans
                "spinner": spinner_value,
                "commit": "Log in",
            },
            referer=PEER_SIGN_IN,
        )

        if status2 == -1:
            self._log(f"curl error: {body2}")
            return False

        # Failure: login page returned with error message
        if "Invalid Email or password" in body2 or "sign_in" in body2.lower() and "error" in body2.lower():
            self._log("Login failed — invalid credentials")
            return False

        # Success indicators
        if "sign_out" in body2.lower() or "log out" in body2.lower():
            self._log("Login successful")
            self._logged_in = True
            return True

        # PEER sometimes redirects to homepage — treat 200 without sign_in as success
        if status2 == 200 and "sign_in" not in body2[:500].lower():
            self._log("Login successful (redirected to dashboard)")
            self._logged_in = True
            return True

        self._log(f"Login result unclear (HTTP {status2}) — treating as success")
        self._logged_in = True
        return True

    def download_rsn(self, rsn: int, out_dir: Path) -> list[Path]:
        """
        Download .AT2 file(s) for a given RSN.
        Returns list of Paths to downloaded files.
        """
        if not self._logged_in:
            raise RuntimeError("Not logged in — call login() first")

        out_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        existing = list(out_dir.glob(f"RSN{rsn}_*.AT2")) + list(out_dir.glob(f"RSN{rsn}.AT2"))
        if existing:
            self._log(f"RSN{rsn}: already in {out_dir}, skipping")
            return existing

        self._log(f"RSN{rsn}: searching…")

        # Try known PEER download URL patterns
        downloaded = self._try_download_patterns(rsn, out_dir)

        if not downloaded:
            manual = f"{PEER_BASE}/spectras/new?sourceDb_flag=1&rsn={rsn}"
            self._log(
                f"RSN{rsn}: automatic download failed.\n"
                f"  Manual download: {manual}\n"
                f"  Extract .AT2 to: {out_dir}"
            )

        return downloaded

    def _try_download_patterns(self, rsn: int, out_dir: Path) -> list[Path]:
        """Try multiple PEER URL patterns to find downloadable .AT2 files."""

        # Pattern 1: NGA-West2 record search (correct endpoint, not spectras)
        search_url = (
            f"{PEER_BASE}/ngawest2/search"
            f"?search[search_record_sequence_number]={rsn}"
            f"&search[source_db_flag]=1"
        )
        status, html = self._get(search_url)
        if self.verbose:
            print(f"  [PEER] search → HTTP {status}, {len(html or '')} chars")
        if status == 200 and html:
            # Save first 3000 chars for diagnosis
            snippet = (html or '')[:3000]
            (Path(tempfile.gettempdir()) / f"peer_rsn{rsn}_search.html").write_text(snippet, encoding="utf-8")
            files = self._parse_and_download(rsn, html, out_dir)
            if files:
                return files

        # Pattern 2: JSON search API variant
        time.sleep(0.5)
        json_url = (
            f"{PEER_BASE}/ngawest2/search.json"
            f"?search[search_record_sequence_number]={rsn}"
            f"&search[source_db_flag]=1"
        )
        status2, body2 = self._get(json_url)
        if self.verbose:
            print(f"  [PEER] json search → HTTP {status2}, body[:200]: {(body2 or '')[:200]}")
        if status2 == 200:
            files = self._try_json_download(rsn, body2, out_dir)
            if files:
                return files

        # Pattern 3: Try spectras/new (may contain download links as fallback)
        time.sleep(0.5)
        spec_url = f"{PEER_BASE}/spectras/new?sourceDb_flag=1&rsn={rsn}"
        status3, html3 = self._get(spec_url)
        if self.verbose:
            print(f"  [PEER] spectras → HTTP {status3}, {len(html3 or '')} chars")
        if status3 == 200 and html3:
            files = self._parse_and_download(rsn, html3, out_dir)
            if files:
                return files

        return []

    def _parse_and_download(self, rsn: int, html: str, out_dir: Path) -> list[Path]:
        """Parse HTML page to find and download .AT2 files."""
        downloaded = []

        # Find .AT2 direct links
        at2_links = re.findall(
            r'href=["\']([^"\']*\.AT2[^"\']*)["\']', html, re.IGNORECASE
        )
        # Find ZIP download links
        zip_links = re.findall(
            r'href=["\']([^"\']*(?:download|zip)[^"\']*)["\']', html, re.IGNORECASE
        )

        self._log(f"RSN{rsn}: found {len(at2_links)} .AT2 links, {len(zip_links)} ZIP links")

        for link in at2_links[:3]:
            url = link if link.startswith("http") else f"{PEER_BASE}{link}"
            fname = re.sub(r'\?.*', '', url.split("/")[-1]) or f"RSN{rsn}.AT2"
            dest = out_dir / fname
            self._log(f"RSN{rsn}: downloading {fname}…")
            tmp = Path(self._tmpdir) / fname
            s, _ = _curl(url, self.cookie_jar, output=tmp, follow=True)
            if s == 200 and tmp.exists() and tmp.stat().st_size > 100:
                tmp.rename(dest)
                downloaded.append(dest)
                self._log(f"RSN{rsn}: saved {dest.name} ({dest.stat().st_size//1024}KB)")

        if not downloaded:
            for link in zip_links[:2]:
                url = link if link.startswith("http") else f"{PEER_BASE}{link}"
                tmp_zip = Path(self._tmpdir) / f"RSN{rsn}.zip"
                s, _ = _curl(url, self.cookie_jar, output=tmp_zip, follow=True)
                if s == 200 and tmp_zip.exists() and tmp_zip.stat().st_size > 100:
                    extracted = _extract_zip(tmp_zip, out_dir, rsn)
                    downloaded.extend(extracted)
                    if extracted:
                        break

        return downloaded

    def _try_json_download(self, rsn: int, body: str, out_dir: Path) -> list[Path]:
        """Parse JSON API response to find record download URLs."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return []

        urls = []
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    for key in ("at2_url", "download_url", "url", "file_url"):
                        if rec.get(key):
                            urls.append(rec[key])
                            break

        downloaded = []
        for url in urls[:3]:
            if not url.startswith("http"):
                url = f"{PEER_BASE}{url}"
            fname = url.split("/")[-1].split("?")[0] or f"RSN{rsn}.AT2"
            dest = out_dir / fname
            tmp = Path(self._tmpdir) / fname
            s, _ = _curl(url, self.cookie_jar, output=tmp, follow=True)
            if s == 200 and tmp.exists() and tmp.stat().st_size > 100:
                tmp.rename(dest)
                downloaded.append(dest)
        return downloaded

    def logout(self) -> None:
        """Sign out politely."""
        if self._logged_in:
            try:
                _curl(PEER_SIGN_OUT, self.cookie_jar, follow=False, verbose=False)
            except (OSError, subprocess.TimeoutExpired, RuntimeError):
                pass  # best-effort signout — failure is non-critical
            self._logged_in = False

    def __del__(self):
        """Cleanup temp directory."""
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except (OSError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# ZIP extraction helper
# ---------------------------------------------------------------------------

def _extract_zip(zip_path: Path, out_dir: Path, rsn: int) -> list[Path]:
    """Extract .AT2 files from a ZIP archive."""
    extracted = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.upper().endswith(".AT2"):
                    fname = Path(name).name
                    dest = out_dir / fname
                    dest.write_bytes(zf.read(name))
                    extracted.append(dest)
    except zipfile.BadZipFile:
        pass
    return extracted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_records(
    rsns: list[int],
    out_dir: Path = DEFAULT_OUT,
    verbose: bool = True,
) -> dict[int, list[Path]]:
    """
    Download multiple RSN records to out_dir.
    Returns {rsn: [paths]} for each RSN.
    Credentials are read from PEER_EMAIL / PEER_PASSWORD (.env or env vars).
    """
    email, password = load_credentials()
    peer = PeerSession(email, password, verbose=verbose)

    if not peer.login():
        print("ERROR: PEER login failed", file=sys.stderr)
        return {rsn: [] for rsn in rsns}

    results: dict[int, list[Path]] = {}
    for rsn in rsns:
        try:
            files = peer.download_rsn(rsn, out_dir)
            results[rsn] = files
            time.sleep(1.5)  # polite delay
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            print(f"[PEER] RSN{rsn}: ERROR — {exc}", file=sys.stderr)
            results[rsn] = []

    peer.logout()
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download PEER NGA-West2 seismic records (.AT2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--rsn",
        nargs="+",
        type=int,
        required=True,
        help="RSN number(s) to download (e.g. --rsn 766 1158 4517)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    return p.parse_args()


def main() -> None:
    if sys.platform == "win32":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    args = _parse_args()
    results = download_records(args.rsn, out_dir=args.out, verbose=not args.quiet)

    print("\n=== PEER Download Summary ===")
    total_files = 0
    for rsn, files in results.items():
        if files:
            print(f"  RSN{rsn}: {len(files)} file(s) → {', '.join(f.name for f in files)}")
            total_files += len(files)
        else:
            print(f"  RSN{rsn}: FAILED — download manually:", file=sys.stderr)
            print(f"    {PEER_BASE}/spectras/new?sourceDb_flag=1&rsn={rsn}", file=sys.stderr)
    print(f"Total: {total_files} .AT2 file(s) downloaded to {args.out}")


if __name__ == "__main__":
    main()
