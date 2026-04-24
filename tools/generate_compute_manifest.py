#!/usr/bin/env python3
"""
generate_compute_manifest.py — COMPUTE gate auto-generator
Scans data/processed/ and db/manifest.yaml to produce COMPUTE_MANIFEST.json.

Usage:
  python tools/generate_compute_manifest.py [--project-id ID] [--design-sources f1,f2]

Arguments:
  --project-id      Project identifier (read from db/manifest.yaml if omitted)
  --design-sources  Comma-separated list of files the design requires in data/processed/
                    (e.g. "drift_results.csv,cv_results.json")
  --emulation       Mark emulation as ran (default: auto-detect)
  --guardian        Mark guardian as validated (default: auto-detect)
  --dry-run         Print the manifest without writing it
  --verify          Verify SSOT integrity of existing manifest (exit 0 match, 1 diff)

Output:
  data/processed/COMPUTE_MANIFEST.json
"""

import argparse
import datetime
import hashlib
import io
import json
import sys
from pathlib import Path


try:
    import yaml
except ImportError:
    print(
        "[MANIFEST] PyYAML not installed. Run: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)


def _running_under_pytest() -> bool:
    import os
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED / "COMPUTE_MANIFEST.json"
DB_MANIFEST = ROOT / "db" / "manifest.yaml"
PARAMS_YAML = ROOT / "config" / "params.yaml"
DOMAINS_DIR = ROOT / "config" / "domains"

# Hash algorithm used for SSOT integrity (stable, stdlib, widely supported).
_HASH_ALGO = "sha256"


def _load_ssot_cm_cfg():
    """Load simulation.compute_manifest section from config/params.yaml.

    Returns a dict with skip_files, emulation_signals, and guardian_results_file.
    Falls back to hardcoded defaults if the section is absent.
    """
    _defaults = {
        "skip_files": [
            "COMPUTE_MANIFEST.json",
            "simulation_summary.json",
            "cv_results.json",
            "guardian_test_results.json",
            "ml_training_set.csv",
        ],
        "emulation_signals": [
            "latest_abort.csv",
            "guardian_test_results.json",
        ],
        "guardian_results_file": "guardian_test_results.json",
    }
    try:
        with open(PARAMS_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cm = cfg.get("simulation", {}).get("compute_manifest", {})
        if not cm:
            return _defaults
        return {
            "skip_files": cm.get("skip_files", _defaults["skip_files"]),
            "emulation_signals": cm.get(
                "emulation_signals", _defaults["emulation_signals"]
            ),
            "guardian_results_file": cm.get(
                "guardian_results_file", _defaults["guardian_results_file"]
            ),
        }
    except FileNotFoundError:
        print(
            "[WARN] config/params.yaml not found — using built-in defaults "
            "for compute_manifest",
            file=sys.stderr,
        )
        return _defaults
    except yaml.YAMLError as e:
        print(
            f"[WARN] config/params.yaml malformed ({e}) — using defaults",
            file=sys.stderr,
        )
        return _defaults
    except OSError as e:
        print(
            f"[WARN] config/params.yaml read error ({e}) — using defaults",
            file=sys.stderr,
        )
        return _defaults


_CM_CFG = _load_ssot_cm_cfg()
SKIP_FILES = set(_CM_CFG["skip_files"])
EMULATION_SIGNALS = set(_CM_CFG["emulation_signals"])
GUARDIAN_RESULTS_FILE = _CM_CFG["guardian_results_file"]


def _read_active_domain() -> str | None:
    """Read the active domain from params.yaml (single-domain, stringly-typed)."""
    try:
        with open(PARAMS_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return None
    project = cfg.get("project", {}) or {}
    return project.get("domain")


def _read_active_domains() -> list[str]:
    """Read the active domain(s) as an ordered list.

    Supports both `project.domains: [A, B]` (multi-domain) and
    `project.domain: A` (single-domain, legacy).
    """
    try:
        with open(PARAMS_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return []
    project = cfg.get("project", {}) or {}
    if "domains" in project and isinstance(project["domains"], list):
        return [d for d in project["domains"] if d]
    if "domain" in project and project["domain"]:
        return [project["domain"]]
    return []


def _hash_file_content(path: Path) -> str:
    """SHA-256 of file content. Used for small SSOT files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _hash_file_metadata(path: Path) -> str:
    """Fast integrity token for large files: sha256(path + mtime_ns + size)."""
    st = path.stat()
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    token = f"{rel}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")
    return f"sha256:{hashlib.sha256(token).hexdigest()}"


def _collect_ssot_inputs(
    active_domains: list[str] | str | None,
) -> list[tuple[str, Path, bool]]:
    """Return list of (logical_name, absolute_path, required) for SSOT sources."""
    inputs: list[tuple[str, Path, bool]] = [
        ("params.yaml", PARAMS_YAML, True),
    ]
    if active_domains is None:
        domains: list[str] = []
    elif isinstance(active_domains, str):
        domains = [active_domains] if active_domains else []
    else:
        domains = list(active_domains)

    for dom in domains:
        dom_yaml = DOMAINS_DIR / f"{dom}.yaml"
        inputs.append((f"domain_{dom}.yaml", dom_yaml, False))
    return inputs


def compute_inputs_integrity(
    processed_dir: Path = PROCESSED,
    files_generated: list[str] | None = None,
    active_domain: str | list[str] | None = None,
) -> dict:
    """Build inputs_integrity dict for the manifest."""
    if active_domain is None:
        active_domains = _read_active_domains()
    elif isinstance(active_domain, str):
        active_domains = [active_domain] if active_domain else []
    else:
        active_domains = list(active_domain)

    legacy_primary = active_domains[0] if active_domains else None

    integrity: dict = {
        "computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "algorithm": _HASH_ALGO,
        "active_domains": active_domains,
        "active_domain": legacy_primary,
        "ssot": {},
        "processed_artifacts": {},
    }

    for name, path, required in _collect_ssot_inputs(active_domains):
        if not path.is_file():
            if required:
                raise RuntimeError(
                    f"Required SSOT input missing: {path} "
                    "(cannot compute integrity)"
                )
            continue
        try:
            integrity["ssot"][name] = _hash_file_content(path)
        except OSError as e:
            if required:
                raise RuntimeError(
                    f"Cannot read required SSOT {path}: {e}"
                ) from e
            print(
                f"[WARN] Could not hash optional SSOT {name}: {e}",
                file=sys.stderr,
            )

    if files_generated:
        for fname in files_generated:
            fpath = processed_dir / fname
            if not fpath.is_file():
                continue
            try:
                integrity["processed_artifacts"][fname] = _hash_file_metadata(
                    fpath
                )
            except OSError as e:
                print(
                    f"[WARN] Could not hash artifact {fname}: {e}",
                    file=sys.stderr,
                )

    return integrity


def verify_inputs_integrity(
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[bool, list[str]]:
    """Recompute hashes and compare against manifest. Returns (ok, diff_messages)."""
    diffs: list[str] = []

    if not manifest_path.is_file():
        return False, [f"COMPUTE_MANIFEST.json not found at {manifest_path}"]

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, [f"Could not read {manifest_path.name}: {e}"]

    integrity = manifest.get("inputs_integrity")
    if not integrity:
        return False, [
            "Old manifest missing 'inputs_integrity' field. "
            "Regenerate with: python tools/generate_compute_manifest.py"
        ]

    stored_ssot = integrity.get("ssot", {}) or {}
    stored_domains = integrity.get("active_domains")
    if stored_domains is None:
        _legacy = integrity.get("active_domain")
        stored_domains = [_legacy] if _legacy else []
    elif not isinstance(stored_domains, list):
        stored_domains = [stored_domains]
    current_domains = _read_active_domains()

    if stored_domains != current_domains:
        diffs.append(
            f"active_domains changed: stored={stored_domains!r} "
            f"current={current_domains!r}"
        )

    for name, stored_hash in stored_ssot.items():
        if name == "params.yaml":
            path = PARAMS_YAML
        elif name.startswith("domain_") and name.endswith(".yaml"):
            dom = name[len("domain_") : -len(".yaml")]
            path = DOMAINS_DIR / f"{dom}.yaml"
        else:
            diffs.append(f"Unknown SSOT name in manifest: {name}")
            continue

        if not path.is_file():
            diffs.append(
                f"SSOT '{name}' declared in manifest but no longer exists: {path}"
            )
            continue
        try:
            actual = _hash_file_content(path)
        except OSError as e:
            diffs.append(f"Could not read {name}: {e}")
            continue
        if actual != stored_hash:
            diffs.append(
                f"SSOT '{name}' changed since the manifest was generated "
                f"(stored={stored_hash[:19]}..., actual={actual[:19]}...)"
            )

    for _dom in current_domains:
        expected_key = f"domain_{_dom}.yaml"
        if (
            expected_key not in stored_ssot
            and (DOMAINS_DIR / f"{_dom}.yaml").is_file()
        ):
            diffs.append(
                f"Active domain '{_dom}' has no hash in the manifest "
                "(domain probably changed after COMPUTE)"
            )

    stored_art = integrity.get("processed_artifacts", {}) or {}
    for fname, stored_hash in stored_art.items():
        fpath = PROCESSED / fname
        if not fpath.is_file():
            diffs.append(
                f"Artifact '{fname}' listed in manifest but no longer exists"
            )
            continue
        try:
            actual = _hash_file_metadata(fpath)
        except OSError as e:
            diffs.append(f"Could not read artifact {fname}: {e}")
            continue
        if actual != stored_hash:
            diffs.append(
                f"Artifact '{fname}' was modified/regenerated since the manifest"
            )

    return (not diffs), diffs


def load_db_manifest():
    try:
        with open(DB_MANIFEST, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(
            "[MANIFEST] db/manifest.yaml not found — "
            "running without record traceability",
            file=sys.stderr,
        )
        return {}
    except yaml.YAMLError as e:
        print(
            f"[MANIFEST] db/manifest.yaml parse error: {e}",
            file=sys.stderr,
        )
        return {}
    except PermissionError as e:
        print(
            f"[MANIFEST] db/manifest.yaml permission denied: {e}",
            file=sys.stderr,
        )
        return {}
    except OSError as e:
        print(
            f"[MANIFEST] db/manifest.yaml read error: {e}",
            file=sys.stderr,
        )
        return {}


def detect_records(db):
    """Extract record filenames from db/manifest.yaml.

    Tolerates both dict entries {filename: X, valid: bool} and plain string
    entries.
    """
    records = []
    excitation = db.get("excitation", {})
    for r in excitation.get("records_present", []):
        if isinstance(r, dict):
            if r.get("valid"):
                records.append(r.get("filename", ""))
        elif isinstance(r, str):
            records.append(r)
    return [r for r in records if r]


def count_simulations(processed_dir):
    """Count CSV/NPY files that look like simulation outputs (not metadata)."""
    count = 0
    for f in processed_dir.glob("*.csv"):
        if f.name not in SKIP_FILES:
            count += 1
    for f in processed_dir.glob("*.npy"):
        count += 1
    return count


def detect_emulation(processed_dir):
    """Emulation ran if any file from emulation_signals exists."""
    return any((processed_dir / name).exists() for name in EMULATION_SIGNALS)


def detect_guardian(processed_dir):
    """Guardian validated if guardian_results_file exists with all_gates_pass=true."""
    gtr = processed_dir / GUARDIAN_RESULTS_FILE
    if not gtr.exists():
        return False
    try:
        with open(gtr, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("all_gates_pass", False)
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"[WARN] detect_guardian: {e}", file=sys.stderr)
        return False


def check_design_sources(design_sources, processed_dir):
    """Verify that each planned data file exists on disk."""
    missing = []
    try:
        root = processed_dir.parent.parent
    except AttributeError:
        root = processed_dir
    for raw in design_sources:
        name = raw.strip()
        if not name:
            continue
        candidate_path = Path(name)
        if candidate_path.is_absolute():
            exists = candidate_path.exists()
        elif "/" in name or "\\" in name:
            root_candidate = root / name
            exists = root_candidate.exists()
            if not exists:
                exists = (processed_dir / name).exists()
        else:
            exists = (processed_dir / name).exists()
        if not exists:
            missing.append(name)
    return missing


def main():
    if sys.platform == "win32" and not _running_under_pytest():
        for _name in ("stdout", "stderr"):
            _s = getattr(sys, _name)
            if hasattr(_s, "buffer"):
                try:
                    setattr(
                        sys, _name,
                        io.TextIOWrapper(
                            _s.buffer, encoding="utf-8", errors="replace"
                        ),
                    )
                except (AttributeError, ValueError):
                    pass
    parser = argparse.ArgumentParser(
        description="Generate COMPUTE_MANIFEST.json"
    )
    parser.add_argument(
        "--project-id",
        help="Project identifier (overrides db/manifest.yaml)",
    )
    parser.add_argument(
        "--design-sources", default="",
        help="Comma-separated list of required files in data/processed/",
    )
    parser.add_argument(
        "--emulation", action="store_true",
        help="Force emulation_ran=true",
    )
    parser.add_argument(
        "--guardian", action="store_true",
        help="Force guardian_validated=true",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print manifest without writing to disk",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify SSOT integrity of existing manifest (exit 0 match, 1 diff)",
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help=(
            "Override active domains (list). Useful for ad-hoc multi-domain "
            "manifests without editing params.yaml. Order matters: first = primary."
        ),
    )
    args = parser.parse_args()

    if args.verify:
        ok, diffs = verify_inputs_integrity(MANIFEST_PATH)
        if ok:
            print(
                "[OK] COMPUTE_MANIFEST integrity: SSOT hashes match the manifest."
            )
            sys.exit(0)
        print(
            "[FAIL] COMPUTE_MANIFEST is STALE:",
            file=sys.stderr,
        )
        for d in diffs:
            print(f"  - {d}", file=sys.stderr)
        print(
            "\nFix: re-run the example that writes data/processed/ "
            "(for instance run_monte_carlo.py) to regenerate the manifest "
            "with the current SSOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    db = load_db_manifest()

    project_id = args.project_id or db.get("project_id") or db.get("paper_id", "")
    if not project_id:
        print(
            "[ERROR] project_id not set. Pass --project-id or fix db/manifest.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        PROCESSED.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"[ERROR] Cannot create data/processed/: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    records = detect_records(db)
    sim_count = count_simulations(PROCESSED)
    emulation_ran = args.emulation or detect_emulation(PROCESSED)
    guardian_ok = args.guardian or detect_guardian(PROCESSED)

    design_sources = [s for s in args.design_sources.split(",") if s.strip()]

    if not design_sources:
        traceability = db.get("traceability") or []
        if isinstance(traceability, list):
            for entry in traceability:
                if isinstance(entry, dict):
                    archivo = (
                        entry.get("data_file") or entry.get("archivo") or ""
                    ).strip()
                    if archivo:
                        design_sources.append(archivo)
        if design_sources:
            print(
                f"[MANIFEST] Auto-loaded {len(design_sources)} design "
                "source(s) from db/manifest.yaml"
            )
        else:
            print(
                "[MANIFEST] WARNING: No design sources declared "
                "(--design-sources empty and db/manifest.yaml traceability is empty). "
                "all_design_sources_exist will be False until traceability is filled.",
                file=sys.stderr,
            )

    missing = check_design_sources(design_sources, PROCESSED)
    all_exist = bool(design_sources) and not missing

    files_generated = [
        f.name for f in sorted(PROCESSED.iterdir())
        if (
            f.is_file()
            and f.suffix in {".csv", ".npy", ".json", ".svg", ".png"}
            and f.name != "COMPUTE_MANIFEST.json"
        )
    ]

    cli_domains: list[str] | None = args.domains
    effective_domains = cli_domains if cli_domains else _read_active_domains()
    try:
        inputs_integrity = compute_inputs_integrity(
            processed_dir=PROCESSED,
            files_generated=files_generated,
            active_domain=effective_domains,
        )
    except RuntimeError as e:
        print(
            f"[ERROR] Could not compute inputs_integrity: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest = {
        "compute_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_id": project_id,
        "active_domains": effective_domains,
        "active_domain": effective_domains[0] if effective_domains else None,
        "records_used": records,
        "simulations_run": sim_count,
        "files_generated": files_generated,
        "emulation_ran": emulation_ran,
        "guardian_validated": guardian_ok,
        "all_design_sources_exist": all_exist,
        "is_template_demo": False,
        "gate_passed": False,
        "inputs_integrity": inputs_integrity,
    }

    if missing:
        manifest["missing_design_sources"] = missing

    output = json.dumps(manifest, indent=2)

    if args.dry_run:
        print(output)
        return

    try:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            f.write(output + "\n")
    except OSError as e:
        print(
            f"[ERROR] Cannot write COMPUTE_MANIFEST.json: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[OK] COMPUTE_MANIFEST.json written to {MANIFEST_PATH}")
    print(f"     project_id:         {project_id}")
    print(f"     records_used:       {records}")
    print(f"     simulations_run:    {sim_count}")
    print(f"     emulation_ran:      {emulation_ran}")
    print(f"     guardian_validated: {guardian_ok}")
    print(f"     all_sources_exist:  {all_exist}")

    blocking_errors = []

    if missing:
        print(f"\n[WARN] Missing design sources:", file=sys.stderr)
        for m in missing:
            print(f"       - {m}", file=sys.stderr)
        print(
            "[WARN] all_design_sources_exist=false — downstream consumers "
            "will treat the data as incomplete",
            file=sys.stderr,
        )
        blocking_errors.append("missing_design_sources")

    if sim_count == 0:
        print(
            "[ERROR] 0 simulations detected in data/processed/. "
            "Re-run the example runner.",
            file=sys.stderr,
        )
        blocking_errors.append("no_simulations")

    if blocking_errors:
        sys.exit(1)

    manifest["gate_passed"] = True
    try:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2) + "\n")
    except OSError as e:
        print(
            f"[ERROR] Cannot finalize COMPUTE_MANIFEST.json: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[COMPUTE] Gate PASSED — downstream tools can consume data/processed/.")


if __name__ == "__main__":
    main()
