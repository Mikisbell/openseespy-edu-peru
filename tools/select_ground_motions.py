#!/usr/bin/env python3
"""Ground motion selection tool using NGA-West2/NGA-Sub flatfiles.

Queries the local flatfile CSV to select ground motions matching site
conditions per ASCE 7-22 SS11.4 criteria. Generates a selection log
and updates db/manifest.yaml with the RSN list.

Usage:
    python tools/select_ground_motions.py
    python tools/select_ground_motions.py --magnitude 6.5 8.0 --distance 0 100 --vs30 250 400
    python tools/select_ground_motions.py --mechanism subduction --records 11
    python tools/select_ground_motions.py --flatfile db/excitation/flatfiles/nga_sub_flatfile.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# YAML mini-loader (stdlib only — no PyYAML dependency required at runtime)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML file. Uses PyYAML if available, else a naive parser."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except yaml.YAMLError as e:
        print(f"[SELECT] WARNING: {path} is malformed YAML: {e} — skipping", file=sys.stderr)
        return {}
    except ImportError:
        # Fallback: extract simple key: value pairs (enough for flat configs)
        data: dict = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" in stripped:
                    key, _, val = stripped.partition(":")
                    val = val.strip().strip('"').strip("'")
                    if val == "null" or val == "~" or val == "":
                        continue
                    try:
                        val = float(val)  # type: ignore[assignment]
                        if val == int(val):
                            val = int(val)  # type: ignore[assignment]
                    except ValueError:
                        pass
                    data[key.strip()] = val
        return data


def _dump_yaml(data: dict, path: Path) -> None:
    """Write *data* as YAML. Uses PyYAML when available, else manual fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    except ImportError:
        print("  WARNING: PyYAML not installed. Using fallback YAML writer.", file=sys.stderr)
        print("  For best results: pip install pyyaml", file=sys.stderr)
        with open(path, "w", encoding="utf-8") as fh:
            _write_yaml_manual(fh, data, indent=0)


def _write_yaml_manual(fh, obj, indent: int = 0) -> None:
    """Recursively write a dict/list as readable YAML (no PyYAML)."""
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                fh.write(f"{prefix}{k}:\n")
                _write_yaml_manual(fh, v, indent + 1)
            else:
                fh.write(f"{prefix}{k}: {_yaml_scalar(v)}\n")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                fh.write(f"{prefix}-\n")
                _write_yaml_manual(fh, item, indent + 1)
            else:
                fh.write(f"{prefix}- {_yaml_scalar(item)}\n")


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        # Quote strings that could be misread
        if v == "" or v[0] in "{}[]&*?|>!%@`":
            return f'"{v}"'
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Column name resolution — NGA-West2 flatfiles have varying column headers
# ---------------------------------------------------------------------------

_COL_ALIASES = {
    "rsn": [
        "Record Sequence Number",
        "RecordSequenceNumber",
        "RSN",
        "Record_Sequence_Number",
    ],
    "magnitude": [
        "Earthquake Magnitude",
        "EarthquakeMagnitude",
        "Magnitude",
        "Mag",
        "mag",
    ],
    "distance": [
        "ClstD",
        "Closest Distance",
        "Rjb",
        "Rjb (km)",
        "Rrup",
        "Rrup (km)",
        "EpiD",
        "Joyner-Boore Distance",
    ],
    "vs30": [
        "Vs30",
        "Vs30_m_per_sec",
        "Vs30(m/s)",
        "Vs30 (m/s)",
        "VS30",
    ],
    "mechanism": [
        "Mechanism",
        "Fault Mechanism",
        "FaultMechanism",
        "Fault_Mechanism",
        "Mech",
    ],
    "event_name": [
        "Earthquake Name",
        "EarthquakeName",
        "Event Name",
        "EventName",
        "Event",
    ],
    "station_name": [
        "Station Name",
        "StationName",
        "Station",
    ],
}

# Mechanism normalisation map (many representations → canonical names)
_MECH_MAP = {
    # Strike-slip
    "strike-slip": "strike-slip",
    "strike slip": "strike-slip",
    "ss": "strike-slip",
    "0": "strike-slip",
    "0.0": "strike-slip",
    # Reverse / thrust
    "reverse": "reverse",
    "reverse-oblique": "reverse",
    "thrust": "reverse",
    "1": "reverse",
    "1.0": "reverse",
    # Normal
    "normal": "normal",
    "normal-oblique": "normal",
    "2": "normal",
    "2.0": "normal",
    # Subduction
    "subduction": "subduction",
    "interface": "subduction",
    "intraslab": "subduction",
    "3": "subduction",
    "3.0": "subduction",
}


def _resolve_col(header: list[str], field: str) -> str | None:
    """Return the actual column name in *header* matching *field*, or None."""
    aliases = _COL_ALIASES.get(field, [])
    lower_header = {h.lower().strip(): h for h in header}
    for alias in aliases:
        if alias in header:
            return alias
        low = alias.lower().strip()
        if low in lower_header:
            return lower_header[low]
    return None


def _norm_mechanism(raw: str) -> str:
    """Normalise a mechanism string to a canonical name."""
    return _MECH_MAP.get(raw.strip().lower(), raw.strip().lower())


# ---------------------------------------------------------------------------
# Site parameter defaults from SSOT
# ---------------------------------------------------------------------------

def _load_site_defaults() -> dict:
    """Read soil_params.yaml and params.yaml for sensible defaults."""
    defaults: dict = {
        "mag_min": 6.0,
        "mag_max": 8.5,
        "dist_min": 0.0,
        "dist_max": 200.0,
        "vs30_min": 180.0,
        "vs30_max": 760.0,
    }

    soil_path = ROOT / "config" / "soil_params.yaml"
    if soil_path.exists():
        soil = _load_yaml(soil_path)

        # Magnitude range heuristic from seismic zone
        zone = None
        if isinstance(soil.get("site_conditions"), dict):
            zone = soil["site_conditions"].get("zone")
        if zone is not None:
            zone = int(zone)
            if zone >= 4:
                defaults["mag_min"] = 6.5
                defaults["mag_max"] = 9.0
            elif zone == 3:
                defaults["mag_min"] = 6.0
                defaults["mag_max"] = 8.0

        # Vs30 from soil type (E.030 Table 3)
        soil_type = None
        if isinstance(soil.get("site_conditions"), dict):
            soil_type = soil["site_conditions"].get("soil_type")
        if soil_type:
            vs30_ranges = {
                "S0": (1500.0, 5000.0),
                "S1": (500.0, 1500.0),
                "S2": (180.0, 500.0),
                "S3": (100.0, 180.0),
                "S4": (50.0, 100.0),
            }
            rng = vs30_ranges.get(str(soil_type).upper())
            if rng:
                defaults["vs30_min"] = rng[0]
                defaults["vs30_max"] = rng[1]

    return defaults


# ---------------------------------------------------------------------------
# Core selection logic
# ---------------------------------------------------------------------------

def load_flatfile(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load the CSV flatfile and return (header, rows)."""
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as e:
        print(f"[SELECT] ERROR: Could not read flatfile {path}: {e}", file=sys.stderr)
        sys.exit(1)
    return header, rows


def _safe_float(val: str | None) -> float | None:
    """Convert to float, returning None on failure."""
    if val is None:
        return None
    val = val.strip()
    if val == "" or val.lower() == "nan":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def select_records(
    header: list[str],
    rows: list[dict[str, str]],
    *,
    mag_min: float,
    mag_max: float,
    dist_min: float,
    dist_max: float,
    vs30_min: float,
    vs30_max: float,
    mechanism: str | None,
    n_records: int,
) -> list[dict]:
    """Filter and rank records, returning up to *n_records* selections."""

    col_rsn = _resolve_col(header, "rsn")
    col_mag = _resolve_col(header, "magnitude")
    col_dist = _resolve_col(header, "distance")
    col_vs30 = _resolve_col(header, "vs30")
    col_mech = _resolve_col(header, "mechanism")
    col_event = _resolve_col(header, "event_name")
    col_station = _resolve_col(header, "station_name")

    if col_rsn is None:
        print("ERROR: Cannot find RSN column in flatfile.", file=sys.stderr)
        print(f"  Available columns: {header[:15]}...", file=sys.stderr)
        sys.exit(1)

    missing = []
    if col_mag is None:
        missing.append("magnitude")
    if col_dist is None:
        missing.append("distance")
    if missing:
        print(f"WARNING: Could not resolve columns: {missing}", file=sys.stderr)
        print(f"  Available columns: {header[:20]}...", file=sys.stderr)

    # Midpoint for ranking
    vs30_mid = (vs30_min + vs30_max) / 2.0
    mag_mid = (mag_min + mag_max) / 2.0

    candidates: list[dict] = []

    for row in rows:
        # --- RSN ---
        rsn_val = _safe_float(row.get(col_rsn or "", ""))
        if rsn_val is None:
            continue
        rsn = int(rsn_val)

        # --- Magnitude filter ---
        mag = _safe_float(row.get(col_mag or "", "")) if col_mag else None
        if mag is not None:
            if mag < mag_min or mag > mag_max:
                continue
        elif col_mag is not None:
            continue  # has column but value is bad — skip

        # --- Distance filter ---
        dist = _safe_float(row.get(col_dist or "", "")) if col_dist else None
        if dist is not None:
            if dist < dist_min or dist > dist_max:
                continue
        elif col_dist is not None:
            continue

        # --- Vs30 filter ---
        vs30 = _safe_float(row.get(col_vs30 or "", "")) if col_vs30 else None
        if vs30 is not None:
            if vs30 < vs30_min or vs30 > vs30_max:
                continue
        elif col_vs30 is not None:
            continue

        # --- Mechanism filter ---
        if mechanism and col_mech:
            raw_mech = row.get(col_mech, "")
            if _norm_mechanism(raw_mech) != mechanism:
                continue

        # --- Ranking score (lower = better) ---
        # Primary: Vs30 closeness to midpoint
        # Secondary: Magnitude closeness to midpoint
        score_vs30 = abs(vs30 - vs30_mid) if vs30 is not None else 999.0
        score_mag = abs(mag - mag_mid) if mag is not None else 999.0
        score = score_vs30 + 0.1 * score_mag

        event = row.get(col_event or "", "").strip() if col_event else ""
        station = row.get(col_station or "", "").strip() if col_station else ""
        mech_str = ""
        if col_mech:
            mech_str = _norm_mechanism(row.get(col_mech, ""))

        candidates.append({
            "rsn": rsn,
            "magnitude": mag,
            "distance_km": dist,
            "vs30": vs30,
            "mechanism": mech_str,
            "event": event,
            "station": station,
            "score": score,
        })

    # Sort by score (best first) and pick top N
    candidates.sort(key=lambda r: r["score"])
    return candidates[:n_records]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_selection(records: list[dict]) -> None:
    """Print a formatted table of selected records."""
    if not records:
        print("\nNo records matched the selection criteria.")
        return

    print(f"\n{'='*90}")
    print(f"  SELECTED GROUND MOTIONS ({len(records)} records)")
    print(f"{'='*90}")
    print(f"  {'RSN':>6}  {'Mw':>5}  {'Dist(km)':>9}  {'Vs30':>6}  {'Mechanism':<12}  {'Event':<25}  {'Station':<20}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*12}  {'-'*25}  {'-'*20}")

    for r in records:
        mag_s = f"{r['magnitude']:.1f}" if r["magnitude"] is not None else "  —"
        dist_s = f"{r['distance_km']:.1f}" if r["distance_km"] is not None else "    —"
        vs30_s = f"{r['vs30']:.0f}" if r["vs30"] is not None else "  —"
        event_s = (r["event"][:25] if r["event"] else "—")
        station_s = (r["station"][:20] if r["station"] else "—")
        print(f"  {r['rsn']:>6}  {mag_s:>5}  {dist_s:>9}  {vs30_s:>6}  {r['mechanism']:<12}  {event_s:<25}  {station_s:<20}")

    print(f"{'='*90}")


def save_selection_log(
    records: list[dict],
    output_path: Path,
    *,
    flatfile_path: Path,
    criteria: dict,
) -> None:
    """Write a YAML selection log."""
    rsn_list = [r["rsn"] for r in records]
    log = {
        "selection": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "flatfile": str(flatfile_path),
            "criteria": criteria,
            "n_selected": len(records),
            "rsn_list": rsn_list,
            "records": [
                {
                    "rsn": r["rsn"],
                    "magnitude": r["magnitude"],
                    "distance_km": r["distance_km"],
                    "vs30": r["vs30"],
                    "mechanism": r["mechanism"],
                    "event": r["event"],
                    "station": r["station"],
                }
                for r in records
            ],
        }
    }
    _dump_yaml(log, output_path)
    print(f"\nSelection log saved: {output_path}")


def update_manifest(rsn_list: list[int]) -> None:
    """Update db/manifest.yaml with the selected RSN list."""
    manifest_path = ROOT / "db" / "manifest.yaml"

    manifest: dict = {}
    if manifest_path.exists():
        manifest = _load_yaml(manifest_path)

    if "excitation" not in manifest or not isinstance(manifest.get("excitation"), dict):
        manifest["excitation"] = {}

    manifest["excitation"]["records_needed"] = rsn_list
    manifest["excitation"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _dump_yaml(manifest, manifest_path)
    print(f"Manifest updated: {manifest_path}  ({len(rsn_list)} RSNs)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Select ground motion records from NGA-West2/NGA-Sub flatfile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/select_ground_motions.py\n"
            "  python tools/select_ground_motions.py --magnitude 6.5 8.0 --vs30 250 400\n"
            "  python tools/select_ground_motions.py --mechanism subduction --records 11\n"
        ),
    )
    p.add_argument(
        "--flatfile",
        type=Path,
        default=ROOT / "db" / "excitation" / "flatfiles" / "nga_west2_flatfile.csv",
        help="Path to flatfile CSV (default: db/excitation/flatfiles/nga_west2_flatfile.csv)",
    )
    p.add_argument(
        "--magnitude",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Magnitude range (default: from soil_params.yaml or 6.0 8.5)",
    )
    p.add_argument(
        "--distance",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Rjb distance in km (default: 0 200)",
    )
    p.add_argument(
        "--vs30",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Vs30 range in m/s (default: from soil_params.yaml or 180 760)",
    )
    p.add_argument(
        "--mechanism",
        type=str,
        default=None,
        choices=["strike-slip", "reverse", "normal", "subduction"],
        help="Fault mechanism filter (default: all)",
    )
    p.add_argument(
        "--records",
        type=int,
        default=11,
        help="Number of records to select (default: 11, per ASCE 7)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Selection log output path (default: db/excitation/selections/selection_YYYYMMDD.yaml)",
    )
    p.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Do NOT update db/manifest.yaml",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Resolve flatfile path ---
    flatfile_path: Path = args.flatfile
    if not flatfile_path.is_absolute():
        flatfile_path = ROOT / flatfile_path

    if not flatfile_path.exists():
        print(f"ERROR: Flatfile not found at {flatfile_path}", file=sys.stderr)
        print(
            "\nDownload NGA-West2 flatfile from:\n"
            "  https://peer.berkeley.edu/nga-west2-flatfiles-and-gmpe-reports-now-available-0\n"
            "\nFor NGA-Sub (subduction) flatfile:\n"
            "  https://www.risksciences.ucla.edu/nhr3/nga-subduction\n"
            f"\nPlace the CSV in: {flatfile_path.parent}/",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load defaults from SSOT ---
    defaults = _load_site_defaults()

    mag_min = args.magnitude[0] if args.magnitude else defaults["mag_min"]
    mag_max = args.magnitude[1] if args.magnitude else defaults["mag_max"]
    dist_min = args.distance[0] if args.distance else defaults["dist_min"]
    dist_max = args.distance[1] if args.distance else defaults["dist_max"]
    vs30_min = args.vs30[0] if args.vs30 else defaults["vs30_min"]
    vs30_max = args.vs30[1] if args.vs30 else defaults["vs30_max"]
    mechanism = args.mechanism

    print("Ground Motion Selection — NGA-West2/NGA-Sub Flatfile Query")
    print(f"  Flatfile:   {flatfile_path}")
    print(f"  Magnitude:  {mag_min:.1f} – {mag_max:.1f}")
    print(f"  Distance:   {dist_min:.0f} – {dist_max:.0f} km")
    print(f"  Vs30:       {vs30_min:.0f} – {vs30_max:.0f} m/s")
    print(f"  Mechanism:  {mechanism or 'all'}")
    print(f"  Records:    {args.records}")

    # --- Load and filter ---
    header, rows = load_flatfile(flatfile_path)
    print(f"\n  Flatfile loaded: {len(rows)} total records, {len(header)} columns")

    selected = select_records(
        header,
        rows,
        mag_min=mag_min,
        mag_max=mag_max,
        dist_min=dist_min,
        dist_max=dist_max,
        vs30_min=vs30_min,
        vs30_max=vs30_max,
        mechanism=mechanism,
        n_records=args.records,
    )

    # --- Output ---
    if not selected:
        print("\nNo records matched the selection criteria.", file=sys.stderr)
        print("Suggestions:", file=sys.stderr)
        print("  - Widen the magnitude range (--magnitude 5.0 9.0)", file=sys.stderr)
        print("  - Widen the distance range  (--distance 0 300)", file=sys.stderr)
        print("  - Widen the Vs30 range      (--vs30 100 1000)", file=sys.stderr)
        print("  - Remove the mechanism filter", file=sys.stderr)
        sys.exit(1)

    print_selection(selected)

    # Selection log
    output_path = args.output
    if output_path is None:
        datestamp = datetime.now().strftime("%Y%m%d")
        output_path = ROOT / "db" / "excitation" / "selections" / f"selection_{datestamp}.yaml"

    criteria = {
        "magnitude": [mag_min, mag_max],
        "distance_km": [dist_min, dist_max],
        "vs30_m_s": [vs30_min, vs30_max],
        "mechanism": mechanism or "all",
    }
    save_selection_log(selected, output_path, flatfile_path=flatfile_path, criteria=criteria)

    # Manifest update
    rsn_list = [r["rsn"] for r in selected]
    if not args.no_update_manifest:
        update_manifest(rsn_list)

    # Download instructions
    print(
        f"\nNext step: Download these {len(rsn_list)} records (AT2 format) from:"
        f"\n  https://ngawest2.berkeley.edu"
        f"\n  RSNs: {', '.join(str(r) for r in rsn_list)}"
        f"\n  Place AT2 files in: {ROOT / 'db' / 'excitation' / 'records'}"
    )


if __name__ == "__main__":
    main()
