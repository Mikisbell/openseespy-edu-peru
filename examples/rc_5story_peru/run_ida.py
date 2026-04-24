#!/usr/bin/env python3
"""
examples/rc_5story_peru/run_ida.py — Incremental Dynamic Analysis
==================================================================

Incremental Dynamic Analysis (Vamvatsikos & Cornell 2002) of the same
5-story RC lumped-plasticity building used by `run_monte_carlo.py`. Each
of the 14 real NGA-West2 horizontal ground motions (7 RSNs x 2 horizontal
components) is scaled to a sequence of 8 intensity measures (IM = PGA in g)
and run through OpenSeesPy. The Engineering Demand Parameter (EDP) is the
maximum interstory drift ratio, amplified by E.030-2018 Art. 28.2 factor
(0.75 * R, R = 8 -> 6.0) to obtain the `drift_inelastic` required for
code-compliance checks.

TOTAL RUNS
----------
14 records x 8 IM levels = 112 NLTH simulations.

OUTPUTS
-------
- `data/processed/ida_results.json` — per-record IM-EDP curves, 16/50/84
  percentile median curve, IM_collapse statistics (EDP > 0.04, FEMA 356
  Table C1-3 near-collapse).
- `articles/figures/fig_09_ida_curves.png` — 14 thin grey curves + solid
  median + 16-84 % shaded band + reference lines (design intensity 0.35 g,
  collapse threshold EDP = 0.04).

HONESTY CLAUSE
--------------
- Numerical divergence in an NLTH run is treated as collapse per
  Vamvatsikos & Cornell 2002 Sec. 3.1; the EDP is clamped to a flatline
  sentinel for plotting, and the run contributes to the collapse
  statistics for that IM level.
- Records that never reach EDP = 0.04 up to PGA = 1.00 g are reported as
  right-censored (IM_collapse = None), not silently clamped.

RECORDS
-------
The 14 NGA-West2 records must be placed in `data/records/` as `.AT2`
files with the filenames listed in `NGA_HORIZONTAL_RECORDS` below.
Download from the PEER NGA-West2 database (https://ngawest2.berkeley.edu)
manually — PEER blocks automated downloads.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

try:
    import openseespy.opensees as ops
except ImportError as exc:
    print(f"[ERR] openseespy not available: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "rc_5story_peru"))

from src.physics.peer_adapter import PeerAdapter  # noqa: E402
import run_monte_carlo as mc  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and reference constants
# ---------------------------------------------------------------------------
DATA_OUT = ROOT / "data" / "processed"
FIG_OUT = ROOT / "articles" / "figures"
DATA_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

# IM ladder: 8 levels from 0.10 g to 1.00 g.
_IM_LEVELS_G = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
# FEMA 356 Table C1-3 near-collapse drift (RC MRF).
_COLLAPSE_DRIFT = 0.04
# E.030-2018 Zone 3 Lima design intensity.
_DESIGN_PGA_G = 0.35
# Flatline divergence sentinel.
_FLATLINE_EDP = 0.10  # 10% drift = well beyond any physical survival

# 7 RSNs x 2 horizontal components each = 14 ground motions.
NGA_HORIZONTAL_RECORDS = [
    {"rsn": 6, "event": "Imperial Valley-02 1940", "component": "ELC180",
     "file": "RSN6_IMPVALL.I_I-ELC180.AT2"},
    {"rsn": 6, "event": "Imperial Valley-02 1940", "component": "ELC270",
     "file": "RSN6_IMPVALL.I_I-ELC270.AT2"},
    {"rsn": 15, "event": "Kern County 1952", "component": "TAF021",
     "file": "RSN15_KERN_TAF021.AT2"},
    {"rsn": 15, "event": "Kern County 1952", "component": "TAF111",
     "file": "RSN15_KERN_TAF111.AT2"},
    {"rsn": 143, "event": "Tabas Iran 1978", "component": "TAB-L1",
     "file": "RSN143_TABAS_TAB-L1.AT2"},
    {"rsn": 143, "event": "Tabas Iran 1978", "component": "TAB-T1",
     "file": "RSN143_TABAS_TAB-T1.AT2"},
    {"rsn": 767, "event": "Loma Prieta 1989", "component": "G03000",
     "file": "RSN767_LOMAP_G03000.AT2"},
    {"rsn": 767, "event": "Loma Prieta 1989", "component": "G03090",
     "file": "RSN767_LOMAP_G03090.AT2"},
    {"rsn": 1158, "event": "Kocaeli Turkey 1999", "component": "DZC180",
     "file": "RSN1158_KOCAELI_DZC180.AT2"},
    {"rsn": 1158, "event": "Kocaeli Turkey 1999", "component": "DZC270",
     "file": "RSN1158_KOCAELI_DZC270.AT2"},
    {"rsn": 1794, "event": "Hector Mine 1999", "component": "JOS090",
     "file": "RSN1794_HECTOR_JOS090.AT2"},
    {"rsn": 1794, "event": "Hector Mine 1999", "component": "JOS360",
     "file": "RSN1794_HECTOR_JOS360.AT2"},
    {"rsn": 4098, "event": "Parkfield 2004", "component": "C01090",
     "file": "RSN4098_PARK2004_C01090.AT2"},
    {"rsn": 4098, "event": "Parkfield 2004", "component": "C01360",
     "file": "RSN4098_PARK2004_C01360.AT2"},
]

# Default location for AT2 records — users place the 14 PEER files here.
RECORDS_DIR = ROOT / "data" / "records"


# ---------------------------------------------------------------------------
# AT2 loader (silences PeerAdapter stdout on Windows cp1252 consoles)
# ---------------------------------------------------------------------------
def _load_record(file_path: Path) -> tuple[np.ndarray, float]:
    adapter = PeerAdapter(target_frequency_hz=100.0)
    with redirect_stdout(io.StringIO()):
        raw = adapter.read_at2_file(file_path)
    return raw["acceleration_g"], raw["dt_original"]


# ---------------------------------------------------------------------------
# IDA core loop
# ---------------------------------------------------------------------------
def _run_single(accel_g: np.ndarray, dt: float) -> dict:
    mc.build_model("lumped_plasticity")
    periods = mc.modal_analysis(4)
    mc.apply_rayleigh_damping(mc.DAMPING_RATIO, periods[0], periods[1])
    return mc.run_time_history(accel_g, dt)


def _ida_for_record(
    meta: dict, accel_native: np.ndarray, dt: float, im_levels: list[float]
) -> list[dict]:
    pga_native = float(np.max(np.abs(accel_native)))
    if pga_native <= 0:
        raise RuntimeError(f"Zero-PGA record: {meta['file']}")
    rows: list[dict] = []
    for im in im_levels:
        scale = im / pga_native
        accel_scaled = accel_native * scale
        t0 = time.time()
        res = _run_single(accel_scaled, dt)
        dt_run = time.time() - t0
        if res["status"] == "ok":
            edp_el = float(res["max_drift_elastic_ratio"])
            edp_in = float(res["max_drift_inelastic_ratio"])
            collapsed = edp_in > _COLLAPSE_DRIFT
            status = "ok"
        else:
            edp_el = float("nan")
            edp_in = _FLATLINE_EDP
            collapsed = True
            status = "diverged_flatline"
        rows.append({
            "im_pga_g": round(im, 3),
            "scale_factor": round(scale, 4),
            "edp_drift_elastic": (
                round(edp_el, 6) if not math.isnan(edp_el) else None
            ),
            "edp_drift_inelastic": round(edp_in, 6),
            "collapsed": bool(collapsed),
            "status": status,
            "runtime_s": round(dt_run, 3),
        })
        print(
            f"    IM={im:.2f}g  scale={scale:.2f}  "
            f"EDP_in={edp_in:.4f}  collapsed={collapsed}",
            flush=True,
        )
    return rows


def _collapse_im_for_record(rows: list[dict]) -> float | None:
    for row in rows:
        if row["collapsed"]:
            return float(row["im_pga_g"])
    return None


# ---------------------------------------------------------------------------
# Figure 9 — IDA curves
# ---------------------------------------------------------------------------
def _plot_ida(
    per_record: list[dict], median: dict, out_path: Path
) -> None:
    if not _HAS_MPL:
        print("[WARN] matplotlib not available — figure skipped")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    for rec in per_record:
        xs = [row["im_pga_g"] for row in rec["rows"]]
        ys = [row["edp_drift_inelastic"] for row in rec["rows"]]
        ax.plot(xs, ys, "-", color="#9090A0", lw=0.8, alpha=0.7)

    im_arr = np.array(median["im_pga_g"])
    ax.fill_between(
        im_arr, median["p16"], median["p84"],
        color="#2060A0", alpha=0.18, label="16%-84% band",
    )
    ax.plot(
        im_arr, median["p50"], "-", color="#0A2A5A", lw=2.2,
        label="Median (50%)",
    )

    ax.axhline(
        _COLLAPSE_DRIFT, ls="--", color="#A02020", lw=1.2,
        label=f"Near-collapse EDP = {_COLLAPSE_DRIFT:.2f}",
    )
    ax.axvline(
        _DESIGN_PGA_G, ls=":", color="#30A030", lw=1.2,
        label=f"E.030 design PGA = {_DESIGN_PGA_G:.2f} g",
    )

    ax.set_xlabel(r"Intensity Measure $IM = PGA$ (g)")
    ax.set_ylabel(r"EDP = max $\theta_\mathrm{story}^\mathrm{inelastic}$")
    ax.set_title(
        "Figure 9. IDA curves for 14 NGA-West2 records (5-story RC)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=max(0.06, 1.05 * np.nanmax(median["p84"])))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    png_path = out_path.with_suffix(".png")
    if png_path != out_path:
        fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[OK] figure: {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--im-levels", type=float, nargs="+", default=_IM_LEVELS_G,
        help="PGA ladder in g (default: 0.10-1.00, 8 levels)",
    )
    parser.add_argument(
        "--records-dir", type=str, default=str(RECORDS_DIR),
        help=f"Directory containing the NGA .AT2 files (default: {RECORDS_DIR})",
    )
    parser.add_argument(
        "--out", type=str, default=str(DATA_OUT / "ida_results.json"),
    )
    parser.add_argument(
        "--fig", type=str, default=str(FIG_OUT / "fig_09_ida_curves.png"),
    )
    args = parser.parse_args()

    records_dir = Path(args.records_dir)

    t0 = time.time()
    print(f"[ida] k_story (physics) = {mc.K_STORY_PHYSICS:.3e} N/m")
    print(f"[ida] V_y (story)       = {mc.V_YIELD_STORY / 1e3:.1f} kN")
    print(
        f"[ida] amplification     = {mc.AMPLIFICATION_INELASTIC:.1f} "
        f"(0.75*R, R={mc.R_FACTOR:.0f})"
    )
    print(f"[ida] IM ladder         = {args.im_levels}")
    print(f"[ida] records           = {len(NGA_HORIZONTAL_RECORDS)}")
    print(
        f"[ida] total NLTH        = "
        f"{len(args.im_levels) * len(NGA_HORIZONTAL_RECORDS)}"
    )

    per_record: list[dict] = []
    n_conv = 0
    n_div = 0
    for j, meta in enumerate(NGA_HORIZONTAL_RECORDS, 1):
        fp = records_dir / meta["file"]
        print(
            f"\n[{j}/{len(NGA_HORIZONTAL_RECORDS)}] "
            f"RSN {meta['rsn']} ({meta['event']} / {meta['component']})",
            flush=True,
        )
        if not fp.exists():
            print(f"  [SKIP] file not found: {fp}", flush=True)
            continue
        accel, dt = _load_record(fp)
        rows = _ida_for_record(meta, accel, dt, args.im_levels)
        collapse_im = _collapse_im_for_record(rows)
        for r in rows:
            if r["status"] == "ok":
                n_conv += 1
            else:
                n_div += 1
        per_record.append({
            **meta,
            "pga_native_g": round(float(np.max(np.abs(accel))), 4),
            "dt_s": float(dt),
            "npts": int(len(accel)),
            "rows": rows,
            "collapse_im_g": collapse_im,
            "right_censored": collapse_im is None,
        })

    # ---- Percentile median curve ----------------------------------
    im_arr = np.array(args.im_levels, dtype=float)
    edp_matrix = np.full((max(len(per_record), 1), len(im_arr)), np.nan)
    for ridx, rec in enumerate(per_record):
        for cidx, row in enumerate(rec["rows"]):
            edp_matrix[ridx, cidx] = row["edp_drift_inelastic"]

    if per_record:
        p16 = np.nanpercentile(edp_matrix, 16, axis=0)
        p50 = np.nanpercentile(edp_matrix, 50, axis=0)
        p84 = np.nanpercentile(edp_matrix, 84, axis=0)
    else:
        p16 = p50 = p84 = np.full(len(im_arr), np.nan)

    median_curve = {
        "im_pga_g": [round(float(x), 3) for x in im_arr],
        "p16": [round(float(x), 6) for x in p16],
        "p50": [round(float(x), 6) for x in p50],
        "p84": [round(float(x), 6) for x in p84],
    }

    # ---- IM_collapse statistics -----------------------------------
    collapse_ims = [
        r["collapse_im_g"] for r in per_record if r["collapse_im_g"] is not None
    ]
    n_right_censored = sum(1 for r in per_record if r["right_censored"])
    if collapse_ims:
        im_collapse_median = float(np.median(collapse_ims))
        im_collapse_p16 = float(np.percentile(collapse_ims, 16))
        im_collapse_p84 = float(np.percentile(collapse_ims, 84))
        im_collapse_min = float(np.min(collapse_ims))
        im_collapse_max = float(np.max(collapse_ims))
    else:
        im_collapse_median = im_collapse_p16 = im_collapse_p84 = None
        im_collapse_min = im_collapse_max = None

    payload = {
        "project_id": mc.PROJECT_ID,
        "domain": "structural",
        "analysis": "incremental_dynamic_analysis",
        "model_reference": (
            "examples/rc_5story_peru/run_monte_carlo.py (lumped_plasticity)"
        ),
        "ssot_block": "config/params.yaml -> rc_mrf",
        "im_definition": "IM = PGA (g), linear scaling per record",
        "edp_definition": (
            "EDP = max interstory drift ratio "
            "(inelastic, amplified by 0.75 * R = 6.0 per E.030 Art. 28.2)"
        ),
        "collapse_threshold": _COLLAPSE_DRIFT,
        "collapse_threshold_source": (
            "FEMA 356 Table C1-3 (RC MRF near-collapse)"
        ),
        "e030_design_pga_g": _DESIGN_PGA_G,
        "im_levels_g": args.im_levels,
        "n_records": len(per_record),
        "n_im_levels": len(args.im_levels),
        "n_nlth_total": len(per_record) * len(args.im_levels),
        "n_converged": n_conv,
        "n_diverged_flatline": n_div,
        "convergence_rate": round(n_conv / max(n_conv + n_div, 1), 4),
        "per_record": per_record,
        "median_curve": median_curve,
        "im_collapse_statistics": {
            "median_g": (
                round(im_collapse_median, 3) if im_collapse_median is not None else None
            ),
            "p16_g": (
                round(im_collapse_p16, 3) if im_collapse_p16 is not None else None
            ),
            "p84_g": (
                round(im_collapse_p84, 3) if im_collapse_p84 is not None else None
            ),
            "min_g": (
                round(im_collapse_min, 3) if im_collapse_min is not None else None
            ),
            "max_g": (
                round(im_collapse_max, 3) if im_collapse_max is not None else None
            ),
            "n_collapsed_records": len(collapse_ims),
            "n_right_censored_records": n_right_censored,
            "right_censored_note": (
                "Records that never exceeded the collapse EDP up to the top "
                "of the IM ladder are excluded from the median; their "
                "collapse_im is reported as 'null' (greater than "
                f"{max(args.im_levels):.2f} g)."
            ),
        },
        "honesty_statement": (
            "Numerical divergence in an NLTH run is interpreted as collapse "
            "per Vamvatsikos & Cornell 2002 Sec. 3.1; the corresponding EDP "
            "is clamped to _FLATLINE_EDP = 0.10 for plotting, and the run "
            "contributes to the collapse statistics for that IM level. "
            "No silent removal of diverged runs."
        ),
        "reproducibility": {
            "records_source": (
                "NGA-West2 PEER (7 RSNs, 14 horizontal components)"
            ),
            "vertical_components_excluded": True,
            "amplification_applied": True,
            "amplification_factor": mc.AMPLIFICATION_INELASTIC,
            "runtime_s": round(time.time() - t0, 2),
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out_path}")

    _plot_ida(per_record, median_curve, Path(args.fig))

    print("\n--- IDA SUMMARY ---")
    print(f"  records            : {len(per_record)}")
    print(f"  IM levels          : {len(args.im_levels)}")
    print(
        f"  NLTH total         : {len(per_record) * len(args.im_levels)}"
    )
    print(f"  converged          : {n_conv}")
    print(f"  diverged->flatline : {n_div}")
    print(
        f"  records collapsed  : {len(collapse_ims)} / {len(per_record)}"
    )
    print(f"  records right-censored : {n_right_censored}")
    if im_collapse_median is not None:
        print(f"  IM_collapse median : {im_collapse_median:.3f} g")
        print(
            f"  IM_collapse 16/84  : "
            f"{im_collapse_p16:.3f} g / {im_collapse_p84:.3f} g"
        )
    else:
        print("  IM_collapse median : none (no record collapsed)")
    print(f"  runtime            : {time.time() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
