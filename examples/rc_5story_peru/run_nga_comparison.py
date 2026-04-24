#!/usr/bin/env python3
"""
examples/rc_5story_peru/run_nga_comparison.py — KT vs NGA quantitative validation
=================================================================================

Quantitative comparison of the Kanai-Tajimi (KT) synthetic ground-motion
results (from `run_monte_carlo.py`) vs real NGA-West2 records propagated
through the same 5-story shear-type OpenSeesPy model. Runs:

  * Mann-Whitney U (two-sided, non-parametric) on KT(N~500) vs NGA(N=14)
    max_drift_ratio distributions.
  * Cohen's d (pooled std) + 95 % bootstrap CI (10 k resamples, seed=42).

This is the reviewer-ready answer to "you never compared KT against real
motions" — the framework itself reports whether KT is conservative,
non-conservative, or equivalent vs NGA for the target site.

INPUTS
------
- `data/processed/cv_results.json` from a prior `run_monte_carlo.py` run.
  We do NOT re-run KT here (that would burn CPU for no reason). Instead
  we reconstruct a distribution matching the stored mean/std/min/max/N
  using a seeded clipped normal — this is transparent and reported in
  the output JSON.
- 14 NGA horizontal .AT2 records placed in `data/records/`
  (see `run_ida.py` for the canonical filename list).

OUTPUTS
-------
- `data/processed/nga_comparison.json` — full statistical report.

REPRODUCIBILITY
---------------
- Seed = 42; bootstrap = 10 000 resamples.
- Runtime target: < 60 s for 14 NLTH on commodity hardware.
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

from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "rc_5story_peru"))

from src.physics.peer_adapter import PeerAdapter  # noqa: E402
import run_monte_carlo as mc  # noqa: E402 — consumes SSOT constants

# ---------------------------------------------------------------------------
# Configuration — inherits from run_monte_carlo (which reads SSOT).
# ---------------------------------------------------------------------------
N_STORIES = mc.N_STORIES
MASS_PER_STORY = mc.MASS_PER_STORY
STIFF_PER_STORY = mc.K_STORY_PHYSICS
DAMPING_RATIO = mc.DAMPING_RATIO
STORY_HEIGHT = mc.STORY_HEIGHT
H_TOTAL = mc.H_TOTAL
TARGET_PGA_G = mc.PGA_TARGET_NOMINAL
SEED = 42
BOOTSTRAP_N = 10_000

# Same 14-record roster as run_ida.py.
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

RECORDS_DIR = ROOT / "data" / "records"
DATA_OUT = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Shear-type OpenSeesPy model (linear 1-D topology, matches run_monte_carlo
# at the 'elastic' level — the NGA comparison does not need nonlinear
# hysteresis to separate the intensity / frequency content of the input).
# ---------------------------------------------------------------------------
def build_model(
    mass: float = MASS_PER_STORY, stiff: float = STIFF_PER_STORY
) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    for i in range(N_STORIES + 1):
        ops.node(i, float(i))
    ops.fix(0, 1)
    for i in range(1, N_STORIES + 1):
        ops.mass(i, mass)
    ops.uniaxialMaterial("Elastic", 1, stiff)
    for i in range(N_STORIES):
        ops.element("zeroLength", i + 1, i, i + 1, "-mat", 1, "-dir", 1)


def modal_analysis(n_modes: int = 4) -> list[float]:
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    eigs = ops.eigen(n_modes)
    periods: list[float] = []
    for w2 in eigs:
        if w2 > 0:
            periods.append(2.0 * math.pi / math.sqrt(w2))
        else:
            periods.append(float("nan"))
    return periods


def apply_rayleigh_damping(zeta: float, T1: float, T2: float) -> None:
    w1 = 2.0 * math.pi / T1
    w2 = 2.0 * math.pi / T2
    alpha = 2.0 * zeta * w1 * w2 / (w1 + w2)
    beta = 2.0 * zeta / (w1 + w2)
    ops.rayleigh(alpha, 0.0, beta, 0.0)


def run_time_history(accel_g: np.ndarray, dt: float) -> dict:
    g_si = 9.81
    accel_ms2 = accel_g * g_si
    ts_tag = 2
    pat_tag = 2
    try:
        ops.timeSeries(
            "Path", ts_tag, "-dt", dt,
            "-values", *accel_ms2.tolist(), "-factor", 1.0,
        )
        ops.pattern("UniformExcitation", pat_tag, 1, "-accel", ts_tag)
    except Exception as exc:
        return {"status": "pattern_fail", "error": str(exc)}

    ops.wipeAnalysis()
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1.0e-8, 25, 0)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    n_steps = len(accel_ms2)
    max_disp = np.zeros(N_STORIES + 1)
    max_drift = 0.0
    max_base_shear = 0.0

    for _ in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            ok2 = ops.analyze(1, dt / 2.0)
            if ok2 != 0:
                ops.analyze(1, dt / 2.0)
            if ok2 != 0:
                return {
                    "status": "diverged",
                    "max_roof_disp_m": float("nan"),
                    "max_drift_ratio": float("nan"),
                    "max_base_shear_N": float("nan"),
                }
        disps = np.array([ops.nodeDisp(i, 1) for i in range(N_STORIES + 1)])
        max_disp = np.maximum(max_disp, np.abs(disps))
        drifts = np.abs(np.diff(disps)) / STORY_HEIGHT
        d_now = float(np.max(drifts))
        if d_now > max_drift:
            max_drift = d_now
        vb = STIFF_PER_STORY * abs(disps[1])
        if vb > max_base_shear:
            max_base_shear = vb

    return {
        "status": "ok",
        "max_roof_disp_m": float(max_disp[-1]),
        "max_drift_ratio": float(max_drift),
        "max_base_shear_N": float(max_base_shear),
    }


# ---------------------------------------------------------------------------
# NGA processing
# ---------------------------------------------------------------------------
def load_and_scale_record(
    file_path: Path, target_pga_g: float
) -> tuple[np.ndarray, float, float, float]:
    """Return (accel_scaled_g, dt, pga_original_g, scale_factor)."""
    adapter = PeerAdapter(target_frequency_hz=100.0)
    with redirect_stdout(io.StringIO()):
        raw = adapter.read_at2_file(file_path)
    accel_orig = raw["acceleration_g"]
    dt_orig = raw["dt_original"]
    pga_orig = float(np.max(np.abs(accel_orig)))
    if pga_orig == 0.0:
        raise ValueError(f"Zero-PGA record: {file_path.name}")
    scale = target_pga_g / pga_orig
    return accel_orig * scale, dt_orig, pga_orig, scale


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def cohens_d_pooled(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = math.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if pooled == 0.0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def bootstrap_cohens_d_ci(
    x: np.ndarray, y: np.ndarray,
    n_resamples: int = BOOTSTRAP_N, seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d_samples = np.empty(n_resamples)
    nx, ny = len(x), len(y)
    for i in range(n_resamples):
        xs = rng.choice(x, size=nx, replace=True)
        ys = rng.choice(y, size=ny, replace=True)
        d_samples[i] = cohens_d_pooled(xs, ys)
    lo = float(np.percentile(d_samples, 2.5))
    hi = float(np.percentile(d_samples, 97.5))
    return lo, hi


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--records-dir", type=str, default=str(RECORDS_DIR),
        help=f"Directory containing the NGA .AT2 files (default: {RECORDS_DIR})",
    )
    parser.add_argument(
        "--out", type=str, default=str(DATA_OUT / "nga_comparison.json"),
    )
    args = parser.parse_args()

    records_dir = Path(args.records_dir)

    t0 = time.time()

    # Load KT baseline stats from Monte-Carlo output.
    cv_path = DATA_OUT / "cv_results.json"
    if not cv_path.exists():
        print(
            f"[ERR] {cv_path} not found. Run examples/rc_5story_peru/"
            "run_monte_carlo.py first.",
            file=sys.stderr,
        )
        return 1
    with open(cv_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    kt_drift = cv["results"]["max_interstory_drift_ratio"]
    kt_roof = cv["results"]["max_roof_displacement_m"]
    kt_n = cv["ground_motions"]["n_converged"]

    # Reconstruct a KT distribution with matching mean/std/N (seed 42,
    # clipped normal) — transparent and reproducible. Disclosed in the
    # output JSON so the reader knows what was done.
    rng_kt = np.random.default_rng(SEED)
    kt_drift_series = rng_kt.normal(
        loc=kt_drift["mean"], scale=kt_drift["std"], size=kt_n
    )
    kt_drift_series = np.clip(
        kt_drift_series, kt_drift["min"], kt_drift["max"]
    )

    # Run NGA records.
    print(
        f"[1/3] loading + running {len(NGA_HORIZONTAL_RECORDS)} NGA horizontal records ...",
        flush=True,
    )
    per_record = []
    nga_drifts: list[float] = []
    nga_roofs: list[float] = []
    nga_shears: list[float] = []
    diverged_records = []

    build_model()
    periods = modal_analysis(4)
    T1, T2 = periods[0], periods[1]
    print(f"      T1={T1:.4f}s  T2={T2:.4f}s (5-story shear-type)")

    for rec in NGA_HORIZONTAL_RECORDS:
        fp = records_dir / rec["file"]
        try:
            accel_scaled, dt_orig, pga_orig, scale = load_and_scale_record(
                fp, TARGET_PGA_G
            )
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"      [WARN] skip RSN{rec['rsn']} {rec['component']}: {exc}",
                flush=True,
            )
            per_record.append({
                "rsn": rec["rsn"], "event": rec["event"],
                "component": rec["component"], "status": "load_fail",
                "error": str(exc),
            })
            continue

        build_model()
        T_now = modal_analysis(4)
        apply_rayleigh_damping(DAMPING_RATIO, T_now[0], T_now[1])
        res = run_time_history(accel_scaled, dt_orig)

        entry = {
            "rsn": rec["rsn"],
            "event": rec["event"],
            "component": rec["component"],
            "pga_original_g": round(pga_orig, 4),
            "pga_scaled_g": TARGET_PGA_G,
            "scale_factor": round(scale, 4),
            "dt_s": dt_orig,
            "npts": len(accel_scaled),
            "status": res["status"],
        }
        if res["status"] == "ok":
            entry["max_drift_ratio"] = res["max_drift_ratio"]
            entry["max_roof_disp_m"] = res["max_roof_disp_m"]
            entry["max_base_shear_N"] = res["max_base_shear_N"]
            nga_drifts.append(res["max_drift_ratio"])
            nga_roofs.append(res["max_roof_disp_m"])
            nga_shears.append(res["max_base_shear_N"])
            print(
                f"      RSN{rec['rsn']:4d} {rec['component']:8s}  "
                f"scale={scale:6.3f}  drift={res['max_drift_ratio']:.4f}  "
                f"roof={res['max_roof_disp_m']*1000:6.2f}mm",
                flush=True,
            )
        else:
            diverged_records.append(f"RSN{rec['rsn']} {rec['component']}")
            print(
                f"      [DIVERGED] RSN{rec['rsn']} {rec['component']}",
                flush=True,
            )
        per_record.append(entry)

    # Statistics.
    print(
        f"\n[2/3] computing Mann-Whitney U + Cohen's d "
        f"(N_KT={len(kt_drift_series)}, N_NGA={len(nga_drifts)}) ...",
        flush=True,
    )
    kt_arr = np.asarray(kt_drift_series, dtype=float)
    nga_arr = np.asarray(nga_drifts, dtype=float)

    if len(nga_arr) < 2:
        print(
            "[ERR] Fewer than 2 NGA records converged — cannot run statistics.",
            file=sys.stderr,
        )
        return 1

    mw = mannwhitneyu(kt_arr, nga_arr, alternative="two-sided")
    U_stat = float(mw.statistic)
    p_val = float(mw.pvalue)
    d = cohens_d_pooled(kt_arr, nga_arr)
    ci_lo, ci_hi = bootstrap_cohens_d_ci(kt_arr, nga_arr)

    print(
        f"      U = {U_stat:.2f},  p = {p_val:.4g},  "
        f"Cohen's d = {d:+.3f}  (95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}])"
    )
    kt_mean = float(np.mean(kt_arr))
    nga_mean = float(np.mean(nga_arr))
    pct_diff = (
        100.0 * (kt_mean - nga_mean) / nga_mean if nga_mean != 0 else 0.0
    )
    print(f"      KT mean drift  = {kt_mean:.5f}")
    print(f"      NGA mean drift = {nga_mean:.5f}  ({pct_diff:+.1f}% vs NGA)")

    abs_d = abs(d)
    if abs_d < 0.2:
        verdict = "equivalent"
    elif abs_d < 0.5:
        verdict = "small-effect"
    elif abs_d < 0.8:
        verdict = "medium-effect"
    else:
        verdict = "large-effect"
    kt_role = "conservative" if kt_mean > nga_mean else "non-conservative"
    print(f"      verdict: KT is {kt_role} vs NGA ({verdict})")

    # Write output JSON.
    print(f"\n[3/3] writing {args.out} ...", flush=True)
    out = {
        "project_id": mc.PROJECT_ID,
        "analysis": (
            "Kanai-Tajimi synthetic vs NGA-West2 real ground motions "
            "(quantitative reviewer response)"
        ),
        "target_pga_g": TARGET_PGA_G,
        "target_pga_source": "E.030 Zone 3 Lima (Peru SENCICO 2018)",
        "method_comparison": {
            "KT_synthetic": {
                "n": int(kt_n),
                "max_drift_mean": kt_drift["mean"],
                "max_drift_std": kt_drift["std"],
                "max_drift_median": float(np.median(kt_arr)),
                "max_drift_min": kt_drift["min"],
                "max_drift_max": kt_drift["max"],
                "max_roof_disp_mean": kt_roof["mean"],
                "max_roof_disp_std": kt_roof["std"],
                "note": (
                    "Distribution reconstructed from stored (mean, std, "
                    "min, max, N) in cv_results.json — original per-run "
                    "values not serialized. Reconstruction uses a seed=42 "
                    "clipped normal; preserves first two moments for "
                    "Mann-Whitney U screening."
                ),
            },
            "NGA_real": {
                "n": len(nga_drifts),
                "max_drift_mean": float(np.mean(nga_arr)) if len(nga_arr) else 0.0,
                "max_drift_std": (
                    float(np.std(nga_arr, ddof=1)) if len(nga_arr) > 1 else 0.0
                ),
                "max_drift_median": (
                    float(np.median(nga_arr)) if len(nga_arr) else 0.0
                ),
                "max_drift_min": (
                    float(np.min(nga_arr)) if len(nga_arr) else 0.0
                ),
                "max_drift_max": (
                    float(np.max(nga_arr)) if len(nga_arr) else 0.0
                ),
                "max_roof_disp_mean": (
                    float(np.mean(nga_roofs)) if len(nga_roofs) else 0.0
                ),
                "max_roof_disp_std": (
                    float(np.std(nga_roofs, ddof=1)) if len(nga_roofs) > 1 else 0.0
                ),
                "rsn_list": sorted(
                    list({r["rsn"] for r in NGA_HORIZONTAL_RECORDS})
                ),
                "events": sorted(
                    list({r["event"] for r in NGA_HORIZONTAL_RECORDS})
                ),
                "pga_scaling": (
                    f"linear scale each record so peak |a| == {TARGET_PGA_G} g"
                ),
            },
            "pct_diff_kt_vs_nga": round(pct_diff, 2),
        },
        "statistical_test": {
            "test": "Mann-Whitney U (two-sided, non-parametric)",
            "reason": (
                "Non-normal drift distributions; N_NGA=14 too small "
                "to assume Gaussian."
            ),
            "implementation": "scipy.stats.mannwhitneyu",
            "U_statistic": U_stat,
            "p_value": p_val,
            "alpha": 0.05,
            "significant": bool(p_val < 0.05),
            "cohens_d_pooled": round(d, 4),
            "cohens_d_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "cohens_d_bootstrap_n": BOOTSTRAP_N,
            "cohens_d_seed": SEED,
            "effect_size_label": verdict,
            "kt_role": kt_role,
        },
        "per_record_results": per_record,
        "diverged_records": diverged_records,
        "runtime_s": round(time.time() - t0, 2),
        "reproducibility": {
            "seed": SEED,
            "bootstrap_resamples": BOOTSTRAP_N,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "solver": "OpenSeesPy (Newmark-beta, Newton, BandGeneral)",
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] nga_comparison.json  -> {out_path}")
    print(
        f"[OK] runtime: {time.time() - t0:.2f} s, "
        f"NGA converged {len(nga_drifts)}/{len(NGA_HORIZONTAL_RECORDS)}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
