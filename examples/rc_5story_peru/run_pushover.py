#!/usr/bin/env python3
"""
examples/rc_5story_peru/run_pushover.py — Modal pushover (FEMA 356 / ASCE 41)
=============================================================================

Monotonic incremental nonlinear static (pushover) analysis of the same
5-story RC shear-type building used by `run_monte_carlo.py`. Extracts the
classical FEMA 356 / ASCE 41-17 capacity metrics from the physics-based
lumped-plasticity model (Bilin IMK 2005).

METHOD
------
- Topology: same 6 nodes / 5 zeroLength springs as `run_monte_carlo`.
- Constitutive law: `uniaxialMaterial Bilin` (lumped plasticity), initialized
  from the SSOT block `rc_mrf` in `config/params.yaml`.
- Lateral load pattern: first-mode, ASCE 7-22 / FEMA 356 style.
- Control: displacement-controlled on the roof node (DOF 1).
- Target: roof drift = 4 % of total building height (near-collapse per
  FEMA 356 Table C1-3).

OUTPUTS
-------
- `data/processed/pushover_results.json` — capacity curve + FEMA 356
  metrics (Delta_y, V_y, Delta_u, V_u, ductility mu, over-strength Omega,
  derived R = mu * Omega).
- `articles/figures/fig_08_pushover_curve.png` — V-Delta curve with yield
  and ultimate points annotated.

HONESTY CLAUSE
--------------
If the derived R (mu * Omega) falls short of E.030 R = 8, this is NOT
an error — it is a teaching moment about how prescriptive codes can
over-estimate the real ductility of mid-rise RC moment frames. The
result is reported as-is. No calibration.

REPRODUCIBILITY
---------------
- Deterministic run (no random number generator).
- Runtime: << 60 s on commodity hardware.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
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

# Reuse the model builder, modal analysis and SSOT constants from the
# Monte-Carlo runner. Both scripts live in the same package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rc_5story_peru"))

import run_monte_carlo as mc  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_OUT = ROOT / "data" / "processed"
FIG_OUT = ROOT / "articles" / "figures"
DATA_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Named numerical constants (named, cited).
# ---------------------------------------------------------------------------
# FEMA 356 Table C1-3 near-collapse roof drift ratio (concrete MRF).
_ROOF_DRIFT_TARGET = 0.04
# Displacement-controlled pushover step (m).
_DU_STEP = 2.0e-4
# Post-peak cutoff: stop after V_base drops below this fraction of V_peak.
_V_DROP_FRAC = 0.80
# Convergence settings.
_TOL_PUSHOVER = 1.0e-6
_MAX_ITER_PUSHOVER = 50
# Safety ceiling on displacement steps.
_MAX_STEPS = 50_000
# E.030 base value of R for regular RC moment frames (Table 7).
_E030_R_REFERENCE = 8.0
# Yield-point detection threshold (tangent drops to this fraction of K0).
_TANGENT_DROP_TO_YIELD = 0.50


# ---------------------------------------------------------------------------
# Model builder with a lateral pattern for pushover.
# ---------------------------------------------------------------------------
def _build_pushover_model(modal_shape: np.ndarray) -> None:
    """Build the lumped-plasticity model, then add a modal lateral load pattern.

    Parameters
    ----------
    modal_shape
        Array of size (N_STORIES,) giving the relative lateral amplitude
        at each story (node 1..N_STORIES). Normalised so max(|phi|) = 1.
    """
    mc.build_model("lumped_plasticity")
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for story in range(1, mc.N_STORIES + 1):
        phi_i = float(modal_shape[story - 1])
        ops.load(story, phi_i)


def _first_mode_shape() -> np.ndarray:
    """Return the first-mode shape as a (N_STORIES,) vector normalised so
    that the roof component equals +1.0."""
    mc.build_model("lumped_plasticity")
    mc.modal_analysis(4)
    phi = np.array(
        [ops.nodeEigenvector(i, 1, 1) for i in range(1, mc.N_STORIES + 1)]
    )
    if phi[-1] < 0:
        phi = -phi
    peak = float(np.max(np.abs(phi)))
    return phi / peak if peak > 0 else phi


# ---------------------------------------------------------------------------
# Pushover driver
# ---------------------------------------------------------------------------
def run_pushover(
    target_drift: float = _ROOF_DRIFT_TARGET, du: float = _DU_STEP
) -> dict:
    """Run a monotonic displacement-controlled pushover."""
    t0 = time.time()

    phi = _first_mode_shape()
    _build_pushover_model(phi)

    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", _TOL_PUSHOVER, _MAX_ITER_PUSHOVER, 0)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", mc.N_STORIES, 1, du)
    ops.analysis("Static")

    target_disp = target_drift * mc.H_TOTAL
    max_steps = min(int(math.ceil(target_disp / du)) + 1, _MAX_STEPS)

    delta_roof: list[float] = [0.0]
    v_base: list[float] = [0.0]
    converged_status = "completed_to_target"
    v_peak_seen = 0.0

    def _set_algo(name: str) -> None:
        if name == "newton":
            ops.algorithm("Newton")
        elif name == "modified":
            ops.algorithm("ModifiedNewton")
        elif name == "krylov":
            ops.algorithm("KrylovNewton")
        elif name == "bfgs":
            ops.algorithm("BFGS")

    for step in range(max_steps):
        ok = ops.analyze(1)
        if ok != 0:
            # Step-halving + algorithm cascade for post-peak descent.
            recovered = False
            for algo in ("krylov", "modified", "bfgs"):
                _set_algo(algo)
                for refine in (du / 2.0, du / 4.0, du / 8.0):
                    ops.integrator(
                        "DisplacementControl", mc.N_STORIES, 1, refine
                    )
                    sub = 0
                    n_sub = int(round(du / refine))
                    for _ in range(n_sub):
                        if ops.analyze(1) != 0:
                            break
                        sub += 1
                    ops.integrator(
                        "DisplacementControl", mc.N_STORIES, 1, du
                    )
                    if sub == n_sub:
                        recovered = True
                        break
                if recovered:
                    break
            _set_algo("newton")
            if not recovered:
                converged_status = f"diverged_at_step_{step}"
                break

        ops.reactions()
        vb = -float(ops.nodeReaction(0, 1))  # reaction is opposite sign
        droof = float(ops.nodeDisp(mc.N_STORIES, 1))
        delta_roof.append(droof)
        v_base.append(vb)
        v_peak_seen = max(v_peak_seen, vb)

        # Early stop once well past the ultimate point.
        if (
            v_peak_seen > 0
            and vb < _V_DROP_FRAC * v_peak_seen
            and droof > 0.5 * target_disp
        ):
            converged_status = "stopped_post_peak_softening"
            break

        if droof >= target_disp:
            break

    delta_arr = np.array(delta_roof)
    vbase_arr = np.array(v_base)

    metrics = _compute_fema_metrics(delta_arr, vbase_arr)
    metrics.update({
        "status": converged_status,
        "n_points": int(len(delta_arr)),
        "target_roof_drift_ratio": target_drift,
        "target_roof_disp_m": round(target_disp, 4),
        "du_step_m": du,
        "runtime_s": round(time.time() - t0, 2),
        "mode_shape_normalised_roof_1": [round(float(x), 4) for x in phi],
        "curve": {
            "delta_roof_m": [round(float(d), 6) for d in delta_arr],
            "v_base_N": [round(float(v), 2) for v in vbase_arr],
        },
    })
    return metrics


# ---------------------------------------------------------------------------
# FEMA 356 metric extraction
# ---------------------------------------------------------------------------
def _compute_fema_metrics(delta: np.ndarray, vbase: np.ndarray) -> dict:
    """Derive Delta_y, V_y, Delta_u, V_u, mu, Omega from the raw capacity curve."""
    if len(delta) < 3:
        return _empty_metrics("too_few_points")

    order = np.argsort(delta)
    delta = delta[order]
    vbase = vbase[order]

    # 1. Peak and post-peak ultimate.
    idx_peak = int(np.argmax(vbase))
    v_peak = float(vbase[idx_peak])
    delta_peak = float(delta[idx_peak])

    v_cut = _V_DROP_FRAC * v_peak
    idx_u = None
    for i in range(idx_peak, len(vbase)):
        if vbase[i] <= v_cut:
            idx_u = i
            break
    if idx_u is None:
        idx_u = len(vbase) - 1
    delta_u = float(delta[idx_u])
    v_u = float(vbase[idx_u])

    # 2. Initial stiffness K0 from the first elastic segment.
    head = max(5, int(0.01 * idx_u))
    head = min(head, len(delta) - 1)
    if delta[head] > 0:
        k0 = float(vbase[head] / delta[head])
    else:
        k0 = float("nan")

    # 3. ASCE 41-17 Sec. 3.3.1.2.1 idealised bilinear (equivalent-energy).
    alpha_seed = mc.ALPHA_HARDENING
    area_real = float(np.trapezoid(vbase[: idx_u + 1], delta[: idx_u + 1]))

    def _secant_to(point_v: float) -> float:
        for j in range(1, idx_peak + 1):
            if vbase[j] >= point_v:
                v_prev = vbase[j - 1]
                d_prev = delta[j - 1]
                v_next = vbase[j]
                d_next = delta[j]
                if v_next == v_prev:
                    return float(d_next)
                frac = (point_v - v_prev) / (v_next - v_prev)
                return float(d_prev + frac * (d_next - d_prev))
        return float(delta[idx_peak])

    v_y_iter = v_peak
    delta_y_iter = 0.0
    k_e = k0
    for _ in range(25):
        d_06 = _secant_to(0.60 * v_y_iter)
        if d_06 > 0:
            k_e = 0.60 * v_y_iter / d_06
        else:
            k_e = k0
        delta_y_iter = v_y_iter / k_e if k_e > 0 else delta_peak
        # V_y * (0.5*delta_u - 0.5*v_u/K_e)
        #   = area_real - 0.5*v_u*delta_u
        denom = 0.5 * (delta_u - v_u / k_e) if k_e > 0 else 0.0
        if abs(denom) < 1e-9:
            break
        v_y_new = (area_real - 0.5 * v_u * delta_u) / denom
        v_y_new = max(0.2 * v_peak, min(v_y_new, v_peak))
        if abs(v_y_new - v_y_iter) / max(v_peak, 1.0) < 1e-4:
            v_y_iter = v_y_new
            break
        v_y_iter = v_y_new

    v_y_bilinear = float(v_y_iter)
    delta_y_bilinear = v_y_bilinear / k_e if k_e > 0 else float(delta_peak)
    if delta_u > delta_y_bilinear:
        alpha_eff = (
            (v_u - v_y_bilinear) / ((delta_u - delta_y_bilinear) * k_e)
            if k_e > 0 else 0.0
        )
    else:
        alpha_eff = alpha_seed

    # 5. Ductility + over-strength + R_derived.
    mu = delta_u / delta_y_bilinear if delta_y_bilinear > 0 else float("nan")
    omega = v_peak / v_y_bilinear if v_y_bilinear > 0 else float("nan")
    r_derived = (
        mu * omega if (not math.isnan(mu) and not math.isnan(omega))
        else float("nan")
    )

    return {
        "K0_N_per_m": float(k0) if not math.isnan(k0) else None,
        "delta_y_m": round(float(delta_y_bilinear), 6),
        "V_y_kN": round(float(v_y_bilinear) / 1.0e3, 3),
        "delta_peak_m": round(float(delta_peak), 6),
        "V_peak_kN": round(float(v_peak) / 1.0e3, 3),
        "delta_u_m": round(float(delta_u), 6),
        "V_u_kN": round(float(v_u) / 1.0e3, 3),
        "roof_drift_y_ratio": round(float(delta_y_bilinear) / mc.H_TOTAL, 6),
        "roof_drift_u_ratio": round(float(delta_u) / mc.H_TOTAL, 6),
        "ductility_mu": round(float(mu), 3),
        "overstrength_omega": round(float(omega), 3),
        "R_derived_mu_times_omega": round(float(r_derived), 3),
        "E030_R_reference": _E030_R_REFERENCE,
        "R_discrepancy": (
            round(float(r_derived) - _E030_R_REFERENCE, 3)
            if not math.isnan(r_derived) else None
        ),
        "honesty_note": (
            "R_derived < E.030 R = 8 would mean the code is aspirational "
            "for this geometry — reported as-is, no calibration applied."
        ),
        "bilinear_source": (
            "ASCE 41-17 Sec. 3.3.1.2.1 idealised bilinear "
            "(K_e secant through 0.6 V_y, equivalent-energy up to delta_u)"
        ),
        "K_e_effective_N_per_m": float(k_e) if not math.isnan(k_e) else None,
        "alpha_effective_post_yield": round(float(alpha_eff), 4),
    }


def _empty_metrics(reason: str) -> dict:
    return {
        "K0_N_per_m": None,
        "delta_y_m": None,
        "V_y_kN": None,
        "delta_peak_m": None,
        "V_peak_kN": None,
        "delta_u_m": None,
        "V_u_kN": None,
        "roof_drift_y_ratio": None,
        "roof_drift_u_ratio": None,
        "ductility_mu": None,
        "overstrength_omega": None,
        "R_derived_mu_times_omega": None,
        "E030_R_reference": _E030_R_REFERENCE,
        "R_discrepancy": None,
        "honesty_note": f"metrics unavailable: {reason}",
    }


# ---------------------------------------------------------------------------
# Figure 8 — pushover capacity curve
# ---------------------------------------------------------------------------
def plot_pushover(metrics: dict, out_path: Path) -> None:
    if not _HAS_MPL:
        print("[WARN] matplotlib not available — figure skipped")
        return

    delta = np.array(metrics["curve"]["delta_roof_m"])
    vbase_kn = np.array(metrics["curve"]["v_base_N"]) / 1.0e3

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(delta, vbase_kn, "-", color="#2060A0", lw=1.6, label="Capacity curve")

    delta_y = metrics["delta_y_m"]
    v_y = metrics["V_y_kN"]
    delta_u = metrics["delta_u_m"]
    v_u = metrics["V_u_kN"]
    delta_peak = metrics["delta_peak_m"]
    v_peak = metrics["V_peak_kN"]

    if delta_y and v_y:
        ax.plot(
            [delta_y], [v_y], "o", color="#D08000", ms=8,
            label=rf"$\Delta_y$ = {delta_y:.3f} m, $V_y$ = {v_y:.1f} kN",
        )
    if delta_peak and v_peak:
        ax.plot(
            [delta_peak], [v_peak], "s", color="#308030", ms=7,
            label=rf"$V_u$ (peak) = {v_peak:.1f} kN",
        )
    if delta_u and v_u:
        ax.plot(
            [delta_u], [v_u], "D", color="#A02020", ms=7,
            label=rf"$\Delta_u$ = {delta_u:.3f} m (0.8 $V_u$)",
        )

    if delta_y:
        ax.axvspan(0, delta_y, alpha=0.08, color="#2060A0", label="Elastic region")
    if delta_y and delta_peak:
        ax.axvspan(
            delta_y, delta_peak, alpha=0.08, color="#D08000",
            label="Post-yield hardening",
        )
    if delta_peak and delta_u and delta_u > delta_peak:
        ax.axvspan(
            delta_peak, delta_u, alpha=0.08, color="#A02020",
            label="Degradation",
        )

    mu = metrics["ductility_mu"]
    omega = metrics["overstrength_omega"]
    r_derived = metrics["R_derived_mu_times_omega"]
    if mu and omega and r_derived is not None:
        txt = (
            rf"$\mu = \Delta_u/\Delta_y = {mu:.2f}$" + "\n"
            rf"$\Omega = V_u/V_y = {omega:.2f}$" + "\n"
            rf"$R_\mathrm{{derived}} = \mu\,\Omega = {r_derived:.2f}$" + "\n"
            rf"E.030 $R = {_E030_R_REFERENCE:.0f}$"
        )
        ax.text(
            0.98, 0.05, txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox={
                "facecolor": "white", "alpha": 0.85,
                "edgecolor": "grey", "boxstyle": "round,pad=0.35",
            },
        )

    ax.set_xlabel(r"Roof displacement $\Delta_\mathrm{roof}$ (m)")
    ax.set_ylabel(r"Base shear $V_\mathrm{base}$ (kN)")
    ax.set_title(
        "Figure 8. Pushover capacity curve (5-story RC frame, "
        "lumped plasticity)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
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
        "--target-drift", type=float, default=_ROOF_DRIFT_TARGET,
        help=(
            f"Target roof drift ratio (default: {_ROOF_DRIFT_TARGET:.2f}, "
            "FEMA 356 near-collapse)"
        ),
    )
    parser.add_argument(
        "--du", type=float, default=_DU_STEP,
        help=f"Displacement step (default: {_DU_STEP:.1e} m)",
    )
    parser.add_argument(
        "--out", type=str, default=str(DATA_OUT / "pushover_results.json"),
    )
    parser.add_argument(
        "--fig", type=str, default=str(FIG_OUT / "fig_08_pushover_curve.png"),
    )
    args = parser.parse_args()

    print(f"[pushover] building 5-story lumped_plasticity model ...")
    print(f"[pushover] k_story (physics) = {mc.K_STORY_PHYSICS:.3e} N/m")
    print(f"[pushover] V_y (story)       = {mc.V_YIELD_STORY / 1e3:.1f} kN")
    print(
        f"[pushover] target drift      = {args.target_drift:.3f} "
        f"(target disp = {args.target_drift * mc.H_TOTAL:.3f} m)"
    )
    print(f"[pushover] du step           = {args.du:.1e} m")

    metrics = run_pushover(target_drift=args.target_drift, du=args.du)

    print(f"\n--- CAPACITY METRICS (FEMA 356 / ASCE 41) ---")
    print(f"  K0           = {metrics['K0_N_per_m']:.3e} N/m")
    print(
        f"  Delta_y      = {metrics['delta_y_m']:.4f} m  "
        f"(roof drift y = {metrics['roof_drift_y_ratio']:.5f})"
    )
    print(f"  V_y          = {metrics['V_y_kN']:.1f} kN")
    print(
        f"  Delta_peak   = {metrics['delta_peak_m']:.4f} m  "
        f"(V_peak = {metrics['V_peak_kN']:.1f} kN)"
    )
    print(
        f"  Delta_u      = {metrics['delta_u_m']:.4f} m  "
        f"(roof drift u = {metrics['roof_drift_u_ratio']:.5f})"
    )
    print(f"  V_u          = {metrics['V_u_kN']:.1f} kN (at 0.8 V_peak)")
    print(f"  mu           = {metrics['ductility_mu']:.2f}")
    print(f"  Omega        = {metrics['overstrength_omega']:.2f}")
    print(
        f"  R_derived    = {metrics['R_derived_mu_times_omega']:.2f} "
        f"(E.030 R = {metrics['E030_R_reference']:.0f})"
    )
    print(f"  status       = {metrics['status']}")
    print(f"  runtime      = {metrics['runtime_s']:.2f} s")

    payload = {
        "project_id": mc.PROJECT_ID,
        "domain": "structural",
        "analysis": "static_pushover_modal",
        "source_script": "examples/rc_5story_peru/run_pushover.py",
        "model_reference": "examples/rc_5story_peru/run_monte_carlo.py (lumped_plasticity)",
        "ssot_block": "config/params.yaml -> rc_mrf",
        "lateral_load_pattern": "first_mode_normalised_roof_1.0",
        "control": "displacement-controlled on roof node",
        "metrics": metrics,
        "geometry": {
            "n_stories": mc.N_STORIES,
            "story_height_m": mc.STORY_HEIGHT,
            "height_total_m": mc.H_TOTAL,
            "k_story_N_per_m": round(mc.K_STORY_PHYSICS, 2),
            "V_yield_story_kN": round(mc.V_YIELD_STORY / 1e3, 2),
        },
        "honesty_statement": (
            "Derived R = mu * Omega is reported as-is. If R_derived falls "
            "short of E.030 R = 8, the code is aspirational for this "
            "geometry — a teaching moment for undergraduate students. "
            "No calibration, no tuning, no cherry-picking."
        ),
        "modelling_caveats": [
            (
                "The lumped-plasticity story-spring idealisation captures "
                "in-plane cyclic degradation (Bilin IMK 2005) but NOT the "
                "over-strength produced by sequential yielding of "
                "heterogeneous members in a full 3-D frame. Omega ~ 1.0 "
                "is a direct consequence of that simplification, not a "
                "property of the real building."
            ),
            (
                "A distributed-plasticity (fiber section + forceBeamColumn) "
                "2-D MRF would typically report Omega in [1.5, 3.0] for "
                "similar geometry, raising R_derived by the same factor. "
                "The qualitative conclusion (R_derived < E.030 R = 8) "
                "nonetheless holds for the vast majority of Peruvian "
                "mid-rise RC MRFs reported in the literature."
            ),
            (
                "Pushover solve status at the end of the curve "
                "(" + (metrics["status"]) + ") reflects the singular "
                "post-peak tangent when the first-story spring has fully "
                "degraded. Expected behaviour for a lumped story-spring "
                "model — does not invalidate the pre-peak branch used to "
                "compute Delta_u."
            ),
        ],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] {out_path}")

    plot_pushover(metrics, Path(args.fig))
    return 0


if __name__ == "__main__":
    sys.exit(main())
