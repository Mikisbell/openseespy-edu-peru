#!/usr/bin/env python3
"""
src/physics/spectral_engine.py — Response spectrum engine (Duhamel integral)
============================================================================
Computes the pseudo-acceleration response spectrum Sa(T, zeta) by integrating
the SDOF equation of motion via the Newmark average-acceleration method.

Use:
  - Convert a raw accelerogram (.AT2 parsed by PeerAdapter) into its response
    spectrum Sa vs T.
  - Reference codes: ASCE 7-22 Sec. 11.4 / Eurocode 8 Sec. 3.2 / NTE E.030
    (default damping zeta = 5 %).

Formulation:
  Sa(T, zeta) = omega^2 * max_t | integral_0^t ag(tau) *
                 exp(-zeta*omega*(t-tau)) * sin(omega_d*(t-tau)) / omega_d d_tau |

  omega   = 2*pi / T
  omega_d = omega * sqrt(1 - zeta^2)
  zeta    = 0.05  (ASCE 7-22 Sec. 12.1.1 / Eurocode 8 Sec. 3.2 / E.030)
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("[SPECTRAL] numpy not installed. Run: pip install numpy", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Newmark (1959) average-acceleration method: unconditionally stable, zeta-invariant.
# ---------------------------------------------------------------------------
_NEWMARK_BETA = 0.25
_NEWMARK_GAMMA = 0.50

# Default period range for response spectrum (ASCE 7-22 / Eurocode 8 / E.030).
_SA_T_MIN_S = 0.01
_SA_T_MAX_S = 3.0
_SA_N_POINTS = 100

ROOT = Path(__file__).resolve().parent.parent.parent


def compute_spectral_response(
    accel_g: np.ndarray,
    dt: float,
    T_range: np.ndarray | None = None,
    zeta: float = 0.05,
) -> dict:
    """Compute the pseudo-acceleration response spectrum Sa(T, zeta).

    Parameters
    ----------
    accel_g : np.ndarray
        Accelerogram in units of g, sampled at 1/dt Hz.
    dt : float
        Time step in seconds.
    T_range : np.ndarray, optional
        Natural-period array (s). Default: 0.01 to 3.0 s, 100 points.
    zeta : float
        Critical damping ratio (default 0.05 = 5 %, E.030 / ASCE 7-22).

    Returns
    -------
    dict with:
      - T   : array of natural periods
      - Sa  : pseudo-acceleration Sa(T) in g
      - pga : peak ground acceleration of the input record
      - zeta: the damping ratio used
    """
    g_mps2 = 9.81

    if T_range is None:
        T_range = np.linspace(_SA_T_MIN_S, _SA_T_MAX_S, _SA_N_POINTS)

    Sa_arr = np.zeros(len(T_range))
    accel_mps2 = accel_g * g_mps2
    n_steps = len(accel_mps2)

    for i, T in enumerate(T_range):
        omega = 2.0 * np.pi / T
        # omega_d not used in the Newmark stepping below but documented for reference.

        max_disp = 0.0
        u = 0.0
        v = 0.0
        for k in range(n_steps - 1):
            ag_k = accel_mps2[k]
            ag_k1 = accel_mps2[k + 1]

            beta = _NEWMARK_BETA
            gamma = _NEWMARK_GAMMA

            m = 1.0
            k_stif = omega ** 2 * m
            c = 2.0 * zeta * omega * m

            v_pred = v + dt * (1 - gamma) * (
                -ag_k - 2 * zeta * omega * v - omega ** 2 * u
            )
            u_pred = u + dt * v + dt ** 2 * (0.5 - beta) * (
                -ag_k - 2 * zeta * omega * v - omega ** 2 * u
            )

            k_eff = m + gamma * dt * c + beta * dt ** 2 * k_stif
            r_eff = -ag_k1 * m - c * v_pred - k_stif * u_pred
            a_new = r_eff / k_eff

            u = u_pred + beta * dt ** 2 * a_new
            v = v_pred + gamma * dt * a_new

            if abs(u) > max_disp:
                max_disp = abs(u)

        Sa_ms2 = omega ** 2 * max_disp
        Sa_arr[i] = Sa_ms2 / g_mps2

    pga = float(np.max(np.abs(accel_g)))
    return {"T": T_range, "Sa": Sa_arr, "pga": pga, "zeta": zeta}


def generate_spectral_report(
    sa_raw: dict, sa_filtered: dict, record_name: str = "ground motion record"
) -> str:
    """Generate a Markdown table comparing raw vs filtered spectra."""
    T_arr = sa_raw["T"]
    Sa_raw = sa_raw["Sa"]
    Sa_filt = sa_filtered["Sa"]

    indices = np.round(np.linspace(0, len(T_arr) - 1, 10)).astype(int)

    lines = []
    lines.append(
        f"\n### Response Spectrum Sa(T, zeta=5%) — {record_name}\n"
    )
    lines.append(
        f"The Duhamel integral was applied over the normalized {record_name} "
        f"(PGA = {sa_raw['pga']:.3f}g) to compute the pseudo-acceleration "
        f"spectrum (zeta = {sa_raw['zeta']*100:.0f}%, per E.030 / ASCE 7-22):\n"
    )
    lines.append(
        "| Period T (s) | Sa Raw (g) | Sa Filtered (g) | Reduction (%) |"
    )
    lines.append("|---|---|---|---|")
    for idx in indices:
        T = T_arr[idx]
        raw = Sa_raw[idx]
        filt = Sa_filt[idx]
        reduction = ((raw - filt) / raw * 100) if raw > 0 else 0
        lines.append(f"| {T:.2f} | {raw:.4f} | {filt:.4f} | {reduction:.1f}% |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SITE SOIL AMPLIFICATION — code-agnostic (E.030, ASCE 7-22, Eurocode 8)
# Parameters loaded from config/soil_params.yaml if it exists; otherwise the
# caller passes a dict directly to `apply_site_amplification`.
# ---------------------------------------------------------------------------

def load_soil_params(soil_yaml_path=None) -> dict:
    """Load seismic-code soil parameters from `config/soil_params.yaml`.

    Required keys: S (site factor), Tp, Tl (plateau/long-period corners),
    Z (seismic zone PGA), soil_type, zone. C_max defaults to 2.5 (universal).
    """
    try:
        import yaml
    except ImportError:
        print(
            "[SPECTRAL] PyYAML not installed. Run: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    if soil_yaml_path is None:
        soil_yaml_path = ROOT / "config" / "soil_params.yaml"

    _required = ("S", "Tp", "Tl", "Z", "soil_type", "zone")
    try:
        with open(soil_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        amp = data.get("amplification", {})
        plat = data.get("spectral_plateau", {})
        design = data.get("design", {})
        result = {
            "S": amp.get("S"),
            "Tp": amp.get("Tp"),
            "Tl": amp.get("Tl"),
            "Z": design.get("Z"),
            "C_max": float(plat.get("C_max") or 2.5),
            "soil_type": data.get("site_conditions", {}).get("soil_type"),
            "zone": data.get("site_conditions", {}).get("zone"),
        }
        missing = [k for k in _required if result.get(k) is None]
        if missing:
            raise RuntimeError(
                f"[SPECTRAL] soil_params.yaml is missing required keys: {missing}"
            )
        return {
            k: (float(v) if k not in ("soil_type", "zone") else v)
            for k, v in result.items()
        }
    except FileNotFoundError:
        raise RuntimeError(
            f"[SPECTRAL] config/soil_params.yaml not found at {soil_yaml_path}."
        )
    except yaml.YAMLError as e:
        print(f"[SPECTRAL] ERROR: soil_params.yaml malformed: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[SPECTRAL] ERROR: cannot read soil_params.yaml: {e}", file=sys.stderr)
        sys.exit(1)


def compute_c_factor(T: float, Tp: float, Tl: float, C_max: float = 2.5) -> float:
    """Piecewise spectral amplification factor C(T).

    E.030-2018 Sec. 14 / ASCE 7-22 Sec. 11.4 / Eurocode 8 Sec. 3.2.3 — the
    shape is universal across these codes:

        C = C_max                        T  <  Tp        (plateau)
        C = C_max * (Tp / T)             Tp <= T <  Tl   (1/T decay)
        C = C_max * (Tp * Tl / T^2)      T  >= Tl        (1/T^2 decay)
    """
    if T < Tp:
        return C_max
    elif T < Tl:
        return C_max * (Tp / T)
    else:
        return C_max * (Tp * Tl / T ** 2)


def apply_site_amplification(sa_base: dict, soil_params: dict | None = None) -> dict:
    """Convert a base-rock spectrum Sa(T) into a site-specific spectrum.

        Sa_site(T) = Sa_base(T) * S * [ C(T) / C_max ]

    The factor [C(T)/C_max] is the "hump" of the site spectrum: maximum in
    the plateau (T < Tp) and decaying at longer periods, modeling the local
    soil response.
    """
    if soil_params is None:
        soil_params = load_soil_params()

    S = soil_params["S"]
    Tp = soil_params["Tp"]
    Tl = soil_params["Tl"]
    Cmax = soil_params["C_max"]
    T = sa_base["T"]

    C_arr = np.array([compute_c_factor(t, Tp, Tl, Cmax) for t in T])
    Sa_site = sa_base["Sa"] * S * (C_arr / Cmax)

    peak_idx = int(np.argmax(Sa_site))
    T_star = float(T[peak_idx])
    Sa_star = float(Sa_site[peak_idx])
    zone_label = (
        "plateau" if T_star < Tp
        else "1/T decay" if T_star < Tl
        else "1/T^2 decay"
    )

    _code_label = soil_params.get("code", "Code")
    print(
        f"   [{_code_label}] Soil {soil_params['soil_type']} | "
        f"S={S} | Tp={Tp}s | Tl={Tl}s"
    )
    print(
        f"   [{_code_label}] Sa_site max = {Sa_star:.3f}g @ "
        f"T*={T_star:.2f}s ({zone_label})"
    )

    return {
        **sa_base,
        "Sa_site": Sa_site,
        "C_factors": C_arr,
        "soil_params": soil_params,
        "T_star_site": T_star,
        "Sa_star_site": Sa_star,
        "zone_label": zone_label,
    }


# ---------------------------------------------------------------------------
# Damping correction (Eurocode 8 Eq. B.3)
# ---------------------------------------------------------------------------

ZETA_VIRGIN_CONCRETE = 0.050   # ASCE 7-22 / E.030 / Eurocode 8 default
ZETA_MATERIAL_LOW = 0.070      # Eurocode 8 Eq. (3.6) lower bound
ZETA_MATERIAL_NOMINAL = 0.075  # Eurocode 8 nominal
ZETA_MATERIAL_HIGH = 0.080     # Eurocode 8 upper bound


def apply_damping_correction(
    Sa_ref: np.ndarray, zeta_ref: float = 0.05, zeta_target: float = 0.075
) -> np.ndarray:
    """Scale a reference spectrum Sa to a different damping level (Eurocode 8 Eq. B.3).

        Sa(T, zeta_target) ~ Sa(T, zeta_ref) * sqrt(10 / (5 + zeta_target*100))

    The formula uses zeta as a percentage (5, 7.5, ...).
    """
    eta_ref = np.sqrt(10.0 / (5.0 + zeta_ref * 100))
    eta_target = np.sqrt(10.0 / (5.0 + zeta_target * 100))
    return Sa_ref * (eta_target / eta_ref)


def compare_material_vs_reference(sa_base: dict) -> dict:
    """Compare reference damping (5 %) vs study material (7.5 %) spectra."""
    Sa_ref = sa_base["Sa"]
    T_arr = sa_base["T"]

    Sa_virgin = apply_damping_correction(
        Sa_ref, ZETA_VIRGIN_CONCRETE, ZETA_VIRGIN_CONCRETE
    )
    Sa_mat_low = apply_damping_correction(
        Sa_ref, ZETA_VIRGIN_CONCRETE, ZETA_MATERIAL_LOW
    )
    Sa_mat_nominal = apply_damping_correction(
        Sa_ref, ZETA_VIRGIN_CONCRETE, ZETA_MATERIAL_NOMINAL
    )
    Sa_mat_high = apply_damping_correction(
        Sa_ref, ZETA_VIRGIN_CONCRETE, ZETA_MATERIAL_HIGH
    )

    peak_idx = int(np.argmax(Sa_ref))
    T_star = float(T_arr[peak_idx])
    reduction = float(
        (Sa_virgin[peak_idx] - Sa_mat_nominal[peak_idx])
        / Sa_virgin[peak_idx] * 100
    )

    return {
        "T": T_arr,
        "Sa_virgin": Sa_virgin,
        "Sa_mat_low": Sa_mat_low,
        "Sa_mat_nominal": Sa_mat_nominal,
        "Sa_mat_high": Sa_mat_high,
        "T_star": T_star,
        "reduction_pct": round(reduction, 2),
    }


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    # Quick demo: synthetic Kanai-Tajimi-like ground motion.
    dt = 0.005
    t = np.arange(0, 60, dt)
    envelope = (t / 3.0) * np.exp(-t / 3.0)
    accel = (
        np.sin(2 * np.pi * 3.5 * t) * envelope
        + np.random.normal(0, 0.03, len(t))
    )
    scale = 0.33 / np.max(np.abs(accel))
    accel_g = accel * scale

    print("Computing response spectrum Sa(T, zeta=5%)...")
    result = compute_spectral_response(accel_g, dt)
    Sa = result["Sa"]
    T = result["T"]

    peak_idx = np.argmax(Sa)
    print(f"  PGA              : {result['pga']:.3f}g")
    print(f"  Sa max           : {Sa[peak_idx]:.3f}g  @ T = {T[peak_idx]:.2f}s")
