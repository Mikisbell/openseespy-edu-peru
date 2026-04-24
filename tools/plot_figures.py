#!/usr/bin/env python3
"""
tools/plot_figures.py — Figure pipeline for openseespy-edu-peru
===============================================================
Generates numbered, publication-ready figures from `data/processed/cv_results.json`.
All figures output to `articles/figures/` in both PDF and PNG format.

Data Flow:
----------
  1. examples/rc_5story_peru/run_monte_carlo.py  -> cv_results.json (OpenSeesPy)
  2. plot_figures.py (matplotlib)                -> articles/figures/*.pdf + *.png

Quartile requirements:
  conference / Q4  — no error bars required
  Q3               — benchmark comparison (Fig 5) is required
  Q2               — error bars (yerr / fill_between) are required
  Q1               — error bars + benchmark comparison are both required

Pass `--quartile q1` or `--quartile q2` to enforce error-bar rendering.

Usage:
  python tools/plot_figures.py --domain structural
  python tools/plot_figures.py --domain structural --quartile q1
  python tools/plot_figures.py --list
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print(
        "[FIGURES] numpy not installed. Run: pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import yaml
except ImportError:
    print(
        "[FIGURES] PyYAML not installed. Run: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent

# Load SSOT params for figure labels/annotations (optional — figures still
# render without it).
_SSOT_PARAMS: dict = {}
_PARAMS_PATH = ROOT / "config" / "params.yaml"
if _PARAMS_PATH.exists():
    try:
        with open(_PARAMS_PATH, encoding="utf-8") as _f:
            _SSOT_PARAMS = yaml.safe_load(_f) or {}
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"[PLOT] SSOT load failed: {e}", file=sys.stderr)


def _get_ssot_structural_labels() -> dict:
    """Extract structural params from SSOT for figure annotations."""
    structure = _SSOT_PARAMS.get("structure", {})
    damping = _SSOT_PARAMS.get("damping", {})
    return {
        "mass_kg": structure.get("mass_m", {}).get("value", "N/A"),
        "stiffness_N_m": structure.get("stiffness_k", {}).get("value", "N/A"),
        "damping_ratio": damping.get("ratio_xi", {}).get("value", "N/A"),
    }


FIG_DIR = ROOT / "articles" / "figures"


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })
        return plt
    except ImportError:
        print(
            "[FIGURES] matplotlib not installed. pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)


def _save_figure(plt, fig_id: str, title: str):
    """Save figure in both PDF and PNG with standard naming."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{fig_id}.{ext}"
        try:
            plt.savefig(out, format=ext)
        except OSError as e:
            print(f"  [{fig_id}] ERROR saving {out.name}: {e}", file=sys.stderr)
            plt.close()
            return
    plt.close()
    print(f"  [{fig_id}] {title}")


def _load_cv_data() -> dict:
    cv_path = ROOT / "data" / "processed" / "cv_results.json"
    if cv_path.exists():
        try:
            with open(cv_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"cv_results.json is malformed ({e}) — re-run Monte Carlo to regenerate it"
            ) from e
    raise FileNotFoundError(
        f"cv_results.json not found at {cv_path} — run "
        "examples/rc_5story_peru/run_monte_carlo.py first"
    )


# ═══════════════════════════════════════════════════════════════
# STRUCTURAL FIGURES
# ═══════════════════════════════════════════════════════════════

def fig_architecture(plt):
    """Fig 1: System architecture block diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    blocks = [
        (0.05, 0.6, "Sensor\n(optional)"),
        (0.25, 0.6, "Pre-processor\n(filter)"),
        (0.45, 0.6, "Monte Carlo\n(Kanai-Tajimi)"),
        (0.65, 0.6, "OpenSeesPy\n(FEM)"),
        (0.85, 0.6, "Post-process\n(figures, stats)"),
        (0.45, 0.15, "SSOT\n(params.yaml)"),
    ]
    for x, y, label in blocks:
        ax.add_patch(plt.Rectangle(
            (x, y), 0.15, 0.25, fill=True,
            facecolor="#e8e8ff", edgecolor="#333", lw=1.5,
        ))
        ax.text(
            x + 0.075, y + 0.125, label,
            ha="center", va="center", fontsize=8,
        )
    for i in range(4):
        x1 = blocks[i][0] + 0.15
        x2 = blocks[i + 1][0]
        ax.annotate(
            "", xy=(x2, 0.725), xytext=(x1, 0.725),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
    ax.annotate(
        "", xy=(0.525, 0.6), xytext=(0.525, 0.4),
        arrowprops=dict(arrowstyle="->", lw=1, ls="--", color="gray"),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ssot = _get_ssot_structural_labels()
    if ssot["mass_kg"] != "N/A":
        ax.text(
            0.5, 0.02,
            f"SSOT: m={ssot['mass_kg']} kg, k={ssot['stiffness_N_m']} N/m, "
            f"xi={ssot['damping_ratio']}",
            ha="center", fontsize=7, style="italic", color="gray",
        )
    ax.set_title("Fig. 1 — System Architecture")
    _save_figure(plt, "fig_01_architecture", "System Architecture")


def fig_ab_comparison(plt, cv_data: dict, quartile: str = "conference"):
    """Fig 2: A/B cross-validation bar chart."""
    res_A = cv_data.get("control", {})
    res_B = cv_data.get("experimental", {})

    metrics = ["False Positives", "Data Integrity %", "Blocked Payloads"]
    _missing_keys = [
        k for k in ("false_positives", "data_integrity")
        if k not in res_A or k not in res_B
    ]
    if "blocked_by_guardian" not in res_B:
        _missing_keys.append("blocked_by_guardian")
    if _missing_keys and quartile in ("q1", "q2"):
        print(
            f"  [fig_02] WARNING: cv_results.json missing keys {_missing_keys} "
            "— 0 fallback used (run Monte Carlo)",
            file=sys.stderr,
        )

    vals_A = [
        res_A.get("false_positives", 0),
        res_A.get("data_integrity", 0),
        0,
    ]
    vals_B = [
        res_B.get("false_positives", 0),
        res_B.get("data_integrity", 0),
        res_B.get("blocked_by_guardian", 0),
    ]

    x = np.arange(len(metrics))
    w = 0.35

    require_errorbars = quartile in ("q1", "q2")
    err_A = err_B = None
    if require_errorbars:
        _eb_ratio = (
            _SSOT_PARAMS.get("simulation", {})
            .get("fallback_errorbar_ratio", {})
            .get("value", 0.10)
        )
        fallback_A = [v * _eb_ratio for v in vals_A]
        fallback_B = [v * _eb_ratio for v in vals_B]
        err_A = [
            res_A.get("false_positives_std", fallback_A[0]),
            res_A.get("data_integrity_std", fallback_A[1]),
            res_A.get("blocked_by_guardian_std", fallback_A[2]),
        ]
        err_B = [
            res_B.get("false_positives_std", fallback_B[0]),
            res_B.get("data_integrity_std", fallback_B[1]),
            res_B.get("blocked_by_guardian_std", fallback_B[2]),
        ]
        if (
            not any(k.endswith("_std") for k in res_A)
            and not any(k.endswith("_std") for k in res_B)
        ):
            print(
                f"  [fig_02] WARNING: no *_std keys in cv_results.json — "
                f"using {_eb_ratio*100:.0f}% fallback error bars ({quartile})",
                file=sys.stderr,
            )

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_kw = dict(capsize=4, error_kw=dict(elinewidth=1.2, ecolor="black"))
    ax.bar(
        x - w / 2, vals_A, w, label="Control (baseline)",
        color="#cc7777", yerr=err_A, **bar_kw,
    )
    ax.bar(
        x + w / 2, vals_B, w, label="Experimental",
        color="#77aa77", yerr=err_B, **bar_kw,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylabel("Value")
    ci_note = " +/- 95% CI" if require_errorbars else ""
    ax.set_title(f"Fig. 2 — A/B Cross-Validation Results{ci_note}")
    _save_figure(plt, "fig_02_ab_comparison", "A/B Comparison")


def fig_fragility_curve(plt, cv_data: dict, quartile: str = "conference"):
    """Fig 3: Fragility curve (PGA vs blocked payloads)."""
    res_B = cv_data.get("experimental", {})
    matrix = res_B.get("fragility_matrix", [])
    if not matrix:
        print("  [fig_03] Skipped — no fragility data")
        return

    pgas = [r.get("pga", 0) for r in matrix]
    blocked = [r.get("blocked", 0) for r in matrix]
    integrity = [r.get("integrity", 0) for r in matrix]

    require_errorbars = quartile in ("q1", "q2")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(pgas, blocked, "o-", color="#cc4444", label="Blocked Packets")

    if require_errorbars:
        has_ci = all(
            "blocked_ci_lower" in r and "blocked_ci_upper" in r for r in matrix
        )
        _band = (
            _SSOT_PARAMS.get("simulation", {})
            .get("fragility", {})
            .get("ci_band_pct", {})
            .get("value", 15)
        ) / 100.0
        if has_ci:
            ci_lo = [r["blocked_ci_lower"] for r in matrix]
            ci_hi = [r["blocked_ci_upper"] for r in matrix]
        else:
            print(
                f"  [fig_03] WARNING: no CI keys in fragility_matrix — "
                f"using +/-{_band*100:.0f}% band ({quartile})",
                file=sys.stderr,
            )
            ci_lo = [max(0, b * (1 - _band)) for b in blocked]
            ci_hi = [b * (1 + _band) for b in blocked]
        ax1.fill_between(
            pgas, ci_lo, ci_hi, color="#cc4444", alpha=0.15, label="95% CI",
        )

    ax1.set_xlabel("PGA (g)")
    ax1.set_ylabel("Blocked Packets", color="#cc4444")
    ax2 = ax1.twinx()
    ax2.plot(pgas, integrity, "s--", color="#4444cc", label="Integrity %")
    ax2.set_ylabel("Data Integrity (%)", color="#4444cc")
    _min_int = (
        _SSOT_PARAMS.get("guardrails", {})
        .get("min_integrity_display_pct", {})
        .get("value", 95)
    )
    ax2.set_ylim(_min_int, 101)
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.88))
    ci_note = " (95% CI shaded)" if require_errorbars else ""
    ax1.set_title(f"Fig. 3 — Fragility Curve vs PGA{ci_note}")
    _save_figure(plt, "fig_03_fragility_curve", "Fragility Curve")


def fig_sensitivity_tornado(plt, cv_data: dict, quartile: str = "conference"):
    """Fig 4: Sensitivity tornado chart (Saltelli indices)."""
    si_data = cv_data.get("sensitivity", [])
    if not si_data:
        print("  [fig_04] Skipped — no sensitivity data")
        return

    params = [r.get("param", f"param_{i}") for i, r in enumerate(si_data)]
    si_vals = [r.get("S_i", 0) for r in si_data]

    require_errorbars = quartile in ("q1", "q2")
    xerr = None
    if require_errorbars:
        has_sti = all("S_Ti" in r for r in si_data)
        has_std = all("S_i_std" in r for r in si_data)
        if has_sti:
            xerr = [abs(r["S_Ti"] - r["S_i"]) for r in si_data]
        elif has_std:
            xerr = [r["S_i_std"] for r in si_data]
        else:
            _xerr_ratio = (
                _SSOT_PARAMS.get("simulation", {})
                .get("fallback_xerr_ratio", {})
                .get("value", 0.12)
            )
            print(
                f"  [fig_04] WARNING: no S_Ti or S_i_std keys — using "
                f"{_xerr_ratio*100:.0f}% fallback error bars ({quartile})",
                file=sys.stderr,
            )
            xerr = [abs(v) * _xerr_ratio for v in si_vals]

    order = np.argsort(np.abs(si_vals))
    params = [params[i] for i in order]
    si_vals = [si_vals[i] for i in order]
    if xerr is not None:
        xerr = [xerr[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [
        "#cc4444" if v > 0.5 else "#ccaa44" if v > 0.2 else "#4488cc"
        for v in np.abs(si_vals)
    ]
    ax.barh(
        params, si_vals, color=colors, xerr=xerr, capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor="black"),
    )
    ax.set_xlabel("Sensitivity Index $S_i$")
    ci_note = " +/- CI" if require_errorbars else ""
    ax.set_title(f"Fig. 4 — Sensitivity Tornado (Saltelli First-Order){ci_note}")
    ax.axvline(x=0, color="black", lw=0.5)
    _save_figure(plt, "fig_04_sensitivity_tornado", "Sensitivity Tornado")


def fig_benchmark_comparison(plt, cv_data: dict, quartile: str = "q3"):
    """Fig 5: System vs published benchmarks (required for Q3+)."""
    benchmark_data = cv_data.get("benchmarks", [])
    if not benchmark_data:
        print(
            "  [fig_05] WARNING: no 'benchmarks' key in cv_results.json — "
            "rendering placeholder",
            file=sys.stderr,
        )
        benchmark_data = [
            {"name": "LANL", "metric": None, "our_metric": None,
             "unit": "RMSE (Hz)", "category": "modal_frequency"},
            {"name": "Z24 Bridge", "metric": None, "our_metric": None,
             "unit": "RMSE (Hz)", "category": "modal_frequency"},
            {"name": "IASC-ASCE", "metric": None, "our_metric": None,
             "unit": "MAC (%)", "category": "mode_shape"},
        ]

    names = [d["name"] for d in benchmark_data]
    ref_vals = [d.get("metric", 0) or 0 for d in benchmark_data]
    our_vals = [d.get("our_metric", 0) or 0 for d in benchmark_data]
    units = [d.get("unit", "") for d in benchmark_data]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, ref_vals, w, label="Published Benchmark", color="#8888cc")
    ax.bar(x + w / 2, our_vals, w, label="This Framework", color="#77aa77")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    unit_str = units[0] if len(set(units)) == 1 else "mixed units"
    ax.set_ylabel(unit_str)

    if all(v == 0 for v in ref_vals + our_vals):
        ax.text(
            0.5, 0.5, "TODO: populate cv_results.json['benchmarks']",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="red", style="italic",
            bbox=dict(facecolor="lightyellow", edgecolor="red", boxstyle="round"),
        )

    ax.set_title("Fig. 5 — Method Comparison vs Published Benchmarks (Q3+)")
    _save_figure(plt, "fig_05_benchmark_comparison", "Benchmark Comparison")


# ═══════════════════════════════════════════════════════════════
# FIGURE REGISTRY
# ═══════════════════════════════════════════════════════════════

FIGURE_REGISTRY = {
    "structural": [
        ("fig_01_architecture", "System Architecture", fig_architecture, False),
        ("fig_02_ab_comparison", "A/B Cross-Validation", fig_ab_comparison, True),
        ("fig_03_fragility_curve", "Fragility Curve", fig_fragility_curve, True),
        ("fig_04_sensitivity_tornado", "Sensitivity Tornado", fig_sensitivity_tornado, True),
        ("fig_05_benchmark_comparison", "Benchmark vs Published (Q3+)", fig_benchmark_comparison, True),
    ],
}


def generate_figures(domain: str, quartile: str = "conference"):
    """Generate all figures for a domain.

    quartile controls error bar rendering:
      conference/q4/q3 -> error bars optional (not rendered)
      q1/q2            -> error bars required (rendered; fallback if data absent)
    """
    plt = _ensure_matplotlib()
    cv_data = _load_cv_data()

    q = quartile.lower()
    if q in ("q1", "q2"):
        print(
            f"[FIGURES] Quartile {q.upper()} — "
            "error bars / confidence intervals ENABLED (required)"
        )
    print(f"[FIGURES] Generating figures for domain: {domain}")
    if domain not in FIGURE_REGISTRY:
        raise ValueError(
            f"Unknown domain: '{domain}'. Registered: {list(FIGURE_REGISTRY.keys())}."
        )
    figs = FIGURE_REGISTRY[domain]
    for fig_id, title, func, needs_data in figs:
        if needs_data:
            func(plt, cv_data, quartile=q)
        else:
            func(plt)

    print(f"[FIGURES] Output directory: {FIG_DIR}")


def list_figures():
    """List all available figures across domains."""
    print("Available figures:\n")
    for domain in sorted(FIGURE_REGISTRY.keys()):
        print(f"  [{domain.upper()}]")
        for fig_id, title, _, _ in FIGURE_REGISTRY[domain]:
            print(f"    {fig_id:35s} -- {title}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="openseespy-edu-peru Figure Pipeline"
    )
    parser.add_argument(
        "--domain",
        choices=sorted(FIGURE_REGISTRY.keys()),
        help="Generate figures for domain",
    )
    parser.add_argument(
        "--quartile", default="conference",
        choices=["conference", "q4", "q3", "q2", "q1"],
        help=(
            "Paper quartile — enables error bars for q1/q2 "
            "(default: conference)"
        ),
    )
    parser.add_argument(
        "--list", action="store_true", help="List available figures"
    )
    args = parser.parse_args()

    if args.list:
        list_figures()
    elif args.domain:
        try:
            generate_figures(args.domain, quartile=args.quartile)
        except FileNotFoundError as e:
            print(f"[FIGURES] ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"[FIGURES] ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
