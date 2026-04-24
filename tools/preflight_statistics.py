#!/usr/bin/env python
"""Preflight Statistical Gate — Q1/Q2 pre-analysis power/assumptions check.

Runs after a Monte-Carlo or IDA result file is written and BEFORE a paper's
Results/Discussion are committed. The rationale is pre-registration mentality
(Scopus Q1): confirm the experiment has sufficient power and that test
assumptions hold before interpreting the numbers.

Read-only over `data/processed/cv_results.json` (or any compatible JSON).
Does NOT mutate the file.

Verdict enum (maps to CLI exit code):
  - VIABLE                (0)  Q1/Q2 ready
  - UNDERPOWERED          (1 for Q1/Q2; 2 WARN for Q3)
  - VIOLATIONS            (1) assumption violations that require corrective action
  - INSUFFICIENT          (1) <3 runs per group — no test is valid

CLI
---
python tools/preflight_statistics.py --quartile q1 \\
    --input data/processed/cv_results.json \\
    [--alpha 0.05] [--power-target 0.80] [--effect-target 0.5] \\
    [--primary-hypotheses 1] [--correction bonferroni|holm] \\
    [--save-report] [--report-path <path>] [--seed 42]

Input JSON shape (flexible — first shape found wins):
  {"group_a": [...], "group_b": [...]}               # preferred, explicit
  {"control": [...], "experimental": [...]}          # compute_statistics style
  {"groups": {"A": [...], "B": [...]}}               # nested
  {"scenarios": {"A": {"values": [...]}, ...}}       # research_director output

Dependencies: scipy.stats, statsmodels.stats.power, numpy. All in requirements.txt.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys

# Force UTF-8 stdout/stderr on Windows so Greek letters and Unicode symbols
# survive cp1252 consoles. Scripts use ASCII markers in logs but may include
# Greek letters in markdown reports — wrap the streams defensively.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover — older Python builds
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

# Required scientific stack — declare early, fail fast.
try:
    import numpy as np
    from scipy import stats as scipy_stats
    from statsmodels.stats.power import TTestIndPower  # Q1 standard for power analysis
except ImportError as e:  # pragma: no cover — environment guard
    print(f"[ERROR] Missing dependency: {e}. Run: pip install numpy scipy statsmodels",
          file=sys.stderr)
    sys.exit(1)


# ─── Constants (cited conventions) ────────────────────────────────────────────
# Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences", 2nd ed.
# p.25 — conventional effect size thresholds.
COHEN_D_TRIVIAL = 0.2
COHEN_D_SMALL = 0.5
COHEN_D_MEDIUM = 0.8  # below this is "medium", at/above is "large"

# Fisher (1925) / Cohen (1988): α=0.05 and power=0.80 are conventional defaults
# for hypothesis testing in experimental science. Q1 reviewers expect these.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_EFFECT = 0.5  # medium effect per Cohen's conventions

# Shapiro-Wilk has excessive power for N>50 (rejects normality on trivial
# deviations). Switch to Kolmogorov-Smirnov for large samples. Threshold from
# Razali & Wah (2011) "Power comparisons of Shapiro-Wilk..." JOSMA 2(1):21-33.
NORMALITY_SW_THRESHOLD = 50

# Minimum per-group N to even attempt a two-sample test. Below this, no amount
# of statistical acrobatics produces a defensible result.
MIN_RUNS_PER_GROUP = 3

# Bootstrap resamples — 10k is the Efron & Tibshirani (1993) recommendation
# for 95% CI stability to 2 decimal places.
BOOTSTRAP_RESAMPLES = 10_000


class Verdict(str, Enum):
    VIABLE = "viable"
    UNDERPOWERED = "underpowered"
    VIOLATIONS = "assumption_violations"
    INSUFFICIENT = "insufficient_data"


@dataclass
class PowerAnalysis:
    n_group_a: int
    n_group_b: int
    effect_target: float
    power_target: float
    achieved_power: float
    required_n_per_group: int

    def passes(self) -> bool:
        return self.achieved_power >= self.power_target


@dataclass
class AssumptionChecks:
    normal_a: Optional[bool]
    normal_a_test: str
    normal_a_pvalue: float
    normal_b: Optional[bool]
    normal_b_test: str
    normal_b_pvalue: float
    homoscedastic: Optional[bool]
    levene_pvalue: float

    @property
    def parametric_viable(self) -> bool:
        return bool(self.normal_a and self.normal_b and self.homoscedastic)


@dataclass
class EffectSizeResult:
    cohens_d: float
    magnitude: str  # trivial/small/medium/large
    ci_low: float
    ci_high: float


@dataclass
class TestPreview:
    test_name: str
    statistic: float
    p_value: float
    p_value_adjusted: float
    adjustment_method: str
    significant: bool


@dataclass
class PreflightReport:
    verdict: Verdict
    quartile: str
    alpha: float
    paper_id: Optional[str]
    power: PowerAnalysis
    assumptions: AssumptionChecks
    effect_size: Optional[EffectSizeResult]
    test_preview: Optional[TestPreview]
    messages: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "quartile": self.quartile,
            "alpha": self.alpha,
            "paper_id": self.paper_id,
            "power": asdict(self.power),
            "assumptions": asdict(self.assumptions),
            "effect_size": asdict(self.effect_size) if self.effect_size else None,
            "test_preview": asdict(self.test_preview) if self.test_preview else None,
            "messages": self.messages,
            "recommendations": self.recommendations,
        }


# ─── Input loading ────────────────────────────────────────────────────────────

def load_groups(input_path: Path) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Load two groups of scalar observations from a JSON file.

    Tries multiple canonical shapes (see module docstring). Returns arrays
    (group_a, group_b, paper_id) where paper_id is optional metadata.

    Raises:
        ValueError: if no recognizable group structure is found.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    paper_id = data.get("paper_id") if isinstance(data, dict) else None

    def _flatten_to_floats(raw) -> np.ndarray:
        # Accept list of numbers, or list of dicts with 'value', or nested lists.
        if isinstance(raw, dict) and "values" in raw:
            raw = raw["values"]
        if not isinstance(raw, list):
            raise ValueError(f"Expected list, got {type(raw).__name__}")
        out = []
        for item in raw:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, dict) and "value" in item:
                out.append(float(item["value"]))
            elif isinstance(item, list):
                out.extend(float(x) for x in item if isinstance(x, (int, float)))
            else:
                raise ValueError(f"Cannot coerce {item!r} to float")
        return np.asarray(out, dtype=float)

    def _extract_observations_from_aggregate(block, label: str) -> np.ndarray:
        """Extract raw observations from an aggregate cv_results dict.

        Structural CrossValidationEngine writes aggregate stats under
        ``control`` / ``experimental`` instead of a list. We look for
        known list-valued fields (``fragility_matrix``, ``observations``,
        ``samples``) and flatten the numeric axis. If none found, raise a
        ValueError with a clear workaround hint so the caller knows this
        backend needs raw observations exported separately.
        """
        if not isinstance(block, dict):
            raise ValueError(
                f"{label!r}: expected list or aggregate dict, got "
                f"{type(block).__name__}"
            )
        # Priority 1: a fragility_matrix-style list of dicts → use "blocked"
        # counts as per-PGA observations (each PGA level is one sample).
        if "fragility_matrix" in block and isinstance(block["fragility_matrix"], list):
            fm = block["fragility_matrix"]
            # Prefer "blocked" (primary outcome), fall back to "integrity".
            for key in ("blocked", "integrity"):
                series = [row.get(key) for row in fm
                          if isinstance(row, dict) and isinstance(row.get(key), (int, float))]
                if len(series) >= 2:
                    return np.asarray(series, dtype=float)
        # Priority 2: explicit list-valued fields.
        for key in ("observations", "samples", "values", "series"):
            if key in block and isinstance(block[key], list):
                return _flatten_to_floats(block[key])
        # Priority 3: aggregate stats only — cannot run preflight without raw data.
        keys_present = sorted(
            k for k, v in block.items() if not isinstance(v, (dict, list))
        )
        raise ValueError(
            f"{label!r} is an aggregate stats dict (keys: {keys_present}) with "
            "no raw observations. preflight_statistics needs per-sample values "
            "to compute power / normality / effect size. Workaround: supply a "
            "JSON with {control: [...], experimental: [...]} where each is a "
            "flat list of observations (one per simulation run)."
        )

    # Shape 1: explicit group_a / group_b
    if isinstance(data, dict) and "group_a" in data and "group_b" in data:
        return _flatten_to_floats(data["group_a"]), _flatten_to_floats(data["group_b"]), paper_id

    # Shape 2: control / experimental (compute_statistics-compatible).
    # Accept either list-of-observations (preferred) or aggregate dict
    # (structural CrossValidationEngine schema — adapt via fragility_matrix).
    if isinstance(data, dict) and "control" in data and "experimental" in data:
        ctrl = data["control"]
        exp = data["experimental"]
        try:
            a = _flatten_to_floats(ctrl) if isinstance(ctrl, list) else \
                _extract_observations_from_aggregate(ctrl, "control")
            b = _flatten_to_floats(exp) if isinstance(exp, list) else \
                _extract_observations_from_aggregate(exp, "experimental")
        except ValueError as exc:
            # Re-raise with the full shape context so the CLI prints actionable
            # guidance. The parent try/except in main() surfaces this as
            # `[ERROR] Failed to load input: ...`.
            raise ValueError(str(exc)) from exc
        return a, b, paper_id

    # Shape 3: nested under 'groups'
    if isinstance(data, dict) and "groups" in data and isinstance(data["groups"], dict):
        keys = list(data["groups"].keys())
        if len(keys) >= 2:
            return (_flatten_to_floats(data["groups"][keys[0]]),
                    _flatten_to_floats(data["groups"][keys[1]]), paper_id)

    # Shape 4: 'scenarios' (research_director style)
    if isinstance(data, dict) and "scenarios" in data and isinstance(data["scenarios"], dict):
        keys = list(data["scenarios"].keys())
        if len(keys) >= 2:
            return (_flatten_to_floats(data["scenarios"][keys[0]]),
                    _flatten_to_floats(data["scenarios"][keys[1]]), paper_id)

    raise ValueError(
        "Unrecognized JSON shape. Expected keys: {group_a, group_b} | "
        "{control, experimental} | {groups: {...}} | {scenarios: {...}}"
    )


# ─── Statistical primitives ──────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled std (Cohen 1988 p.44, eq. 2.5.1).

    scipy/numpy provide no built-in for Cohen's d; this is the canonical
    two-sample formulation. Returns |d| so the magnitude category is stable
    regardless of group ordering.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_sd == 0:
        return 0.0
    return abs(float(np.mean(a) - np.mean(b)) / pooled_sd)


def classify_effect(d: float) -> str:
    """Cohen (1988) conventional magnitude categories."""
    ad = abs(d)
    if ad < COHEN_D_TRIVIAL:
        return "trivial"
    if ad < COHEN_D_SMALL:
        return "small"
    if ad < COHEN_D_MEDIUM:
        return "medium"
    return "large"


def bootstrap_ci_cohens_d(a: np.ndarray, b: np.ndarray, *,
                          n_resamples: int = BOOTSTRAP_RESAMPLES,
                          alpha: float = DEFAULT_ALPHA,
                          seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for Cohen's d.

    Uses np.random.default_rng(seed) for reproducibility (Engram audit trail).
    Efron & Tibshirani (1993) percentile method — simple and robust for d.
    """
    rng = np.random.default_rng(seed)
    ds = np.empty(n_resamples)
    na, nb = len(a), len(b)
    for i in range(n_resamples):
        ra = rng.choice(a, size=na, replace=True)
        rb = rng.choice(b, size=nb, replace=True)
        ds[i] = cohen_d(ra, rb)
    lo = float(np.percentile(ds, 100 * (alpha / 2)))
    hi = float(np.percentile(ds, 100 * (1 - alpha / 2)))
    return lo, hi


def check_normality(arr: np.ndarray) -> tuple[bool, str, float]:
    """Normality check. Shapiro-Wilk for N<50, Kolmogorov-Smirnov otherwise.

    Returns (is_normal, test_name, p_value). Uses α=0.05: p >= α → retain H0
    (normal). Switch at N=50 avoids Shapiro's excessive power on large N
    (Razali & Wah 2011).
    """
    n = len(arr)
    if n < 3:
        return False, "insufficient-N", 0.0
    if n < NORMALITY_SW_THRESHOLD:
        stat, p = scipy_stats.shapiro(arr)
        return bool(p >= DEFAULT_ALPHA), "shapiro-wilk", float(p)
    # K-S against fitted normal — use mean/std of the sample
    # (scipy.stats.kstest with 'norm' requires specifying loc/scale).
    loc, scale = float(np.mean(arr)), float(np.std(arr, ddof=1))
    if scale == 0:
        return False, "kolmogorov-smirnov", 0.0
    stat, p = scipy_stats.kstest(arr, "norm", args=(loc, scale))
    return bool(p >= DEFAULT_ALPHA), "kolmogorov-smirnov", float(p)


def check_homoscedasticity(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    """Levene's test (robust to non-normality). p>=α → homoscedastic."""
    if len(a) < 2 or len(b) < 2:
        return False, 0.0
    stat, p = scipy_stats.levene(a, b, center="median")
    return bool(p >= DEFAULT_ALPHA), float(p)


def power_analysis(a: np.ndarray, b: np.ndarray, *,
                   effect_target: float,
                   power_target: float,
                   alpha: float) -> PowerAnalysis:
    """Two-sample independent t-test power via statsmodels.

    scipy has no built-in power analysis. statsmodels.TTestIndPower is the
    Q1 reference implementation (matches G*Power for two-sample t-tests).
    """
    analyzer = TTestIndPower()
    n_a, n_b = len(a), len(b)
    # Use the smaller group (conservative — the achievable power with the
    # least-represented cell governs the design).
    n_min = min(n_a, n_b) if min(n_a, n_b) > 1 else 2
    ratio = n_b / n_a if n_a > 0 else 1.0
    try:
        achieved = float(analyzer.solve_power(effect_size=effect_target,
                                              nobs1=n_min, alpha=alpha,
                                              ratio=ratio))
    except Exception:
        achieved = 0.0
    try:
        required = analyzer.solve_power(effect_size=effect_target, power=power_target,
                                        alpha=alpha, ratio=1.0)
        required_n = int(math.ceil(float(required))) if required else 0
    except Exception:
        required_n = 0
    return PowerAnalysis(
        n_group_a=n_a, n_group_b=n_b,
        effect_target=effect_target, power_target=power_target,
        achieved_power=achieved, required_n_per_group=required_n,
    )


def multiple_comparisons_adjust(p: float, n_hypotheses: int, method: str) -> float:
    """Bonferroni (Dunn 1961) or Holm (Holm 1979) step-down on a SINGLE
    primary p-value. For the preflight we only adjust the headline p for
    exposition — the full Holm step-down requires the full p-value family,
    which the preflight doesn't run (VERIFY does). We are conservative here:
    Holm-at-rank-1 == Bonferroni, so if method=holm we return the same value
    and note the equivalence in the report.
    """
    if n_hypotheses <= 1:
        return p
    if method in ("bonferroni", "holm"):
        return min(1.0, p * n_hypotheses)
    return p


# ─── Verdict computation ──────────────────────────────────────────────────────

def compute_preflight(a: np.ndarray, b: np.ndarray, *,
                      quartile: str,
                      alpha: float,
                      power_target: float,
                      effect_target: float,
                      n_hypotheses: int,
                      correction: str,
                      seed: int,
                      paper_id: Optional[str]) -> PreflightReport:
    """Run the full preflight pipeline and emit a PreflightReport."""
    messages: list[str] = []
    recs: list[str] = []

    # ── Gate 0: insufficient data (blocks everything) ──
    if len(a) < MIN_RUNS_PER_GROUP or len(b) < MIN_RUNS_PER_GROUP:
        power = PowerAnalysis(n_group_a=len(a), n_group_b=len(b),
                              effect_target=effect_target, power_target=power_target,
                              achieved_power=0.0, required_n_per_group=0)
        assumptions = AssumptionChecks(normal_a=None, normal_a_test="n/a", normal_a_pvalue=0.0,
                                       normal_b=None, normal_b_test="n/a", normal_b_pvalue=0.0,
                                       homoscedastic=None, levene_pvalue=0.0)
        messages.append(
            f"INSUFFICIENT DATA: need >={MIN_RUNS_PER_GROUP} runs/group "
            f"(got {len(a)}/{len(b)}). No statistical test is valid."
        )
        recs.append("Run more simulations / scenarios until each group has "
                    f">={MIN_RUNS_PER_GROUP} observations.")
        return PreflightReport(verdict=Verdict.INSUFFICIENT, quartile=quartile,
                               alpha=alpha, paper_id=paper_id, power=power,
                               assumptions=assumptions, effect_size=None,
                               test_preview=None, messages=messages,
                               recommendations=recs)

    # ── Power analysis ──
    power = power_analysis(a, b, effect_target=effect_target,
                           power_target=power_target, alpha=alpha)

    # ── Assumption checks ──
    normal_a, test_a, p_na = check_normality(a)
    normal_b, test_b, p_nb = check_normality(b)
    homo, p_lev = check_homoscedasticity(a, b)
    assumptions = AssumptionChecks(
        normal_a=normal_a, normal_a_test=test_a, normal_a_pvalue=p_na,
        normal_b=normal_b, normal_b_test=test_b, normal_b_pvalue=p_nb,
        homoscedastic=homo, levene_pvalue=p_lev,
    )

    # ── Effect size (Cohen's d + bootstrap CI) ──
    d = cohen_d(a, b)
    ci_lo, ci_hi = bootstrap_ci_cohens_d(a, b, alpha=alpha, seed=seed)
    effect = EffectSizeResult(cohens_d=d, magnitude=classify_effect(d),
                              ci_low=ci_lo, ci_high=ci_hi)

    # ── Test selection + preview p-value ──
    if assumptions.parametric_viable:
        t_stat, p_val = scipy_stats.ttest_ind(a, b, equal_var=True)
        test_name = "independent t-test"
        stat_value = float(t_stat)
        messages.append("Normality + homoscedasticity OK → parametric t-test.")
    else:
        u_stat, p_val = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Mann-Whitney U"
        stat_value = float(u_stat)
        violations = []
        if not normal_a: violations.append(f"group_a not normal ({test_a} p={p_na:.4f})")
        if not normal_b: violations.append(f"group_b not normal ({test_b} p={p_nb:.4f})")
        if not homo:     violations.append(f"heteroscedastic (Levene p={p_lev:.4f})")
        messages.append("Assumption violation(s): " + "; ".join(violations)
                        + " → falling back to Mann-Whitney U.")

    p_adj = multiple_comparisons_adjust(float(p_val), n_hypotheses, correction)
    method_name = correction if n_hypotheses > 1 else "none"
    preview = TestPreview(test_name=test_name, statistic=stat_value,
                          p_value=float(p_val), p_value_adjusted=p_adj,
                          adjustment_method=method_name,
                          significant=p_adj < alpha)

    # ── Verdict decision ──
    q = quartile.lower()
    has_violations = not assumptions.parametric_viable
    # For the preflight, assumption violations only BLOCK if the quartile is
    # Q1 AND the fallback non-parametric path itself is underpowered.
    # Q2/Q3/lower can proceed with Mann-Whitney U without blocking.

    if not power.passes():
        if q in ("q1", "q2"):
            messages.append(
                f"UNDERPOWERED: achieved power {power.achieved_power:.2f} < "
                f"target {power_target:.2f} at d={effect_target}. "
                f"Required N/group: {power.required_n_per_group}."
            )
            recs.append(f"Increase N to >={power.required_n_per_group} per group.")
            recs.append("OR raise --effect-target to a larger d if scientifically "
                        "justifiable (larger effects need smaller N).")
            if q == "q2":
                recs.append("OR downgrade to Q3 (power>=0.80 is recommended but "
                            "not strictly mandated).")
            return PreflightReport(verdict=Verdict.UNDERPOWERED, quartile=quartile,
                                   alpha=alpha, paper_id=paper_id, power=power,
                                   assumptions=assumptions, effect_size=effect,
                                   test_preview=preview, messages=messages,
                                   recommendations=recs)
        # Q3 and below: WARN but do not block (handled by CLI exit code 2)
        messages.append(
            f"WARN: power {power.achieved_power:.2f} < {power_target:.2f}; "
            f"Q3 accepts this with disclosure but Q1/Q2 would not."
        )
        recs.append("Disclose the power limitation in the Limitations section.")
        return PreflightReport(verdict=Verdict.UNDERPOWERED, quartile=quartile,
                               alpha=alpha, paper_id=paper_id, power=power,
                               assumptions=assumptions, effect_size=effect,
                               test_preview=preview, messages=messages,
                               recommendations=recs)

    # Power OK. Check assumptions downstream consequence.
    if has_violations and q == "q1":
        # Q1 reviewers expect you to justify the test choice explicitly.
        # The preflight does NOT block here — the non-parametric fallback is a
        # valid choice — but it records the decision for the narrator to cite.
        messages.append("Parametric assumptions violated → non-parametric "
                        "fallback will be used. Document this in Methods.")
        recs.append("In Methods, cite the assumption test results and the "
                    "rationale for Mann-Whitney U.")

    return PreflightReport(verdict=Verdict.VIABLE, quartile=quartile,
                           alpha=alpha, paper_id=paper_id, power=power,
                           assumptions=assumptions, effect_size=effect,
                           test_preview=preview, messages=messages,
                           recommendations=recs)


# ─── Report rendering ─────────────────────────────────────────────────────────

def render_markdown(report: PreflightReport, *, input_path: Path) -> str:
    """Render a reviewer-friendly markdown report."""
    v = report.verdict
    icon = {"viable": "[PASS]", "underpowered": "[FAIL]",
            "assumption_violations": "[FAIL]", "insufficient_data": "[FAIL]"}[v.value]

    lines: list[str] = []
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"# Preflight Statistical Analysis — {report.quartile.upper()}")
    lines.append(f"Generated: {ts}")
    lines.append(f"Input: {input_path}")
    if report.paper_id:
        lines.append(f"Paper ID: {report.paper_id}")
    lines.append("")
    lines.append(f"## Verdict: {v.value.upper()} {icon}")
    lines.append("")

    # Power
    p = report.power
    lines.append("## Power Analysis")
    lines.append(f"- N available: {p.n_group_a} (group A) / {p.n_group_b} (group B)")
    lines.append(f"- Target effect size (Cohen's d): {p.effect_target}")
    lines.append(f"- Target power (Fisher/Cohen convention): {p.power_target}")
    lines.append(f"- Achieved power: {p.achieved_power:.3f}")
    lines.append(f"- Required N/group for target: {p.required_n_per_group}")
    lines.append("")

    # Assumptions
    a = report.assumptions
    lines.append("## Assumption Checks")
    if a.normal_a is None:
        lines.append("- Normality: skipped (insufficient data)")
    else:
        mark_a = "OK" if a.normal_a else "FAIL"
        mark_b = "OK" if a.normal_b else "FAIL"
        mark_h = "OK" if a.homoscedastic else "FAIL"
        lines.append(f"- Normality A ({a.normal_a_test}): p={a.normal_a_pvalue:.4f} [{mark_a}]")
        lines.append(f"- Normality B ({a.normal_b_test}): p={a.normal_b_pvalue:.4f} [{mark_b}]")
        lines.append(f"- Homoscedasticity (Levene): p={a.levene_pvalue:.4f} [{mark_h}]")
        chosen = "parametric (t-test)" if a.parametric_viable else "non-parametric (Mann-Whitney U)"
        lines.append(f"- Decision: use {chosen}")
    lines.append("")

    # Effect size
    if report.effect_size is not None:
        es = report.effect_size
        lines.append("## Effect Size (primary hypothesis)")
        lines.append(f"- Cohen's d = {es.cohens_d:.3f} ({es.magnitude})")
        lines.append(f"- 95% CI (bootstrap {BOOTSTRAP_RESAMPLES}): "
                     f"[{es.ci_low:.3f}, {es.ci_high:.3f}]")
        lines.append("")

    # Preview
    if report.test_preview is not None:
        t = report.test_preview
        lines.append("## Test Result (preview — VERIFY runs the authoritative test)")
        lines.append(f"- {t.test_name}: statistic={t.statistic:.3f}, p={t.p_value:.4f}")
        if t.adjustment_method != "none":
            lines.append(f"- Adjusted ({t.adjustment_method}): p={t.p_value_adjusted:.4f}")
        lines.append(f"- Significant at α={report.alpha}: "
                     f"{'yes' if t.significant else 'no'}")
        lines.append("")

    # Messages / recommendations
    if report.messages:
        lines.append("## Notes")
        for m in report.messages:
            lines.append(f"- {m}")
        lines.append("")
    if report.recommendations:
        lines.append("## Recommendations")
        for r in report.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def verdict_to_exit_code(verdict: Verdict, quartile: str) -> int:
    """Exit code contract used by generate_compute_manifest and preflight_check."""
    q = quartile.lower()
    if verdict == Verdict.VIABLE:
        return 0
    if verdict == Verdict.UNDERPOWERED and q == "q3":
        return 2  # WARN
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="C5.5 preflight statistical gate (F09 fix). Q1/Q2 pre-IMPLEMENT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to JSON with group observations "
                         "(data/processed/cv_results.json or test fixture)")
    ap.add_argument("--quartile", required=True,
                    choices=["conference", "q1", "q2", "q3", "q4"],
                    help="Target quartile (governs blocking behavior)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help=f"Significance level (default {DEFAULT_ALPHA})")
    ap.add_argument("--power-target", type=float, default=DEFAULT_POWER,
                    help=f"Target statistical power (default {DEFAULT_POWER}, Fisher/Cohen)")
    ap.add_argument("--effect-target", type=float, default=DEFAULT_EFFECT,
                    help=f"Target effect size Cohen's d (default {DEFAULT_EFFECT}, medium)")
    ap.add_argument("--primary-hypotheses", type=int, default=1,
                    help="Number of primary hypotheses (for multiple-comparisons adjust)")
    ap.add_argument("--correction", default="bonferroni",
                    choices=["bonferroni", "holm"],
                    help="Multiple-comparisons correction method")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for bootstrap (reproducibility)")
    ap.add_argument("--save-report", action="store_true",
                    help="Write markdown report to data/processed/ (or --report-path)")
    ap.add_argument("--report-path", type=Path, default=None,
                    help="Explicit report output path (implies --save-report)")
    ap.add_argument("--json", action="store_true",
                    help="Also emit JSON verdict to stdout (machine-readable)")
    args = ap.parse_args(argv)

    try:
        a, b, paper_id = load_groups(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to load input: {e}", file=sys.stderr)
        return 1

    report = compute_preflight(
        a, b,
        quartile=args.quartile,
        alpha=args.alpha,
        power_target=args.power_target,
        effect_target=args.effect_target,
        n_hypotheses=max(1, args.primary_hypotheses),
        correction=args.correction,
        seed=args.seed,
        paper_id=paper_id,
    )

    md = render_markdown(report, input_path=args.input)
    print(md)

    if args.save_report or args.report_path is not None:
        out = args.report_path
        if out is None:
            # Default: data/processed/preflight_stats_report.md relative to cwd
            out = Path("data") / "processed" / "preflight_stats_report.md"
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n[OK] Report saved: {out}")

    if args.json:
        print("\n---JSON---")
        print(json.dumps(report.to_dict(), indent=2))

    return verdict_to_exit_code(report.verdict, args.quartile)


if __name__ == "__main__":
    sys.exit(main())
