"""
Params (Runtime YAML) — SSOT loader for simulation parameters.

Reads `config/params.yaml` fresh on every import so that mass, stiffness,
damping and Newmark-integrator values always come from the Single Source of
Truth, never from hardcoded constants.

Pipeline: COMPUTE C0 (SSOT sanity) and C2 (OpenSeesPy model build).
Depends on: config/params.yaml, openseespy, pyyaml.
Produces: dict `P` with {mass, k, fy, xi, integrator, gamma, beta};
          function `init_model()` for a 1-DOF oscillator smoke test.

Path resolution — the repo root is the first ancestor of this file that
contains `config/params.yaml`. No external dependency on a belico.yaml
marker: this framework stands alone.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` until we find `config/params.yaml`.

    Falls back to the grandparent of this file (canonical layout) if the
    marker is not found — callers get a clear FileNotFoundError downstream
    when they try to read the SSOT.
    """
    current = (start or Path(__file__)).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "config" / "params.yaml").exists():
            return parent
    # Canonical layout: src/physics/models/params.py -> repo root = 3 levels up
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _find_repo_root()


def get_params_file() -> Path:
    """Return the absolute path to the SSOT YAML file."""
    return REPO_ROOT / "config" / "params.yaml"


def load_sim_params() -> dict:
    """Load the canonical subset of simulation parameters from the SSOT.

    Returns a flat dict used by `init_model()` and by the smoke-test
    examples. Production runners read the YAML directly and consume the
    full SSOT (geometry, materials, code factors, ground-motion config).
    """
    try:
        with open(get_params_file(), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Newmark integrator params from SSOT with documented fallbacks.
        nl_analysis = cfg.get("nonlinear", {}).get("analysis", {})
        integrator = nl_analysis.get("integrator", {}).get("value", "Newmark")
        gamma = nl_analysis.get("gamma", {}).get("value", 0.5)
        beta = nl_analysis.get("beta", {}).get("value", 0.25)

        return {
            "mass": cfg["structure"]["mass_m"]["value"],
            "k": cfg["structure"]["stiffness_k"]["value"],
            "fy": cfg["material"]["yield_strength_fy"]["value"],
            "xi": cfg["damping"]["ratio_xi"]["value"],
            "integrator": integrator if integrator else "Newmark",
            "integrator_gamma": gamma if gamma is not None else 0.5,
            "integrator_beta": beta if beta is not None else 0.25,
        }
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        raise RuntimeError(
            f"SSOT load failed: {e}. "
            "Verify config/params.yaml exists and contains structure/material/damping blocks."
        ) from e


P = load_sim_params()


def init_model():
    """Initialize a 1-DOF oscillator in OpenSeesPy (smoke test only)."""
    try:
        import openseespy.opensees as ops
    except ImportError as e:
        raise ImportError(
            "openseespy is required to run init_model(). "
            "Install it: pip install openseespy==3.6.0"
        ) from e

    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Nodes: base fixed, mass lumped on free node.
    ops.node(1, 0.0)
    ops.node(2, 0.0)
    ops.fix(1, 1)

    # Elastic spring.
    ops.uniaxialMaterial("Elastic", 1, P["k"])
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

    # Mass.
    ops.mass(2, P["mass"])

    # Basic transient analysis setup.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator(P["integrator"], P["integrator_gamma"], P["integrator_beta"])
    ops.algorithm("Newton")
    ops.analysis("Transient")

    print(f"[OPENSEES] 1-DOF model initialized (m={P['mass']}kg, k={P['k']}N/m)")
