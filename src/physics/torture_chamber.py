"""
src/physics/torture_chamber.py — RC column model (Linear fallback / Nonlinear P-Delta)
======================================================================================
A dual-mode column model driven by the SSOT in `config/params.yaml`:

  LINEAR (fallback):
    elasticBeamColumn + Elastic material.
    Used when nonlinear parameters are null (factory template / no project data).

  NONLINEAR (production):
    forceBeamColumn + Concrete02 + Steel02 + fiber section.
    Activated when all required nonlinear parameters are populated.
    Captures cracking, crushing, yielding, cyclic degradation and P-Delta.

All parameters come from `config/params.yaml`. Never hardcode physical
values in this module.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Python version guard — OpenSeesPy requires Python 3.8-3.12 on Windows.
# ---------------------------------------------------------------------------
if sys.platform == "win32" and sys.version_info >= (3, 13):
    raise RuntimeError(
        "[TORTURE] OpenSeesPy requires Python 3.8-3.12 on Windows "
        f"(current: {sys.version_info.major}.{sys.version_info.minor}).\n"
        "  Run with: py -3.12 src/physics/torture_chamber.py\n"
        "  Or install Python 3.12: https://www.python.org/downloads/"
    )

# ---------------------------------------------------------------------------
# Mathematical model constants — named with citations (never hardcoded physics).
# Newmark (1959) average-acceleration: beta=1/4, gamma=1/2 — unconditionally stable.
# ---------------------------------------------------------------------------
_NEWMARK_BETA_LINEAR = 0.25     # Newmark 1959, average acceleration (beta = 1/4)
_NEWMARK_GAMMA_LINEAR = 0.50    # Newmark 1959, average acceleration (gamma = 1/2)
_G_MPS2 = 9.81                  # Standard gravity, m/s^2 (BIPM, exact SI definition)

# Factory defaults for geometry when nonlinear.geometry.* is absent from SSOT.
# Used only by the LINEAR fallback model (template mode / no project data yet).
_FACTORY_L_M = 3.0              # m — typical RC column height (factory placeholder)
_FACTORY_B_M = 0.25             # m — square section width (factory placeholder)

# Fiber discretization defaults — numerical method parameters, not physics.
_DEFAULT_N_FIBER_CORE = 10      # fibers per direction in confined core patch
_DEFAULT_N_FIBER_COVER = 2      # fibers in unconfined cover patches

_CONCRETE02_LAMBDA = 0.1        # Concrete02 unloading/reloading stiffness recovery
_ANALYSIS_TOL_DISP = 1.0e-6     # Displacement convergence tolerance (NormDispIncr)
_AXIAL_LOAD_PCR_RATIO = 0.90    # applied axial load = 0.9 * Pcr (typical gravity, ACI 318-19)
_INERTIA_DIVISOR_SQUARE = 12.0  # I = b^4/12 — second moment for square cross-section

try:
    import yaml
except ImportError as _exc:
    raise ImportError(
        "[TORTURE] PyYAML not installed. Run: pip install pyyaml"
    ) from _exc

try:
    import openseespy.opensees as ops
except ImportError as _exc:
    raise ImportError(
        "[TORTURE] openseespy not installed. Run: pip install openseespy==3.6.0"
    ) from _exc

# Import the SSOT path resolver from the models package (no external
# dependency on config.paths — this framework stands alone).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.physics.models.params import get_params_file  # noqa: E402


def _load_ssot() -> dict:
    """Load the full raw SSOT dict from `config/params.yaml`."""
    params_path = get_params_file()
    try:
        with open(params_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(
            f"[TORTURE] ERROR: params.yaml not found at {params_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[TORTURE] ERROR: params.yaml malformed: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[TORTURE] ERROR: cannot read params.yaml: {e}", file=sys.stderr)
        sys.exit(1)


def _get_nested(cfg: dict, dotpath: str):
    """Get a nested value by dotpath; returns None if any key is absent."""
    keys = dotpath.split(".")
    current = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, dict):
        return current.get("value")
    return current


def _require(cfg: dict, dotpath: str):
    """Get a required SSOT value; sys.exit(1) with diagnostics if absent."""
    val = _get_nested(cfg, dotpath)
    if val is None:
        print(
            f"[TORTURE] ERROR: SSOT missing required key '{dotpath}' "
            "in config/params.yaml",
            file=sys.stderr,
        )
        sys.exit(1)
    return val


def _nonlinear_ready(cfg: dict) -> bool:
    """Check that all required nonlinear parameters are populated (non-null)."""
    nl = cfg.get("nonlinear")
    if nl is None:
        return False

    for section_key in ("concrete", "steel", "section", "geometry"):
        section = nl.get(section_key, {})
        for _param_key, param in section.items():
            if isinstance(param, dict) and param.get("required", False):
                if param.get("value") is None:
                    return False
    return True


def _build_fiber_section(cfg: dict, sec_tag: int = 1):
    """Build a fiber section with confined/unconfined concrete and steel layers."""
    # Concrete properties.
    fc = float(_require(cfg, "material.yield_strength_fy"))
    epsc0 = float(_require(cfg, "nonlinear.concrete.epsc0"))
    fpcu_ratio = float(_require(cfg, "nonlinear.concrete.fpcu_ratio"))
    epsU = float(_require(cfg, "nonlinear.concrete.epsU"))
    ft_ratio = float(_require(cfg, "nonlinear.concrete.ft_ratio"))
    Ets = float(_require(cfg, "nonlinear.concrete.Ets"))
    conf_ratio = float(_require(cfg, "nonlinear.concrete.confinement_ratio"))

    # Steel properties.
    fy_steel = float(_require(cfg, "nonlinear.steel.fy"))
    Es_steel = float(_require(cfg, "nonlinear.steel.Es"))
    b_hard = float(_require(cfg, "nonlinear.steel.b_hardening"))
    R0 = float(_require(cfg, "nonlinear.steel.R0"))
    cR1 = float(_require(cfg, "nonlinear.steel.cR1"))
    cR2 = float(_require(cfg, "nonlinear.steel.cR2"))

    # Section geometry.
    b = float(_require(cfg, "nonlinear.geometry.b"))
    cover = float(_require(cfg, "nonlinear.section.cover"))
    n_bars = int(_require(cfg, "nonlinear.section.n_bars_face"))
    bar_dia = float(_require(cfg, "nonlinear.section.bar_diameter"))
    bar_area = math.pi * (bar_dia / 2.0) ** 2

    # Fiber discretization — SSOT if present, else documented defaults.
    n_fiber_core = int(
        _get_nested(cfg, "nonlinear.section.n_fiber_core") or _DEFAULT_N_FIBER_CORE
    )
    n_fiber_cover = int(
        _get_nested(cfg, "nonlinear.section.n_fiber_cover") or _DEFAULT_N_FIBER_COVER
    )

    # Derived.
    fpc_conf = fc * conf_ratio
    epsc0_conf = epsc0 * conf_ratio  # Mander approximation
    fpcu_conf = fpcu_ratio * fpc_conf
    epsU_conf = epsU * conf_ratio
    ft = ft_ratio * fc
    fpcu_unconf = fpcu_ratio * fc

    # Material tags.
    MAT_CONF = 10     # Confined concrete (core)
    MAT_UNCONF = 20   # Unconfined concrete (cover)
    MAT_STEEL = 30    # Reinforcing steel

    ops.uniaxialMaterial(
        "Concrete02", MAT_CONF,
        -fpc_conf, -epsc0_conf, -fpcu_conf, -epsU_conf,
        _CONCRETE02_LAMBDA, ft, Ets,
    )
    ops.uniaxialMaterial(
        "Concrete02", MAT_UNCONF,
        -fc, -epsc0, -fpcu_unconf, -epsU,
        _CONCRETE02_LAMBDA, ft, Ets,
    )
    ops.uniaxialMaterial(
        "Steel02", MAT_STEEL,
        fy_steel, Es_steel, b_hard, R0, cR1, cR2,
    )

    # Fiber section patches.
    core_b = b - 2.0 * cover
    half_b = b / 2.0
    half_core = core_b / 2.0

    ops.section("Fiber", sec_tag)

    # Core concrete (confined).
    ops.patch(
        "rect", MAT_CONF, n_fiber_core, n_fiber_core,
        -half_core, -half_core, half_core, half_core,
    )

    # Cover concrete (unconfined) — four patches around the core.
    ops.patch(
        "rect", MAT_UNCONF, n_fiber_cover, n_fiber_core,
        -half_b, -half_b, half_b, -half_core,
    )
    ops.patch(
        "rect", MAT_UNCONF, n_fiber_cover, n_fiber_core,
        -half_b, half_core, half_b, half_b,
    )
    ops.patch(
        "rect", MAT_UNCONF, n_fiber_core, n_fiber_cover,
        -half_b, -half_core, -half_core, half_core,
    )
    ops.patch(
        "rect", MAT_UNCONF, n_fiber_core, n_fiber_cover,
        half_core, -half_core, half_b, half_core,
    )

    # Steel layers — top and bottom faces.
    y_steel = half_core
    ops.layer(
        "straight", MAT_STEEL, n_bars, bar_area,
        -y_steel, -y_steel, -y_steel, y_steel,
    )
    ops.layer(
        "straight", MAT_STEEL, n_bars, bar_area,
        y_steel, -y_steel, y_steel, y_steel,
    )

    print(
        f"[OPENSEES]   Fiber section built: core={core_b:.3f}m, "
        f"fc_conf={fpc_conf/1e6:.1f}MPa, fy_steel={fy_steel/1e6:.0f}MPa, "
        f"{2*n_bars} bars dia={bar_dia*1000:.0f}mm"
    )

    return {
        "MAT_CONF": MAT_CONF,
        "MAT_UNCONF": MAT_UNCONF,
        "MAT_STEEL": MAT_STEEL,
        "sec_tag": sec_tag,
    }


def init_model() -> dict:
    """Initialize the column model from the SSOT.

    Automatically selects the linear fallback or the nonlinear fiber model
    based on SSOT completeness.
    """
    cfg = _load_ssot()
    use_nonlinear = _nonlinear_ready(cfg)

    # Material properties.
    E = float(_require(cfg, "material.elastic_modulus_E"))
    fy = float(_require(cfg, "material.yield_strength_fy"))
    rho = float(_require(cfg, "material.density"))

    # Structure.
    m = float(_require(cfg, "structure.mass_m"))
    k = float(_require(cfg, "structure.stiffness_k"))

    # Damping.
    xi = float(_require(cfg, "damping.ratio_xi"))

    mode = (
        "NONLINEAR (Concrete02+Steel02+Fiber)"
        if use_nonlinear
        else "LINEAR (elastic fallback)"
    )
    print(f"[OPENSEES] Torture Chamber — {mode}")
    print(f"[OPENSEES]   E={E/1e9:.1f}GPa  fc={fy/1e6:.1f}MPa  rho={rho:.0f}kg/m3")
    print(f"[OPENSEES]   m={m:.0f}kg  k={k:.0f}N/m  xi={xi:.1%}")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    if use_nonlinear:
        return _init_nonlinear(cfg, E, fy, rho, m, k, xi)
    return _init_linear(cfg, E, fy, rho, m, k, xi)


def _init_linear(
    cfg: dict, E: float, fy: float, rho: float, m: float, k: float, xi: float
) -> dict:
    """Linear elastic model (factory fallback — no project data yet)."""
    # Geometry — SSOT first; factory defaults only if absent.
    L_ssot = _get_nested(cfg, "nonlinear.geometry.L")
    b_ssot = _get_nested(cfg, "nonlinear.geometry.b")
    if L_ssot is not None and b_ssot is not None:
        L = float(L_ssot)
        b = float(b_ssot)
    else:
        L = _FACTORY_L_M
        b = _FACTORY_B_M
        print(
            f"[TORTURE] WARNING: Using factory geometry defaults "
            f"L={_FACTORY_L_M}m b={_FACTORY_B_M}m — set nonlinear.geometry.L/b "
            "in params.yaml for project geometry",
            file=sys.stderr,
        )
    A = b * b
    I = b**4 / _INERTIA_DIVISOR_SQUARE

    print(f"[OPENSEES]   L={L:.1f}m  b={b:.2f}m  A={A:.4f}m2  I={I:.6e}m4")

    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, L)
    ops.fix(1, 1, 1, 1)

    ops.geomTransf("PDelta", 1)
    ops.uniaxialMaterial("Elastic", 1, E)
    ops.element("elasticBeamColumn", 1, 1, 2, A, E, I, 1)

    ops.mass(2, m, m, 0.0)

    # Axial load near Pcr (cantilever K=2: Pcr = pi^2*E*I / (4*L^2)).
    Pcr = math.pi**2 * E * I / (4.0 * L**2)
    P_applied = -_AXIAL_LOAD_PCR_RATIO * Pcr
    print(
        f"[OPENSEES]   Pcr={Pcr:.0f}N  P_applied={abs(P_applied):.0f}N (90% Pcr)"
    )

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, P_applied, 0.0)

    # Static gravity.
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1.0e-8, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ops.analyze(1)

    ops.loadConst("-time", 0.0)

    # Dynamic setup.
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)

    # Rayleigh damping — mass-proportional only (linear model).
    wn = math.sqrt(k / m)
    a0 = xi * 2.0 * wn
    ops.rayleigh(a0, 0.0, 0.0, 0.0)

    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.test("NormDispIncr", _ANALYSIS_TOL_DISP, 20)
    ops.algorithm("Newton")
    ops.integrator("Newmark", _NEWMARK_GAMMA_LINEAR, _NEWMARK_BETA_LINEAR)
    ops.analysis("Transient")

    print(
        f"[OPENSEES] Linear model ready. wn={wn:.2f}rad/s  "
        f"fn={wn/(2*math.pi):.2f}Hz"
    )

    return {
        "mass_kg": m,
        "E_pa": E,
        "fy_pa": fy,
        "I_m4": I,
        "A_m2": A,
        "L_m": L,
        "b_m": b,
        "xi": xi,
        "Pcr_N": Pcr,
        "wn_rad": wn,
        "nonlinear": False,
        "n_elements": 1,
        "top_node": 2,
    }


def _init_nonlinear(
    cfg: dict, E: float, fy: float, rho: float, m: float, k: float, xi: float
) -> dict:
    """Nonlinear fiber model with Concrete02 + Steel02 + P-Delta."""
    # Geometry from SSOT.
    L = float(_require(cfg, "nonlinear.geometry.L"))
    b = float(_require(cfg, "nonlinear.geometry.b"))
    n_elem = int(_require(cfg, "nonlinear.geometry.n_elements"))
    n_ip = int(_require(cfg, "nonlinear.section.n_integration_pts"))
    A = b * b
    I = b**4 / _INERTIA_DIVISOR_SQUARE

    print(
        f"[OPENSEES]   L={L:.1f}m  b={b:.2f}m  n_elem={n_elem}  n_ip={n_ip}"
    )

    # Nodes along column height.
    for i in range(n_elem + 1):
        node_tag = i + 1
        y_coord = i * L / n_elem
        ops.node(node_tag, 0.0, y_coord)

    ops.fix(1, 1, 1, 1)

    ops.geomTransf("PDelta", 1)
    sec_info = _build_fiber_section(cfg, sec_tag=1)

    # Force-based beam-column elements with fiber section.
    for i in range(n_elem):
        elem_tag = i + 1
        node_i = i + 1
        node_j = i + 2
        ops.beamIntegration("Lobatto", elem_tag, sec_info["sec_tag"], n_ip)
        ops.element("forceBeamColumn", elem_tag, node_i, node_j, 1, elem_tag)

    # Mass at top node.
    top_node = n_elem + 1
    ops.mass(top_node, m, m, 0.0)

    # Axial load (gravity).
    Pcr = math.pi**2 * E * I / (4.0 * L**2)
    P_applied = -_AXIAL_LOAD_PCR_RATIO * Pcr
    print(
        f"[OPENSEES]   Pcr={Pcr:.0f}N  P_applied={abs(P_applied):.0f}N (90% Pcr)"
    )

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(top_node, 0.0, P_applied, 0.0)

    # Static gravity analysis.
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.test("NormDispIncr", _ANALYSIS_TOL_DISP, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    ops.analyze(10)

    ops.loadConst("-time", 0.0)

    # Rayleigh damping — mass + committed-stiffness proportional.
    wn = math.sqrt(k / m)
    w1 = wn
    w2 = 3.0 * wn  # 2nd mode approximation for cantilever
    a0 = xi * 2.0 * w1 * w2 / (w1 + w2)
    a1 = xi * 2.0 / (w1 + w2)
    ops.rayleigh(a0, 0.0, a1, 0.0)
    print(
        f"[OPENSEES]   Rayleigh: a0={a0:.4f} (mass), a1={a1:.6f} (stiffness)"
    )

    # Dynamic analysis setup.
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.test("NormDispIncr", _ANALYSIS_TOL_DISP, 30)
    ops.algorithm("Newton")

    beta = float(_require(cfg, "nonlinear.analysis.beta"))
    gamma = float(_require(cfg, "nonlinear.analysis.gamma"))
    ops.integrator("Newmark", gamma, beta)
    ops.analysis("Transient")

    print(
        f"[OPENSEES] Nonlinear model ready. wn={wn:.2f}rad/s  "
        f"fn={wn/(2*math.pi):.2f}Hz"
    )

    return {
        "mass_kg": m,
        "E_pa": E,
        "fy_pa": fy,
        "I_m4": I,
        "A_m2": A,
        "L_m": L,
        "b_m": b,
        "xi": xi,
        "Pcr_N": Pcr,
        "wn_rad": wn,
        "nonlinear": True,
        "n_elements": n_elem,
        "top_node": top_node,
    }


# ---------------------------------------------------------------------------
# SolverBackend implementation (abstract interface in solver_backend.py)
# ---------------------------------------------------------------------------

STRUCTURAL_REQUIRED_PARAMS = {
    "nonlinear.concrete.epsc0": "Strain at peak compressive stress (typ. 0.002)",
    "nonlinear.concrete.fpcu_ratio": "Residual post-peak strength ratio (typ. 0.2)",
    "nonlinear.concrete.epsU": "Ultimate crushing strain (typ. 0.005-0.008)",
    "nonlinear.concrete.ft_ratio": "Tensile strength / fc (typ. 0.08-0.12)",
    "nonlinear.concrete.Ets": "Tension stiffening slope (Pa)",
    "nonlinear.concrete.confinement_ratio": "Confinement factor fcc/fc (typ. 1.2-1.5)",
    "nonlinear.steel.fy": "Steel yield stress (typ. 420e6 Pa Grade 60)",
    "nonlinear.steel.Es": "Steel elastic modulus (typ. 200e9 Pa)",
    "nonlinear.steel.b_hardening": "Strain hardening ratio (typ. 0.01)",
    "nonlinear.section.cover": "Concrete cover (m)",
    "nonlinear.section.n_bars_face": "Rebar per face",
    "nonlinear.section.bar_diameter": "Rebar diameter (m)",
    "nonlinear.section.stirrup_diameter": "Stirrup diameter (m)",
    "nonlinear.section.stirrup_spacing": "Stirrup spacing (m)",
    "nonlinear.geometry.L": "Column length (m)",
    "nonlinear.geometry.b": "Square section width (m)",
}


# Inline abstract base to avoid circular imports with solver_backend.py.
# The ABC defined in solver_backend.py is what callers use; this file's
# StructuralBackend is what solver_backend.py imports and returns.
from abc import ABC, abstractmethod as _abstractmethod


class _StructuralBackendBase(ABC):
    """Private ABC — mirrors src.physics.solver_backend.SolverBackend."""

    @property
    @_abstractmethod
    def domain(self) -> str: ...

    @_abstractmethod
    def init_model(self, cfg: dict) -> dict: ...

    @_abstractmethod
    def step(self, measurement: float, dt: float, model_props: dict) -> dict: ...

    @_abstractmethod
    def check_required_params(self, cfg: dict) -> list[tuple[str, str]]: ...


class StructuralBackend(_StructuralBackendBase):
    """OpenSeesPy solver for the structural domain."""

    @property
    def domain(self) -> str:
        return "structural"

    def init_model(self, cfg: dict) -> dict:
        return init_model()

    def step(self, measurement: float, dt: float, model_props: dict) -> dict:
        top_node = model_props.get("top_node", 2)
        for key in ("mass_kg", "I_m4", "b_m"):
            if key not in model_props:
                print(
                    f"[CHAMBER] ERROR: step() missing required model_props key '{key}'",
                    file=sys.stderr,
                )
                sys.exit(1)
        mass_kg = model_props["mass_kg"]
        I_m4 = model_props["I_m4"]
        b_m = model_props["b_m"]
        c = b_m / 2.0

        force = mass_kg * measurement * _G_MPS2  # N (measurement = accel_g)
        ops.load(top_node, force, 0.0, 0.0)
        ok = ops.analyze(1, dt)

        try:
            ops.reactions()
            Mz_base = abs(ops.nodeReaction(1, 3))
            stress_pa = (Mz_base * c) / I_m4
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            print(
                f"[TORTURE] WARNING: stress computation failed: {e} — stress_pa=NaN",
                file=sys.stderr,
            )
            stress_pa = float("nan")

        return {
            "converged": ok == 0,
            "stress_pa": stress_pa,
        }

    def check_required_params(self, cfg: dict) -> list[tuple[str, str]]:
        missing = []
        for dotpath, desc in STRUCTURAL_REQUIRED_PARAMS.items():
            if _get_nested(cfg, dotpath) is None:
                missing.append((dotpath, desc))
        return missing
