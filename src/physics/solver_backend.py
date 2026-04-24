"""
src/physics/solver_backend.py — Abstract solver backend interface
=================================================================
Every domain solver (structural, fluid, ...) must implement this interface.
This framework ships with a structural backend (OpenSeesPy). New domains
can be added by subclassing `SolverBackend` and registering the subclass in
`get_solver_backend()`.

Pattern:
    backend = get_solver_backend(cfg)
    props   = backend.init_model(cfg)
    result  = backend.step(measurement, dt, props)
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

try:
    from src.physics.torture_chamber import StructuralBackend
except (ImportError, RuntimeError):
    # RuntimeError covers cases like openseespy requiring Python 3.8-3.12
    # on Windows; ImportError covers missing dependency.
    StructuralBackend = None  # type: ignore


class SolverBackend(ABC):
    """Interface every physics solver must implement."""

    @property
    @abstractmethod
    def domain(self) -> str:
        """Return the domain name (e.g. 'structural')."""

    @abstractmethod
    def init_model(self, cfg: dict) -> dict:
        """Initialize the solver model from an SSOT dict and return its properties."""

    @abstractmethod
    def step(self, measurement: float, dt: float, model_props: dict) -> dict:
        """Advance the model by one timestep given a sensor measurement.

        Returns at minimum {'converged': bool, 'stress_pa': float}.
        """

    @abstractmethod
    def check_required_params(self, cfg: dict) -> list[tuple[str, str]]:
        """Return list of (dotpath, description) for missing required params."""


def get_solver_backend(cfg: dict) -> SolverBackend:
    """Factory: return the structural backend by default.

    Extend this function to dispatch on `cfg["project"]["domain"]` when
    adding new domains.
    """
    domain = cfg.get("project", {}).get("domain", "structural")

    if domain == "structural":
        if StructuralBackend is None:
            print(
                "[SOLVER] ERROR: StructuralBackend unavailable — "
                "openseespy may not be installed.",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(
                "StructuralBackend could not be imported "
                "(src.physics.torture_chamber)."
            )
        return StructuralBackend()

    raise ValueError(
        f"Unknown domain '{domain}' in SSOT. "
        "Built-in domains: structural. "
        "To add a new domain, implement a SolverBackend subclass and "
        "register it in get_solver_backend()."
    )
