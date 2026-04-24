# openseespy-edu-peru

**Open-source reproducible earthquake-engineering lab for structural dynamics education — Peru-case (NTE E.030-2018).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Status: v1.0.0-pending](https://img.shields.io/badge/status-v1.0.0--pending-orange.svg)](CHANGELOG.md)

> ⚠️ **This repository is under active development.** The v1.0.0 release (scheduled for **2026-05-08**) will include a Zenodo-archived snapshot with a formal DOI. Until then, the code is provisional.

---

## What this is

A reproducible pipeline that bridges classical nonlinear structural dynamics with AI-ready educational datasets, designed for undergraduate earthquake-engineering courses in Latin America. It integrates:

1. **OpenSeesPy** numerical modelling (5-story RC MRF, lumped-plasticity IMK hinges)
2. **Kanai-Tajimi** synthetic ground-motion generator with Saragoni-Hart envelope
3. **NTE E.030-2018** Peruvian seismic code compliance, including the Art. 28.2 inelastic drift amplification (0.75·R = 6.0)
4. **Three complementary analyses:** Monte-Carlo fragility, modal pushover (ASCE 41-17), incremental dynamic analysis (Vamvatsikos & Cornell 2002)
5. **SSOT governance** via a single `config/params.yaml`, SHA-256-sealed COMPUTE manifests, and AI-ready structured outputs

## What this is NOT

- A production-grade seismic design tool (use ETABS, SAP2000, or OpenSees native for that)
- A novel algorithmic contribution (the framework integrates existing methods; it does not invent them)
- A classroom-validated pedagogy (N=1 at this release; classroom pilot planned for v1.1)

## Quick start

```bash
# Requires Python 3.12 (OpenSeesPy is unstable on 3.13 Windows)
git clone https://github.com/Mikisbell/openseespy-edu-peru.git
cd openseespy-edu-peru
python -m venv .venv
source .venv/Scripts/activate  # Git Bash Windows, or .venv\Scripts\Activate.ps1 in PowerShell
pip install -r requirements.txt

# Run the full pipeline (approx. 75 seconds on a commodity laptop)
python examples/rc_5story_peru/run_monte_carlo.py
python examples/rc_5story_peru/run_pushover.py
python examples/rc_5story_peru/run_ida.py
python tools/plot_figures.py --domain structural
```

All figures and statistics are regenerated locally in the working directory (not shipped with the repo) and sealed with SHA-256 integrity hashes via `tools/generate_compute_manifest.py`.

## Citation

If you use this framework in academic work, please cite both the paper that introduced it and this code release:

```bibtex
@software{belico_openseespy_edu_peru_2026,
  author = {TBD},
  title = {{openseespy-edu-peru: Open-Source Reproducible Earthquake-Engineering Lab}},
  year = {2026},
  version = {v1.0.0},
  doi = {10.5281/zenodo.PENDING},
  url = {https://github.com/Mikisbell/openseespy-edu-peru},
  license = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## Philosophy

**Honest disclosure over narrative polish.** The framework ships with seven explicit limitations (shear-type R-derived artifact, NGA-West2 crustal records vs Peru subduction, Kanai-Tajimi mild conservatism, Rayleigh damping choice, N=1 pedagogical benchmark, Δt=0.01s integration step, statistical power = 0.789 below Cohen target). Each is documented in the accompanying paper and in the CHANGELOG. No calibration, no cherry-picking.

## Licence

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

Built on the OpenSeesPy ecosystem (McKenna 2011; Arroyo et al. 2024) and the PEER NGA-West2 database. Developed as a child project of the [belico-stack](https://github.com/Mikisbell/belico-stack) paper-production framework.

---

**Maintainer:** [@Mikisbell](https://github.com/Mikisbell)
**Contact:** see GitHub profile
**Status page:** [`CHANGELOG.md`](CHANGELOG.md)
