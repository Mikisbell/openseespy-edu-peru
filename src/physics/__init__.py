"""
Physics Package — Root package for the numerical simulation domain.

Groups the OpenSeesPy structural backend modules and the SSOT-derived parameters.
The `models` sub-package contains model definitions and runtime params reader.

Data flow:
  config/params.yaml (SSOT)
      -> models.params.load_sim_params()   (runtime dict)
      -> torture_chamber.init_model()      (OpenSeesPy build)
      -> examples/* runners                 (Monte Carlo, pushover, IDA)

Depends on: config/params.yaml, openseespy, numpy, scipy, pyyaml.
"""
