"""5-story RC moment-frame example (NTE E.030-2018 Peru, Zone 3 Lima).

Three complementary analyses share the same SSOT-driven model:

  - run_monte_carlo.py  Fragility via Kanai-Tajimi synthetic ground motions.
  - run_pushover.py     Modal pushover (ASCE 41-17 / FEMA 356 metrics).
  - run_ida.py          Incremental Dynamic Analysis on real NGA-West2 records.

All three read `config/params.yaml` (block `rc_mrf`) and write to
`data/processed/`.
"""
