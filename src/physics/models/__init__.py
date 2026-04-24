"""
Models Package — Model definitions and SSOT-derived parameter access.

Exports `load_sim_params()` which reads `config/params.yaml` fresh on each
import, guaranteeing that mass, stiffness, damping and Newmark-integrator
values come from the SSOT. Also exposes `init_model()` to initialize a
1-DOF oscillator in OpenSeesPy for quick smoke tests.

For production runs (Monte Carlo, pushover, IDA) the example runners in
`examples/rc_5story_peru/` build their own multi-DOF models directly.
"""
