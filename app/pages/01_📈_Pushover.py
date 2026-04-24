"""Illapa — Pushover analysis page.

Static nonlinear pushover: lateral load grows monotonically until roof drift
reaches target. The plot shows base-shear V vs roof displacement Δ.

This page runs a PEDAGOGICAL DEMO solver. The full OpenSeesPy 5-story nonlinear
model used in the LAICSEE 2026 paper lives in `examples/rc_5story_peru/run_pushover.py`
and is too heavy for an interactive Streamlit page (30+ s per run). Here we use a
closed-form bilinear capacity curve parameterised from ASCE 41-17 Sec. 3.3.1.2.1,
calibrated against the real paper outputs (K0, V_y, delta_y, alpha_post) so the
shape matches what the full solver produces.

If you want the real OpenSeesPy run, use the CLI script — this page teaches the
idea; the script computes the truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Add parent to path so `theme` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import CHART, inject_css, sidebar_brand  # noqa: E402

st.set_page_config(
    page_title="Illapa — Pushover",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
sidebar_brand()

# ---------------------------------------------------------------------------
# Header.
# ---------------------------------------------------------------------------
st.markdown('<div class="ill-kicker">Análisis Estático No Lineal · ASCE 41-17</div>', unsafe_allow_html=True)
st.markdown("# Pushover")
st.markdown(
    '<p class="ill-lead">Aplicá una distribución lateral modal creciente y observá '
    'cómo se degrada la rigidez, aparece la fluencia, y la estructura transita hacia '
    'el colapso. El corte basal <em>V</em> contra el desplazamiento de azotea <em>Δ</em> '
    'es la firma no lineal del edificio.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ill-demo-banner">'
    '<strong>Simulación pedagógica</strong> · Esta página ejecuta un modelo bilineal '
    'cerrado en forma (ASCE 41-17 idealización), calibrado contra las salidas reales del '
    'modelo OpenSeesPy del paper LAICSEE 2026. Para la corrida NLTH completa, usá '
    '<code>python examples/rc_5story_peru/run_pushover.py</code>.'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Controls.
# ---------------------------------------------------------------------------
col_inputs, col_plot = st.columns([1, 2.2], gap="large")

with col_inputs:
    st.markdown("### Parámetros")
    st.caption("Defaults = caso LAICSEE 2026 (5 niveles, Lima Zona 3).")

    height_m = st.slider(
        "Altura total H (m)",
        min_value=8.0,
        max_value=24.0,
        value=15.0,
        step=0.5,
        help="Altura total del edificio. Defaults a un 5-niveles de 3 m entre pisos.",
    )
    mass_tons = st.slider(
        "Masa total M (toneladas)",
        min_value=100.0,
        max_value=1200.0,
        value=500.0,
        step=10.0,
        help="Masa sísmica agrupada. 500 t ≈ losa típica de concreto armado × 5 niveles.",
    )
    k_MNm = st.slider(
        "Rigidez inicial K₀ (MN/m)",
        min_value=20.0,
        max_value=200.0,
        value=89.4,
        step=0.5,
        help="Rigidez lateral elástica inicial. Default del LAICSEE: 89.4 MN/m.",
    )
    fy_kN = st.slider(
        "Cortante de fluencia V_y (kN)",
        min_value=1000,
        max_value=8000,
        value=3643,
        step=25,
        help="Corte basal al cual aparece la primera rótula plástica significativa.",
    )
    xi = st.slider(
        "Amortiguamiento ξ (%)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Amortiguamiento viscoso equivalente. 5% es el valor E.030 por defecto.",
    )
    alpha_post = st.slider(
        "Pendiente post-fluencia α",
        min_value=-0.20,
        max_value=0.10,
        value=-0.116,
        step=0.005,
        help="Rigidez post-fluencia normalizada. α < 0 indica ablandamiento (suavizado).",
        format="%.3f",
    )

    run = st.button("Correr simulación", use_container_width=True)

with col_plot:
    # Compute the curve every render (fast enough for this closed-form model).
    # ASCE 41-17 idealised bilinear with post-yield slope.
    K0 = k_MNm * 1e6  # N/m
    Vy_N = fy_kN * 1e3
    delta_y = Vy_N / K0  # m

    # FEMA 356 / ASCE 41-17 Sec. 3.3.1.2.1 definition of Delta_u:
    # "first post-peak drop to 0.80 V_peak" (paper §2.4).
    # In the bilinear idealisation with softening (alpha < 0), V_peak = V_y
    # and V(delta) = Vy + alpha*K0*(delta - delta_y) for delta > delta_y.
    # Setting V(delta_u) = 0.80 * V_y gives:
    #     delta_u = delta_y + (-0.20 * V_y) / (alpha * K0)        (alpha < 0)
    # For alpha >= 0 (hardening / perfectly plastic) the curve never drops to
    # 0.80 V_peak, so we fall back to FEMA 356 Table C1-3 near-collapse roof
    # drift of 0.04 * H. In practice the FEMA criterion also gets capped by
    # 0.04 * H if alpha is very small (near-elastic-perfectly-plastic).
    roof_drift_cap = 0.04 * height_m  # FEMA 356 near-collapse roof drift
    if alpha_post < 0:
        delta_u_fema = delta_y + (-0.20 * Vy_N) / (alpha_post * K0)
    else:
        delta_u_fema = float("inf")
    delta_u = min(delta_u_fema, roof_drift_cap)
    cap_active = delta_u_fema > roof_drift_cap  # True when 4% H governs

    n_pts = 200
    delta = np.linspace(0.0, delta_u, n_pts)
    V = np.where(
        delta <= delta_y,
        K0 * delta,
        Vy_N + alpha_post * K0 * (delta - delta_y),
    )
    V_peak = float(np.max(V))
    V_final = float(V[-1])
    mu = delta_u / delta_y

    fig = go.Figure()

    # Bilinear curve.
    fig.add_trace(
        go.Scatter(
            x=delta * 1000,  # mm
            y=V / 1e3,  # kN
            mode="lines",
            line=dict(color=CHART[0], width=3),
            name="Curva capacidad",
            hovertemplate="Δ = %{x:.1f} mm<br>V = %{y:.0f} kN<extra></extra>",
        )
    )

    # Yield marker.
    fig.add_trace(
        go.Scatter(
            x=[delta_y * 1000],
            y=[Vy_N / 1e3],
            mode="markers+text",
            marker=dict(color=CHART[1], size=12, symbol="circle", line=dict(color="#14110F", width=1.5)),
            text=["  Fluencia"],
            textposition="middle right",
            textfont=dict(family="Fraunces, serif", size=14, color="#14110F"),
            name="Punto de fluencia",
            hovertemplate="Δ_y = %{x:.1f} mm<br>V_y = %{y:.0f} kN<extra></extra>",
        )
    )

    # Elastic slope dashed reference.
    fig.add_trace(
        go.Scatter(
            x=[0, delta_u * 1000],
            y=[0, K0 * delta_u / 1e3],
            mode="lines",
            line=dict(color=CHART[3], width=1, dash="dot"),
            name="Extensión elástica (referencia)",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="simple_white",
        font=dict(family="Inter, sans-serif", size=14, color="#14110F"),
        title=dict(
            text="Curva de capacidad (pushover)",
            font=dict(family="Fraunces, serif", size=22, color="#14110F"),
            x=0,
        ),
        xaxis=dict(
            title="Desplazamiento de azotea Δ (mm)",
            gridcolor="#D4C6B0",
            zerolinecolor="#8A7E71",
        ),
        yaxis=dict(
            title="Corte basal V (kN)",
            gridcolor="#D4C6B0",
            zerolinecolor="#8A7E71",
        ),
        plot_bgcolor="#FAF6EC",
        paper_bgcolor="#F5EEDC",
        margin=dict(l=60, r=20, t=60, b=60),
        height=460,
        legend=dict(
            bgcolor="rgba(250,246,236,0.8)",
            bordercolor="#D4C6B0",
            borderwidth=1,
            x=0.02,
            y=0.98,
            font=dict(size=12),
        ),
        hoverlabel=dict(bgcolor="#FAF6EC", font=dict(family="Inter, sans-serif")),
    )

    st.plotly_chart(fig, use_container_width=True)

    if run:
        st.caption(
            f"Modelo bilineal (ASCE 41-17) recomputado · {n_pts} puntos en la curva · "
            "respuesta instantánea (forma cerrada, sin solver)."
        )

    # Key metrics.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Δ_y (mm)", f"{delta_y*1000:.1f}")
    m2.metric("V_y (kN)", f"{Vy_N/1e3:,.0f}")
    mu_help = (
        "Δ_u = 0.04·H (tope FEMA 356 near-collapse). α ≥ 0 no ablanda."
        if cap_active
        else "Δ_u donde V cae al 80 % de V_peak (FEMA 356 / ASCE 41-17)."
    )
    m3.metric("Ductilidad μ", f"{mu:.2f}", help=mu_help)
    m4.metric("V_peak (kN)", f"{V_peak/1e3:,.0f}")

# ---------------------------------------------------------------------------
# Pedagogical explanation.
# ---------------------------------------------------------------------------
st.markdown('<hr class="ill-divider">', unsafe_allow_html=True)

exp_a, exp_b = st.columns(2)
with exp_a:
    st.markdown("### ¿Qué estás viendo?")
    st.markdown(
        """
La curva **V vs Δ** resume el comportamiento no lineal del edificio:

1. **Tramo elástico** — pendiente = K₀. La estructura es reversible.
2. **Punto de fluencia** — primera rótula plástica; la rigidez cae.
3. **Post-fluencia** — si α > 0, endurecimiento (poco común). Si α < 0, ablandamiento
   (típico en estructuras reales de concreto, porque la carga axial induce daño geométrico P-Δ).
4. **Ductilidad μ = Δ_u / Δ_y** — cuántas veces la deformación de fluencia aguanta
   la estructura antes de que el corte basal caiga al **80 % del pico** (criterio FEMA 356
   / ASCE 41-17 §3.3.1.2.1). Con rama de ablandamiento, Δ_u sale del slope α directamente;
   si α ≥ 0 se aplica el tope FEMA 356 de 0.04·H (near-collapse roof drift).

El caso LAICSEE reporta μ ≈ 2.73 y α ≈ -0.116, valores consistentes con un pórtico
peruano de 5 niveles sin muros de corte. Observá que μ depende fuertemente de α: a mayor
ablandamiento (α más negativo), menor ductilidad.
"""
    )

with exp_b:
    st.markdown("### Lectura en inglés")
    st.markdown(
        """
**Capacity curve V vs Δ — the nonlinear signature of the building.**

1. **Elastic branch** — slope = K₀. Structure is reversible.
2. **Yield point** — first significant plastic hinge; stiffness drops.
3. **Post-yield** — α > 0 means hardening (rare). α < 0 means softening, typical of
   real RC frames because axial load triggers P-Δ geometric damage.
4. **Ductility μ = Δ_u / Δ_y** — how many yield-deformations the building sustains
   before base shear drops to **80 % of the peak** (FEMA 356 / ASCE 41-17 §3.3.1.2.1).
   With softening branch, Δ_u follows directly from α; if α ≥ 0 we fall back to the
   FEMA 356 near-collapse cap of 0.04·H.

The LAICSEE case reports μ ≈ 2.73 and α ≈ −0.116, consistent with a 5-story Peruvian
frame without shear walls. Note μ is mostly controlled by α: the more negative the
softening, the lower the available ductility.
"""
    )

st.caption(
    "Fuente de los defaults: `config/params.yaml` (SSOT) y "
    "`data/processed/pushover_results.json` del paper LAICSEE 2026. "
    "Referencia normativa: ASCE 41-17 Sec. 3.3.1.2.1."
)
