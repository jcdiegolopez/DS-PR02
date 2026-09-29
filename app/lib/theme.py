"""Paleta, estilos y plantilla de Plotly de la app.

La justificación de cada color está en reports/app_paleta.md. Resumen:
- Neutros: fondo gris claro (proyector con luz de salón) y tinta casi negra.
- Caja de subtítulo: negro con texto blanco, como los closed captions.
- Amarillo caption: reservado para lo activo (frame o carácter actual).
- Modelos: Okabe-Ito, paleta categórica segura para daltonismo.
- Tipos de carácter: otros tonos Okabe-Ito que no chocan con los de los modelos.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GROUND = "#EEEFEC"
PANEL = "#F8F8F6"
INK = "#111213"
MUTED = "#55585C"
RULE = "#C9CBC6"
CAPTION_BG = "#0B0B0C"
CAPTION_FG = "#F5F5F0"
ACTIVE = "#F0E442"
ERROR = "#D55E00"

MODEL_COLORS = {"m1": "#E69F00", "m2": "#56B4E9", "m3": "#009E73"}

CATEGORY_COLORS = {
    "letras": "#0072B2",
    "dígitos": "#CC79A7",
    "símbolos": "#8C6D1F",
    "espacios": "#8A8D91",
}
CATEGORY_KEYS = {"letters": "letras", "digits": "dígitos", "symbols": "símbolos", "spaces": "espacios"}

FONT = "Atkinson Hyperlegible Next, sans-serif"
MONO = "Atkinson Hyperlegible Mono, monospace"

_CSS = f"""
<style>
:root {{
  --ground: {GROUND}; --panel: {PANEL}; --ink: {INK}; --muted: {MUTED}; --rule: {RULE};
  --cap-bg: {CAPTION_BG}; --cap-fg: {CAPTION_FG}; --active: {ACTIVE}; --error: {ERROR};
  --mono: {MONO};
}}
::selection {{ background: var(--active); color: var(--ink); }}
html {{ scrollbar-color: #9A9DA2 var(--ground); }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.015em; text-wrap: balance; }}
h1 {{ font-size: 2.35rem !important; line-height: 1.1 !important; }}
h2 {{ margin-top: 2.2rem !important; font-size: 1.55rem !important; }}
p, li {{ max-width: 72ch; }}
[data-testid="stMetricValue"], .tnum, table {{ font-variant-numeric: tabular-nums; }}
:focus-visible {{ outline: 3px solid var(--active) !important; outline-offset: 2px; }}

/* Línea de subtítulo: caja negra, texto blanco, monoespaciado */
.cap {{ background: var(--cap-bg); color: var(--cap-fg); font-family: var(--mono);
  padding: .9rem 1.1rem 1rem; margin: 0 0 .5rem; }}
.cap-tag {{ font-family: var(--mono); font-size: .78rem; color: #A9ABA6; display: flex;
  justify-content: space-between; gap: 1rem; margin-bottom: .45rem; }}
.cap-tag b {{ color: var(--cap-fg); font-weight: 600; }}
.cap-tag > span {{ display: inline-flex; gap: 1.1rem; align-items: center; flex-wrap: wrap; }}
.cap .model-mark {{ color: var(--cap-fg); }}
.cap-grid {{ display: flex; flex-wrap: wrap; row-gap: .55rem; }}
.cap-col {{ display: grid; grid-template-rows: 2.1rem 2.1rem; width: 1.05rem; text-align: center;
  font-size: 1.5rem; line-height: 2.1rem; }}
.cap-col span {{ display: block; }}
.cap.grid .cap-col span {{ box-shadow: inset 0 0 0 1px #34373B; }}
.cap-col .ref {{ color: #BDBFBA; }}
.cap-col.sub .hyp {{ background: var(--error); color: #FFFFFF; }}
.cap-col.sub .ref {{ text-decoration: underline 2px var(--error); text-underline-offset: 3px; color: var(--cap-fg); }}
.cap-col.del .ref {{ text-decoration: underline 2px var(--error); text-underline-offset: 3px; color: var(--cap-fg); }}
.cap-col.del .hyp::after {{ content: "_"; color: var(--error); }}
.cap-col.ins .hyp {{ outline: 1px dashed #F5F5F0; outline-offset: -3px; }}
.cap-rowlabels {{ display: grid; grid-template-rows: 2.1rem 2.1rem; font-size: .7rem; color: #A9ABA6;
  margin-right: .8rem; line-height: 2.1rem; }}
.cap-line {{ font-size: 1.35rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }}
.cap-line .now {{ background: var(--active); color: var(--ink); }}
.cap-legend {{ display: flex; flex-wrap: wrap; gap: 1.2rem; font-size: .82rem; color: var(--muted); margin: .2rem 0 1rem; }}
.cap-legend i {{ display: inline-block; width: .9rem; height: .9rem; vertical-align: -2px; margin-right: .35rem; }}

/* Marca de modelo: barra de su color + nombre */
.model-mark {{ display: inline-flex; align-items: center; gap: .45rem; font-weight: 600; }}
.model-mark i {{ width: .8rem; height: .8rem; display: inline-block; }}

/* Pasos del preprocesamiento: columna numerada estricta, resultado debajo del paso */
.steps {{ margin: .3rem 0 1rem; border-top: 1px solid var(--rule); }}
.steps .row {{ display: grid; grid-template-columns: 1.7rem 1fr; padding: .45rem 0; border-bottom: 1px solid var(--rule); }}
.steps .n {{ font-family: var(--mono); color: var(--muted); grid-row: span 2; }}
.steps .k {{ font-weight: 600; font-size: .92rem; }}
.steps .v {{ font-size: .9rem; color: var(--muted); font-variant-numeric: tabular-nums; }}

/* Cifras en línea con decimales fijos */
.figline {{ font-size: 1.08rem; line-height: 1.7; }}
.figline b {{ font-family: var(--mono); font-variant-numeric: tabular-nums; font-weight: 600; }}
.pending {{ color: var(--muted); font-style: italic; }}

/* Tabla comparativa */
table.cmp {{ border-collapse: collapse; width: 100%; font-size: .95rem; }}
table.cmp th {{ text-align: left; font-weight: 600; color: var(--muted); padding: .45rem .6rem; border-bottom: 2px solid var(--ink); }}
table.cmp td {{ padding: .5rem .6rem; border-bottom: 1px solid var(--rule); }}
table.cmp td.num {{ font-family: var(--mono); text-align: right; }}
table.cmp th.num {{ text-align: right; }}
table.cmp tr.best td.num {{ font-weight: 600; }}

section[data-testid="stSidebar"] .sidebar-note {{ font-size: .8rem; color: #A9ABA6; line-height: 1.45; }}
@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  h1 {{ font-size: 1.8rem !important; }}
  .cap-col {{ width: .82rem; font-size: 1.15rem; grid-template-rows: 1.7rem 1.7rem; line-height: 1.7rem; }}
  .cap-rowlabels {{ grid-template-rows: 1.7rem 1.7rem; line-height: 1.7rem; }}
  .cap-tag {{ flex-direction: column; gap: .35rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def apply() -> None:
    """Inyecta los estilos y registra la plantilla de Plotly (una vez por ejecución)."""
    st.html(_CSS)
    pio.templates["subtitulo"] = _template()
    pio.templates.default = "subtitulo"


def model_mark(key: str, label: str) -> str:
    return f'<span class="model-mark"><i style="background:{MODEL_COLORS.get(key, MUTED)}"></i>{label}</span>'


def _template() -> go.layout.Template:
    axis = dict(gridcolor="#D9DAD6", zerolinecolor="#B9BBB6", linecolor=INK, ticks="outside",
                tickcolor=RULE, title_font=dict(size=13, color=MUTED), tickfont=dict(size=12, color=INK),
                automargin=True)
    return go.layout.Template(layout=dict(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        colorway=list(MODEL_COLORS.values()) + list(CATEGORY_COLORS.values()),
        xaxis=axis, yaxis=axis,
        margin=dict(l=10, r=10, t=36, b=10),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor=CAPTION_BG, font=dict(family=MONO, color=CAPTION_FG, size=13),
                        bordercolor=CAPTION_BG),
        bargap=0.25,
    ))
