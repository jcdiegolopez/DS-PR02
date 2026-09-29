"""Figuras de Plotly de la app."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.tokenizer import BLANK_ID
from src.visualizers import HAND_CONNECTIONS

from lib import theme
from lib.inference import Prepared, Transcription, emitted_so_far, token_char

FINGERTIPS = [4, 8, 12, 16, 20]
MAX_PLAYBACK_FRAMES = 320


PALM = {(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)}


def _bones(points: np.ndarray, palm: bool) -> tuple[list, list]:
    xs, ys = [], []
    for a, b in HAND_CONNECTIONS:
        if ((a, b) in PALM) == palm:
            xs += [points[a, 0], points[b, 0], None]
            ys += [points[a, 1], points[b, 1], None]
    return xs, ys


def playback(prepared: Prepared, tr: Transcription | None, color: str) -> go.Figure:
    """Esqueleto de la mano (lo que ve el modelo) + pista de alineación CTC + subtítulo que se escribe."""
    hand = prepared.valid[:, :63].reshape(-1, 21, 3).astype(float)
    hand[..., 1] *= -1  # en MediaPipe y crece hacia abajo
    n = len(hand)
    step = max(1, math.ceil(n / MAX_PLAYBACK_FRAMES))
    idx = np.arange(0, n, step)
    lo = np.nanpercentile(hand[..., :2].reshape(-1, 2), 1, axis=0) - 0.35
    hi = np.nanpercentile(hand[..., :2].reshape(-1, 2), 99, axis=0) + 0.35
    so_far = emitted_so_far(tr.frame_ids) if tr is not None else [""] * n

    fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2], vertical_spacing=0.06)

    def frame_traces(t: int) -> list[go.Scatter]:
        pts = hand[t]
        px, py = _bones(pts, palm=True)
        fx, fy = _bones(pts, palm=False)
        return [
            go.Scatter(x=px, y=py, mode="lines", line=dict(color="#8A8D91", width=2), hoverinfo="skip"),
            go.Scatter(x=fx, y=fy, mode="lines", line=dict(color=theme.INK, width=4), hoverinfo="skip"),
            go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", hoverinfo="skip",
                       marker=dict(size=[13 if i in FINGERTIPS else 8 for i in range(21)],
                                   color=[color if i in FINGERTIPS else theme.INK for i in range(21)],
                                   line=dict(color=theme.PANEL, width=1.5))),
            go.Scatter(x=[t, t], y=[-0.2, 1.25], mode="lines", line=dict(color=theme.INK, width=2),
                       hoverinfo="skip"),
        ]

    for trace, row in zip(frame_traces(0), (1, 1, 1, 2)):
        fig.add_trace(trace, row=row, col=1)

    # Pista de alineación: un tick por frame; los frames que emiten carácter se marcan con su letra
    if tr is not None:
        emits = np.array([tok != BLANK_ID for tok in tr.frame_ids])
        frames = np.arange(n)
        fig.add_trace(go.Bar(x=frames, y=np.where(emits, 1.0, 0.18), marker_color=np.where(emits, color, "#B9BBB6"),
                             marker_line_width=0, width=1.0,
                             customdata=np.stack([[token_char(t) or "blank" for t in tr.frame_ids], tr.frame_conf], 1),
                             hovertemplate="frame %{x}<br>%{customdata[0]} (p=%{customdata[1]:.2f})<extra></extra>"),
                      row=2, col=1)
        labels = [(i, token_char(t)) for i, t in enumerate(tr.frame_ids)
                  if t != BLANK_ID and (i == 0 or tr.frame_ids[i - 1] != t)]
        fig.add_trace(go.Scatter(x=[i for i, _ in labels], y=[1.12] * len(labels),
                                 text=[c if c != " " else "␣" for _, c in labels], mode="text",
                                 textfont=dict(family=theme.MONO, size=12, color=theme.INK), hoverinfo="skip"),
                      row=2, col=1)
    fig.update_traces(selector=dict(type="bar"), offset=-0.5)

    def annotations(t: int) -> list[dict]:
        text = so_far[t]
        return [
            dict(text=f"<b>{text[-34:]}</b>" if text else "", x=0.5, y=0.25, xref="paper", yref="paper",
                 showarrow=False, font=dict(family=theme.MONO, size=24, color=theme.CAPTION_FG),
                 bgcolor=theme.CAPTION_BG if text else "rgba(0,0,0,0)", borderpad=10,
                 xanchor="center", yanchor="bottom"),
            dict(text=f"f {t:03d}/{n - 1:03d}", x=1, y=1.0, xref="paper", yref="paper", showarrow=False,
                 xanchor="right", yanchor="bottom", font=dict(family=theme.MONO, size=12, color=theme.MUTED)),
        ]

    fig.frames = [go.Frame(name=str(t), data=frame_traces(int(t)), traces=[0, 1, 2, 3],
                           layout=dict(annotations=annotations(int(t)))) for t in idx]
    duration = int(1000 / 30 * step)
    fig.update_layout(
        height=600, showlegend=False, annotations=annotations(0), margin=dict(l=10, r=10, t=30, b=70),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.11, xanchor="left", yanchor="top",
                          pad=dict(r=8, t=0), showactive=False, bgcolor=theme.PANEL, bordercolor=theme.INK,
                          font=dict(family=theme.FONT, size=13, color=theme.INK),
                          buttons=[dict(label="Reproducir", method="animate",
                                        args=[None, dict(frame=dict(duration=duration, redraw=True),
                                                         transition=dict(duration=0), fromcurrent=True)]),
                                   dict(label="Pausa", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(active=0, x=0.22, len=0.78, y=-0.09, yanchor="top", pad=dict(t=0, b=0),
                      currentvalue=dict(visible=False), ticklen=0, bgcolor=theme.RULE,
                      activebgcolor=theme.ACTIVE, bordercolor=theme.INK, font=dict(size=1, color=theme.PANEL),
                      steps=[dict(method="animate", label="", args=[[str(t)], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for t in idx])],
    )
    fig.update_xaxes(range=[lo[0], hi[0]], visible=False, row=1, col=1)
    fig.update_yaxes(range=[lo[1] - 0.55, hi[1]], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(range=[-0.5, n - 0.5], showgrid=False, ticks="", row=2, col=1)
    fig.update_yaxes(range=[0, 1.3], visible=False, row=2, col=1)
    return fig


# ------------------------------------------------------------------ Explorar

def char_frequency(meta: pd.DataFrame, category_of) -> go.Figure:
    counts = pd.Series(list("".join(meta["phrase"]))).value_counts()
    df = counts.rename_axis("char").reset_index(name="n")
    df["share"] = df["n"] / df["n"].sum()
    df["cat"] = df["char"].map(category_of)
    df["label"] = df["char"].replace({" ": "␣"})
    fig = go.Figure()
    for cat, color in theme.CATEGORY_COLORS.items():
        part = df[df["cat"] == cat]
        fig.add_trace(go.Bar(x=part["label"], y=part["share"], name=cat, marker_color=color,
                             hovertemplate="%{x}: %{y:.2%}<extra>" + cat + "</extra>"))
    fig.update_layout(height=360, barmode="overlay", yaxis_tickformat=".0%",
                      xaxis=dict(categoryorder="array", categoryarray=df["label"].tolist(), tickfont=dict(family=theme.MONO)),
                      yaxis_title="proporción de caracteres")
    return fig


def phrase_lengths(meta: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(x=meta["phrase_length"], xbins=dict(size=1), marker_color=theme.INK,
                                 hovertemplate="%{x} caracteres: %{y:,} frases<extra></extra>"))
    median = meta["phrase_length"].median()
    fig.add_vline(x=median, line=dict(color=theme.ERROR, width=2, dash="dot"),
                  annotation_text=f"mediana {median:.0f}", annotation_position="top right",
                  annotation_font=dict(color=theme.ERROR, family=theme.MONO))
    fig.update_layout(height=320, xaxis_title="caracteres por frase", yaxis_title="frases", bargap=0.05)
    return fig


def participants(meta: pd.DataFrame, highlight_test: bool) -> go.Figure:
    per = meta.groupby(["participant_id", "split"]).size().reset_index(name="n").sort_values("n", ascending=False)
    colors = np.where(per["split"] == "test", theme.ACTIVE, theme.INK) if highlight_test else theme.INK
    line = np.where(per["split"] == "test", theme.INK, "rgba(0,0,0,0)") if highlight_test else "rgba(0,0,0,0)"
    fig = go.Figure(go.Bar(x=np.arange(len(per)), y=per["n"], marker=dict(color=colors, line=dict(color=line, width=1)),
                           customdata=np.stack([per["participant_id"], per["split"]], 1),
                           hovertemplate="participante %{customdata[0]}<br>%{y:,} secuencias (%{customdata[1]})<extra></extra>"))
    fig.update_layout(height=320, xaxis_title="participantes ordenados por aporte", yaxis_title="secuencias",
                      xaxis_showticklabels=False, bargap=0.1)
    return fig


def presence_strip(raw: pd.DataFrame) -> go.Figure:
    rows = {"mano izquierda": "left_hand", "mano derecha": "right_hand", "pose": "pose"}
    z, labels = [], []
    for label, kind in rows.items():
        cols = [c for c in raw.columns if f"_{kind}_" in c]
        z.append(raw[cols].notna().all(axis=1).astype(int).to_numpy() if cols else np.zeros(len(raw), int))
        labels.append(label)
    fig = go.Figure(go.Heatmap(z=z, y=labels, x=np.arange(len(raw)), colorscale=[[0, "#DADBD7"], [1, theme.INK]],
                               showscale=False, zmin=0, zmax=1, xgap=0, ygap=4,
                               hovertemplate="frame %{x}<br>%{y}: %{z}<extra></extra>"))
    fig.update_layout(height=190, xaxis_title="frame crudo", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def trajectory(prepared: Prepared, color: str) -> go.Figure:
    hand = prepared.valid[:, :63].reshape(-1, 21, 3)
    tip = hand[:, 8, :2].astype(float)
    t = np.arange(len(tip))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=tip[:, 0], name="x (índice)", line=dict(color=theme.INK, width=2)))
    fig.add_trace(go.Scatter(x=t, y=-tip[:, 1], name="y (índice)", line=dict(color=color, width=2)))
    fig.update_layout(height=260, xaxis_title="frame válido", yaxis_title="posición relativa a la muñeca")
    return fig


# ------------------------------------------------------------------ Rendimiento

def metric_bars(rows: pd.DataFrame, column: str, title: str, fmt: str, lower_is_better: bool) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=rows[column], y=rows["label"], orientation="h", marker_color=rows["color"],
        marker_line=dict(color=theme.INK, width=1),
        text=[format(v, fmt) for v in rows[column]], textposition="outside",
        textfont=dict(family=theme.MONO, size=13), cliponaxis=False,
        hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    better = "menor es mejor" if lower_is_better else "mayor es mejor"
    fig.update_layout(title=f"{title} <span style='font-size:12px;color:{theme.MUTED}'>({better})</span>",
                      height=110 + 46 * len(rows), yaxis=dict(autorange="reversed"), xaxis_showgrid=True,
                      xaxis=dict(range=[0, max(rows[column].max() * 1.3, 1e-9)]))
    if "%" in fmt:
        fig.update_xaxes(tickformat=".0%")
    return fig


def category_errors(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for key, part in frame.groupby("model", sort=False):
        fig.add_trace(go.Bar(x=part["categoría"], y=part["error"], name=part["label"].iloc[0],
                             marker_color=part["color"].iloc[0], marker_line=dict(color=theme.INK, width=1),
                             text=[f"{v:.0%}" for v in part["error"]],
                             textposition="outside", textfont=dict(family=theme.MONO), cliponaxis=False,
                             hovertemplate="%{x}: %{y:.1%} de error<extra>" + part["label"].iloc[0] + "</extra>"))
    fig.update_layout(height=360, barmode="group", yaxis_tickformat=".0%", yaxis_title="tasa de error por carácter",
                      yaxis_range=[0, frame["error"].max() * 1.25])
    return fig


def frequency_vs_error(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for key, part in frame.groupby("model", sort=False):
        fig.add_trace(go.Scatter(
            x=part["frecuencia"], y=part["error"], mode="markers+text", name=part["label"].iloc[0],
            text=part["categoría"], textposition="top center", textfont=dict(size=12),
            marker=dict(size=16, color=part["color"].iloc[0], line=dict(color=theme.INK, width=1)),
            hovertemplate="%{text}<br>%{x:.1%} del corpus<br>%{y:.1%} de error<extra>" + part["label"].iloc[0] + "</extra>"))
    fig.update_layout(height=380, xaxis_title="proporción del corpus de entrenamiento (EDA)",
                      yaxis_title="tasa de error", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    return fig


def training_curves(histories: dict[str, pd.DataFrame], labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for key, hist in histories.items():
        color = theme.MODEL_COLORS[key]
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name=f"{labels[key]} entrenamiento",
                                 line=dict(color=color, width=2, dash="dot"), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name=f"{labels[key]} validación",
                                 line=dict(color=color, width=3), mode="lines+markers"))
    fig.update_layout(height=360, xaxis_title="época", yaxis_title="pérdida CTC", xaxis_dtick=1)
    return fig


def cost_vs_quality(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for _, row in frame.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["ms"]], y=[row["cer"]], mode="markers+text", name=row["label"], text=[row["label"]],
            textposition="middle right", marker=dict(size=10 + 30 * math.sqrt(row["params"] / frame["params"].max()),
                                                     color=row["color"], line=dict(color=theme.INK, width=1)),
            hovertemplate=f"{row['label']}<br>%{{x:.1f}} ms por secuencia<br>CER %{{y:.3f}}<br>"
                          f"{row['params']:,.0f} parámetros<extra></extra>"))
    fig.update_layout(height=340, showlegend=False, xaxis_title="milisegundos por secuencia (CPU de esta máquina)",
                      yaxis_title="CER (menor es mejor)", xaxis_rangemode="tozero", yaxis_rangemode="tozero")
    return fig
