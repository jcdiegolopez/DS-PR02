"""Rendimiento: comparación de modelos desde models/metrics.json."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import charts, theme
from lib.data import MODEL_INFO, InputError, available_models, load_history, load_manifest, load_metadata, load_metrics, load_sample
from lib.inference import prepare, transcribe


@st.cache_data(show_spinner=False)
def prepared_sample(file_name: str):
    return prepare(load_sample(file_name))


def category_of(char: str) -> str:
    if char.isalpha():
        return "letras"
    if char.isdigit():
        return "dígitos"
    if char.isspace():
        return "espacios"
    return "símbolos"


ARTICLE = {"letras": "las letras", "dígitos": "los dígitos", "símbolos": "los símbolos", "espacios": "los espacios"}

try:
    metrics = load_metrics()
except InputError as exc:
    st.error(str(exc))
    st.stop()
evaluated = [k for k in MODEL_INFO if k in metrics]
pending = [k for k in MODEL_INFO if k not in metrics]

st.title("Cómo rinden los modelos")
if not evaluated:
    st.info("Todavía no hay métricas en models/metrics.json.")
    st.stop()

cohorts = list(dict.fromkeys((metrics[k]["split"], metrics[k]["sequences"]) for k in evaluated))
if len(cohorts) > 1:
    st.info("Hay resultados de distintos conjuntos o tamaños. Elige un grupo para compararlo por separado.")
    cohort = st.selectbox("Conjunto evaluado", cohorts,
                          format_func=lambda v: f"{'Prueba' if v[0] == 'test' else 'Validación'} · {v[1]:,} frases")
    evaluated = [k for k in evaluated if (metrics[k]["split"], metrics[k]["sequences"]) == cohort]
evaluated = st.multiselect("Modelos para comparar", evaluated, default=evaluated,
                          format_func=lambda k: MODEL_INFO[k]["label"])
if not evaluated:
    st.info("Selecciona al menos un modelo para ver sus métricas.")
    st.stop()
first = metrics[evaluated[0]]
split_name = "prueba" if first.get("split") == "test" else "validación"
st.markdown(f"Métricas sobre el conjunto de **{split_name}** ({first['sequences']:,} frases de firmantes separados "
            "de los de entrenamiento).")
if first["split"] == "val":
    st.caption("La validación se usa para seleccionar checkpoints e hiperparámetros. Estos resultados no son la evaluación final en test.")
else:
    st.caption("Resultados registrados como test en metrics.json. Consulta el notebook para el protocolo de evaluación.")
if len(evaluated) == 1:
    st.info("Solo hay métricas de un modelo en este conjunto; aún no se puede determinar cuál de las tres arquitecturas rinde mejor.")

rows = pd.DataFrame([{
    "model": k, "label": MODEL_INFO[k]["label"], "family": MODEL_INFO[k]["family"], "color": theme.MODEL_COLORS[k],
    "ned": metrics[k]["normalized_edit_distance"], "cer": metrics[k]["cer"], "exact": metrics[k]["exact_match"],
    "params": metrics[k]["parameters"], "gpu_ms": metrics[k].get("inference_ms_per_sequence", np.nan),
    "epoch": metrics[k].get("best_epoch"),
} for k in evaluated])

# ------------------------------------------------------------------ tabla comparativa
best_ned = rows["ned"].min()
body = "".join(
    f'<tr class="{"best" if r.ned == best_ned and len(rows) > 1 else ""}"><td>{theme.model_mark(r.model, r.label)}</td>'
    f'<td>{r.family}</td><td class="num">{r.ned:.4f}</td><td class="num">{r.cer:.4f}</td>'
    f'<td class="num">{r.exact:6.1%}</td><td class="num">{r.params:,}</td></tr>'
    for r in rows.itertuples())
body += "".join(
    f'<tr><td>{theme.model_mark(k, MODEL_INFO[k]["label"])}</td><td>{MODEL_INFO[k]["family"]}</td>'
    f'<td class="num pending" colspan="4">sin métricas registradas</td></tr>' for k in pending)
st.html('<table class="cmp"><thead><tr><th>Modelo</th><th>Arquitectura</th><th class="num">Distancia de edición</th>'
        '<th class="num">CER</th><th class="num">Frase exacta</th><th class="num">Parámetros</th></tr></thead>'
        f"<tbody>{body}</tbody></table>")
st.caption("Distancia de edición: promedio por frase de las ediciones necesarias dividido entre su longitud. "
           "CER: ediciones totales entre caracteres totales. Ambas: menor es mejor.")
st.download_button("Descargar métricas", rows.drop(columns=["color"]).to_csv(index=False).encode("utf-8-sig"),
                   file_name="metricas.csv", mime="text/csv")

if st.toggle("Mostrar las métricas globales en gráfica", value=True):
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(charts.metric_bars(rows, "ned", "Distancia de edición", ".3f", True), config={"displayModeBar": False})
    c2.plotly_chart(charts.metric_bars(rows, "cer", "CER", ".3f", True), config={"displayModeBar": False})
    c3.plotly_chart(charts.metric_bars(rows, "exact", "Frase exacta", ".1%", False), config={"displayModeBar": False})

# ------------------------------------------------------------------ errores por tipo de carácter
cat_rows = []
for k in evaluated:
    for key, value in metrics[k]["by_character_type"].items():
        cat_rows.append({"model": k, "label": MODEL_INFO[k]["label"], "color": theme.MODEL_COLORS[k],
                         "categoría": theme.CATEGORY_KEYS[key], "error": value["error_rate"],
                         "referencia": value["reference"]})
cats = pd.DataFrame(cat_rows)
lead = cats[(cats["model"] == rows.loc[rows["ned"].idxmin(), "model"]) & (cats["referencia"] > 0)].set_index("categoría")
if lead.empty:
    st.info("No hay caracteres de referencia para analizar errores por categoría.")
    st.stop()
worst = lead["error"].idxmax()
st.header(f"Lo que más falla son {ARTICLE[worst]}: {lead.loc[worst, 'error']:.0%} de error")
best_cat = lead["error"].idxmin()
st.markdown(
    "Cada barra es la proporción de caracteres de ese tipo que el modelo no transcribió bien (sustituidos u omitidos). "
    f"Lo que mejor sale son {ARTICLE[best_cat]}, con {lead.loc[best_cat, 'error']:.0%} de error."
)
if st.toggle("Mostrar el error por tipo de carácter", value=True):
    st.plotly_chart(charts.category_errors(cats), config={"displayModeBar": False})

# ------------------------------------------------------------------ frecuencia en el corpus vs error
meta = load_metadata()
train_chars = pd.Series(list(meta.loc[meta["split"] == "train", "phrase"].str.cat()))
freq = train_chars.map(category_of).value_counts(normalize=True)
cats["frecuencia"] = cats["categoría"].map(freq)
st.header("Cruce con el análisis exploratorio: ¿falla más lo que menos se vio?")
ranked = lead.assign(frecuencia=lead.index.map(freq)).sort_values("frecuencia")
monotone = ranked["error"].is_monotonic_decreasing
if monotone:
    finding = "El orden coincide: cuanto menos se vio un tipo de carácter al entrenar, más falla."
else:
    # Primer par que rompe la regla "más escaso, más error"
    pairs = [(a, b) for i, a in enumerate(ranked.index) for b in ranked.index[i + 1:]
             if ranked.loc[a, "error"] < ranked.loc[b, "error"]]
    a, b = pairs[0]
    finding = (f"La frecuencia sola no lo explica: {ARTICLE[a]} son el {ranked.loc[a, 'frecuencia']:.1%} del texto "
               f"y fallan {ranked.loc[a, 'error']:.0%}, menos que {ARTICLE[b]} ({ranked.loc[b, 'frecuencia']:.0%} del texto, "
               f"{ranked.loc[b, 'error']:.0%} de error).")
st.markdown(
    f"En el EDA, {ARTICLE[ranked.index[0]]} eran el tipo más escaso ({ranked['frecuencia'].iloc[0]:.1%} del texto de "
    f"entrenamiento) y {ARTICLE[ranked.index[-1]]} el más común ({ranked['frecuencia'].iloc[-1]:.0%}). {finding} "
    "Los espacios no tienen una seña propia en el deletreo; una explicación probable de su error es que el modelo "
    "tiene que deducirlos de las pausas entre palabras."
)
if st.toggle("Mostrar frecuencia contra error", value=False):
    st.plotly_chart(charts.frequency_vs_error(cats), config={"displayModeBar": False})

# ------------------------------------------------------------------ entrenamiento
histories = {k: h for k in evaluated if (h := load_history(k)) is not None}
if histories:
    st.header("Curvas de entrenamiento")
    notes = []
    for k, h in histories.items():
        tail = h["val_loss"].dropna().tail(2).to_numpy()
        if len(tail) == 2 and tail[1] < tail[0]:
            notes.append(f"{MODEL_INFO[k]['label']} redujo su pérdida de validación entre las dos últimas épocas "
                         f"registradas ({len(h)} épocas). Eso no garantiza mejoras con más entrenamiento.")
    st.markdown(" ".join(notes) or "Pérdida CTC por época en entrenamiento (punteada) y validación (continua).")
    if st.toggle("Mostrar las curvas", value=True):
        st.plotly_chart(charts.training_curves(histories, {k: MODEL_INFO[k]["label"] for k in histories}),
                        config={"displayModeBar": False})

# ------------------------------------------------------------------ eficiencia
st.header("Costo contra calidad")
st.markdown("El botón mide solo inferencia y decodificación en CPU sobre las "
            f"{len(load_manifest())} muestras, con una llamada de calentamiento por modelo. Excluye carga y limpieza; "
            "no recalcula las métricas de test.")
ready = [k for k in available_models() if k in evaluated]
if st.button("Medir en esta máquina", type="primary", disabled=not ready):
    manifest = load_manifest()
    timings = {}
    progress = st.progress(0.0, text="Midiendo")
    total = len(ready) * len(manifest)
    done = 0
    for k in ready:
        ms = []
        warmed = False
        for file_name in manifest["file"]:
            prepared = prepared_sample(file_name)
            if not prepared.mask.any():
                continue
            if not warmed:
                transcribe(k, prepared)
                warmed = True
            ms.append(transcribe(k, prepared).milliseconds)
            done += 1
            progress.progress(done / total, text=f"Midiendo {MODEL_INFO[k]['label']}")
        if ms:
            timings[k] = float(np.mean(ms))
    progress.empty()
    st.session_state["cpu_ms"] = timings

timings = st.session_state.get("cpu_ms")
if timings:
    cost = rows[rows["model"].isin(timings)].assign(ms=lambda d: d["model"].map(timings))
    st.plotly_chart(charts.cost_vs_quality(cost), config={"displayModeBar": False})
    st.html('<p class="figline">' + "<br>".join(
        f'{theme.model_mark(r.model, r.label)}: <b>{r.ms:6.1f}</b> ms por secuencia, <b>{r.params:,}</b> parámetros'
        for r in cost.itertuples()) + "</p>")
    st.caption("El tamaño del punto es proporcional al número de parámetros.")
else:
    gpu = rows.dropna(subset=["gpu_ms"])
    if len(gpu):
        st.caption("Referencia del entrenamiento (GPU, por lotes): " + ", ".join(
            f"{r.label} {r.gpu_ms:.1f} ms por secuencia" for r in gpu.itertuples()) + ".")
