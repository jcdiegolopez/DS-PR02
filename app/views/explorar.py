"""Explorar datos: el corpus de frases y cómo se ve una secuencia por dentro."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, theme
from lib.data import load_manifest, load_metadata, load_sample
from lib.inference import prepare

def category_of(char: str) -> str:
    if char.isalpha():
        return "letras"
    if char.isdigit():
        return "dígitos"
    if char.isspace():
        return "espacios"
    return "símbolos"


@st.cache_data(show_spinner=False)
def prepared_sample(file_name: str):
    return prepare(load_sample(file_name))


meta = load_metadata()
by_split = meta.groupby("split")["participant_id"].nunique()

st.title("Qué hay en los datos")
st.html(
    '<p class="figline">'
    f'<b>{len(meta):,}</b> frases deletreadas por <b>{meta["participant_id"].nunique()}</b> firmantes, '
    f'con <b>{int(meta["phrase_length"].sum()):,}</b> caracteres en total. La partición separa firmantes, no frases: '
    f'<b>{by_split["train"]}</b> para entrenar, <b>{by_split["val"]}</b> para validar y <b>{by_split["test"]}</b> para la prueba final.</p>'
)

# ------------------------------------------------------------------ caracteres
chars = pd.Series(list(meta["phrase"].str.cat()))
shares = chars.map(category_of).value_counts(normalize=True)
rare = chars.value_counts(normalize=True)
rare_symbols = [c for c in rare.index if category_of(c) == "símbolos" and rare[c] < 0.001]
st.header(f"Las letras son {shares['letras']:.0%} del texto; los símbolos, apenas {shares['símbolos']:.1%}")
st.markdown(
    f"Los dígitos ocupan {shares['dígitos']:.0%} porque muchas frases son teléfonos y direcciones. "
    f"{len(rare_symbols)} símbolos aparecen en menos de una de cada mil posiciones: el modelo casi no los ve al entrenar."
)
st.plotly_chart(charts.char_frequency(meta, category_of), config={"displayModeBar": False})
if st.toggle("Mostrar la tabla de frecuencias"):
    table = chars.value_counts().rename_axis("carácter").reset_index(name="apariciones")
    table["proporción"] = table["apariciones"] / table["apariciones"].sum()
    table["tipo"] = table["carácter"].map(category_of)
    table["carácter"] = table["carácter"].replace({" ": "␣ (espacio)"})
    st.dataframe(table, hide_index=True, column_config={
        "proporción": st.column_config.NumberColumn(format="percent"),
        "apariciones": st.column_config.NumberColumn(format="localized")}, height=320)

# ------------------------------------------------------------------ longitudes
lengths = meta["phrase_length"]
st.header(f"La mitad de las frases tiene {lengths.median():.0f} caracteres o menos")
st.markdown(f"Van de {lengths.min()} a {lengths.max()} caracteres. CTC necesita al menos un frame por carácter, "
            "así que las frases largas con secuencias cortas son las más difíciles de alinear.")
st.plotly_chart(charts.phrase_lengths(meta), config={"displayModeBar": False})

# ------------------------------------------------------------------ participantes
per = meta.groupby("participant_id").size()
st.header(f"Un firmante aporta {per.max():,} frases y otro solo {per.min():,}")
st.markdown("Si el mismo firmante estuviera en entrenamiento y en prueba, el modelo podría memorizar su estilo. "
            "Por eso los firmantes de prueba (en amarillo) quedan fuera del entrenamiento.")
highlight = st.toggle("Resaltar firmantes de prueba", value=True)
st.plotly_chart(charts.participants(meta, highlight), config={"displayModeBar": False})

# ------------------------------------------------------------------ una secuencia
st.header("Una secuencia por dentro")
manifest = load_manifest()
labels = {row.file: row.phrase for row in manifest.itertuples()}
choice = st.selectbox("Secuencia de prueba", list(labels), format_func=labels.get, index=2, key="explore-seq")
raw = load_sample(choice)
prepared = prepared_sample(choice)
coverage = manifest.set_index("file").loc[choice, "coverage"]
hand = "derecha" if prepared.hand == "right_hand" else "izquierda"
st.markdown(f"La mano activa es la **{hand}** y queda útil en **{coverage:.0%}** de los frames. En negro, los frames donde "
            "MediaPipe detectó cada parte del cuerpo; los huecos son pérdidas de tracking que el preprocesamiento recorta, "
            "interpola o marca como inválidas.")
st.plotly_chart(charts.presence_strip(raw), config={"displayModeBar": False})
st.markdown("Trayectoria de la punta del índice respecto a la muñeca, ya normalizada. Cada cambio brusco es un "
            "cambio de letra.")
st.plotly_chart(charts.trajectory(prepared, theme.MODEL_COLORS["m1"]), config={"displayModeBar": False})

with st.expander("Cobertura de la mano en las 30 secuencias de prueba"):
    cov = manifest["coverage"]
    st.markdown(f"Mediana **{cov.median():.0%}**; {int((cov < 0.5).sum())} de {len(cov)} quedan por debajo del 50% "
                "que el análisis exploratorio proponía como umbral de descarte.")
    st.bar_chart(manifest.sort_values("coverage").set_index("phrase")["coverage"], color=theme.INK,
                 horizontal=True, height=560, x_label="cobertura de la mano activa", y_label="")
