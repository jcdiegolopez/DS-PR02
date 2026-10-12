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


st.title("Qué hay en los datos")
corpus = load_metadata()
manifest = load_manifest()
scope = st.segmented_control("Datos para explorar", ["Muestras limpias", "Corpus original"], default="Muestras limpias")
if scope == "Muestras limpias":
    meta = corpus[corpus["sequence_id"].isin(manifest["sequence_id"])].copy()
    st.caption(f"{len(meta)} secuencias de muestra con limpieza compartida y landmarks normalizados. "
               "Las distribuciones de frases corresponden a esta muestra de test, no a todo el entrenamiento.")
else:
    meta = corpus
    st.caption("Metadatos originales de train.csv para contextualizar el EDA. La limpieza de landmarks se muestra abajo.")
by_split = corpus.groupby("split")["participant_id"].nunique()
st.html(
    '<p class="figline">'
    f'<b>{len(meta):,}</b> frases deletreadas por <b>{meta["participant_id"].nunique()}</b> firmantes, '
    f'con <b>{int(meta["phrase_length"].sum()):,}</b> caracteres en esta selección. La partición completa separa firmantes: '
    f'<b>{by_split["train"]}</b> para entrenar, <b>{by_split["val"]}</b> para validar y <b>{by_split["test"]}</b> para la prueba final.</p>'
)

# ------------------------------------------------------------------ caracteres
chars = pd.Series(list(meta["phrase"].str.cat()))
shares = chars.map(category_of).value_counts(normalize=True)
rare = chars.value_counts(normalize=True)
rare_symbols = [c for c in rare.index if category_of(c) == "símbolos" and rare[c] < 0.001]
st.header(f"Las letras son {shares.get('letras', 0):.0%} del texto; los símbolos, {shares.get('símbolos', 0):.1%}")
st.markdown(
    f"Los dígitos ocupan {shares.get('dígitos', 0):.0%}; algunas frases son teléfonos y direcciones. "
    f"{len(rare_symbols)} símbolos aparecen en menos de una de cada mil posiciones de esta selección."
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
st.markdown(f"Van de {lengths.min()} a {lengths.max()} caracteres. CTC necesita al menos un frame por carácter "
            "y uno adicional por cada repetición adyacente; una secuencia demasiado corta no puede alinearse.")
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
labels = {row.file: row.phrase for row in manifest.itertuples()}
choice = st.selectbox("Secuencia de prueba", list(labels), format_func=labels.get, index=min(2, len(labels) - 1), key="explore-seq")
raw = load_sample(choice)
prepared = prepared_sample(choice)
coverage = manifest.set_index("file").loc[choice, "coverage"]
hand = "derecha" if prepared.hand == "right_hand" else "izquierda"
st.markdown(f"La mano activa es la **{hand}** y queda útil en **{coverage:.0%}** de los frames. En negro, los frames donde "
            "MediaPipe detectó cada parte del cuerpo; los huecos son pérdidas de tracking que el preprocesamiento recorta, "
            "interpola o marca como inválidas.")
st.plotly_chart(charts.presence_strip(raw), config={"displayModeBar": False})
st.markdown("Trayectoria de la punta del índice respecto a la muñeca, ya normalizada. Los cambios bruscos pueden "
            "reflejar movimientos de la mano o pérdidas de tracking; no delimitan por sí solos una letra.")
st.plotly_chart(charts.trajectory(prepared, theme.MODEL_COLORS["m1"]), config={"displayModeBar": False})
if prepared.mask.any() and st.toggle("Mostrar los canales normalizados de la mano", value=False):
    st.caption("Distancia de cada landmark a la muñeca, en escalas de palma. El azul más oscuro indica una magnitud mayor.")
    st.plotly_chart(charts.hand_distances(prepared), config={"displayModeBar": False})
if prepared.mask.any() and st.toggle("Mostrar el esqueleto animado", value=False):
    st.plotly_chart(charts.playback(prepared, None, theme.INK), config={"displayModeBar": False})

with st.expander(f"Cobertura de la mano en las {len(manifest)} secuencias de muestra"):
    cov = manifest["coverage"]
    st.markdown(f"Mediana **{cov.median():.0%}**; {int((cov < 0.5).sum())} de {len(cov)} quedan por debajo del 50% "
                "que el análisis exploratorio proponía como umbral de descarte.")
    st.bar_chart(manifest.sort_values("coverage").set_index("phrase")["coverage"], color=theme.INK,
                 horizontal=True, height=560, x_label="cobertura de la mano activa", y_label="")
    quality = pd.DataFrame([{"mano": "derecha" if (p := prepared_sample(row.file)).hand == "right_hand" else "izquierda",
                             "cobertura": float(p.mask.mean()) if len(p.mask) else 0.0}
                            for row in manifest.itertuples()])
    st.caption("Calidad por mano activa en las muestras, después de la limpieza; no representa todo el corpus.")
    st.dataframe(quality.groupby("mano").agg(muestras=("cobertura", "size"),
                                            cobertura_mediana=("cobertura", "median")),
                 column_config={"cobertura_mediana": st.column_config.NumberColumn(format="percent")})
