"""Predecir: secuencia cruda -> preprocesamiento invisible -> transcripción por modelo."""

from __future__ import annotations

from html import escape

import streamlit as st

from lib import captions, charts, theme
from lib.data import MODEL_INFO, InputError, available_models, load_manifest, load_sample, read_upload
from lib.inference import prepare, transcribe


@st.cache_data(show_spinner=False)
def prepared_sample(file_name: str):
    return prepare(load_sample(file_name))


st.title("Del deletreo al texto")
st.markdown("Elige una secuencia de un firmante que ningún modelo vio al entrenar, o sube la tuya en el "
            "formato de Kaggle. El archivo entra crudo y la limpieza ocurre sola, con el mismo código del entrenamiento.")

ready = available_models()
manifest = load_manifest()

controls, stage = st.columns([1, 2.3], gap="large")

with controls:
    source = st.segmented_control("Origen de la secuencia", ["Muestras de prueba", "Subir archivo"],
                                  default="Muestras de prueba")
    reference, prepared, tag = "", None, ""
    try:
        if source == "Subir archivo":
            upload = st.file_uploader("Archivo Parquet o CSV de una secuencia", type=["parquet", "csv"],
                                      help="Columnas como x_right_hand_0 ... z_right_hand_20; la pose y la cara son opcionales.")
            reference = st.text_input("Frase real (opcional, para comparar)", max_chars=120).strip().lower()
            if upload is not None:
                raw = read_upload(upload.name, upload.getvalue())
                prepared = prepare(raw)
                tag = escape(upload.name)
        else:
            labels = {row.file: row.phrase for row in manifest.itertuples()}
            choice = st.selectbox("Secuencia de prueba", list(labels), format_func=labels.get, index=2)
            row = manifest.set_index("file").loc[choice]
            reference = row.phrase
            prepared = prepared_sample(choice)
            tag = f"participante {row.participant_id}"
    except InputError as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - el público nunca debe ver una traza
        st.error("La secuencia no se pudo procesar. Verifica que sea una sola secuencia en el formato de Kaggle.")

    if ready:
        chosen = st.pills("Modelos", ready, selection_mode="multi", default=ready,
                          format_func=lambda k: MODEL_INFO[k]["label"])
    else:
        chosen = []
        st.warning("Todavía no hay pesos de ningún modelo en models/.")
    pending = [MODEL_INFO[k]["label"] for k in MODEL_INFO if k not in ready]
    if pending:
        st.caption(f"{', '.join(pending)}: se agregan cuando estén entrenados.")

    show_steps = st.toggle("Mostrar el preprocesamiento", value=False)
    show_grid = st.toggle("Mostrar la rejilla carácter a carácter", value=False)

    if prepared is not None and show_steps:
        st.html('<div class="steps">' + "".join(
            f'<div class="row"><div class="n">{n}</div><div class="k">{name}</div><div class="v">{value}</div></div>'
            for n, name, value in prepared.steps) + "</div>")
        if prepared.missing_columns:
            st.caption(f"Faltaban {len(prepared.missing_columns)} columnas (pose u otras); se trataron como ausentes.")

with stage:
    if prepared is None:
        st.info("Sube un archivo para transcribirlo." if source == "Subir archivo" else "Elige una secuencia.")
        st.stop()
    if prepared.mask.sum() == 0:
        st.error("No se detectó ninguna mano en la secuencia, así que no hay nada que transcribir.")
        st.stop()
    if prepared.mask.sum() < max(len(reference), 8):
        st.warning("La mano aparece en muy pocos frames; la transcripción puede quedar incompleta.")

    results = [transcribe(key, prepared) for key in (chosen or [])]
    lead = results[0] if results else None
    lead_color = theme.MODEL_COLORS.get(lead.key, theme.INK) if lead else theme.INK
    st.plotly_chart(charts.playback(prepared, lead, lead_color), config={"displayModeBar": False},
                    key=f"play-{tag}-{lead.key if lead else ''}")

    if not results:
        st.info("Elige al menos un modelo para transcribir.")
        st.stop()

    for tr in results:
        label = MODEL_INFO[tr.key]["label"]
        if reference:
            dist = captions.edit_distance(reference, tr.text)
            right = (f"<span><b>{dist}</b> ediciones</span><span><b>{dist / max(1, len(reference)):.3f}</b> normalizada</span>"
                     f"<span><b>{tr.milliseconds:.1f}</b> ms</span>")
            st.html(captions.diff_caption(reference, tr.text, f"{theme.model_mark(tr.key, label)}<span>{tag}</span>",
                                          right, grid=show_grid))
        else:
            st.html(f'<div class="cap"><div class="cap-tag"><span>{theme.model_mark(tr.key, label)}</span><span><b>{tr.milliseconds:.1f}</b> ms</span></div>'
                    f'<div class="cap-line">{escape(tr.text) or "(sin texto)"}</div></div>')
    if reference:
        st.html(captions.legend())
