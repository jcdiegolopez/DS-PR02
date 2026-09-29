"""App de demostración: reconocimiento de fingerspelling de ASL con modelos CTC.

Ejecutar desde la raíz del repositorio:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))  # para importar src/; app/ ya está en sys.path[0]

import streamlit as st

st.set_page_config(page_title="ASL a texto", page_icon=":material/closed_caption:", layout="wide")

from lib import theme
from lib.data import MODEL_INFO, available_models

theme.apply()

pages = [
    st.Page("views/explorar.py", title="Explorar datos", icon=":material/dataset:", url_path="explorar"),
    st.Page("views/predecir.py", title="Predecir", icon=":material/closed_caption:", url_path="predecir",
            default=True),
    st.Page("views/rendimiento.py", title="Rendimiento", icon=":material/speed:", url_path="rendimiento"),
]

with st.sidebar:
    st.markdown("### ASL a texto")
    ready = available_models()
    pending = [MODEL_INFO[k]["label"] for k in MODEL_INFO if k not in ready]
    st.html(
        '<p class="sidebar-note">Deletreo manual de ASL leído desde landmarks de MediaPipe '
        "con modelos CTC. CC3084 Data Science, reto Google ASL Fingerspelling.</p>"
        f'<p class="sidebar-note">Modelos listos: {", ".join(MODEL_INFO[k]["label"] for k in ready) or "ninguno"}'
        + (f"<br>Pendientes: {', '.join(pending)}" if pending else "") + "</p>"
    )

st.navigation(pages, position="sidebar").run()
