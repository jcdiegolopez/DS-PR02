# ASL Fingerspelling — resultados

El notebook [`notebooks/Proyecto2_Resultados_Master.ipynb`](notebooks/Proyecto2_Resultados_Master.ipynb) prepara las secuencias del reto de Kaggle y entrena el primer modelo BiGRU + CTC. El preprocesamiento es compartido por el notebook y la aplicación mediante `src/preprocessing.py`.

## Datos procesados

Para reutilizar la ejecución completa de preparación de datos en Kaggle, inicia sesión con la CLI y descarga sus salidas:

```powershell
kaggle kernels output diegolcc/asl-fingerspelling-resultados -p data --file-pattern "(features\.npy|masks\.npy|index\.csv|config\.json)$"
```

Los cuatro archivos deben quedar en `data/processed/`. Esta carpeta está excluida de Git. El kernel fuente es privado y requiere acceso a él.

## Entrenar y evaluar M1

```powershell
python -m pip install -r requirements.txt
python -m src.train --processed-dir data/processed --output-dir models --epochs 8 --batch-size 16
```

Si se interrumpe tras una época completa, el estado de entrenamiento se conserva en `models/m1_last.pt`. Para continuar hasta el número total de épocas indicado, agrega `--resume` al comando. Un checkpoint antiguo que solo contiene pesos también se puede retomar; en ese caso AdamW se reinicia.

El entrenamiento usa únicamente `train` para actualizar pesos y `val` para seleccionar el checkpoint. No evalúa `test`. Guarda `models/m1_bigru_ctc.pt`, `models/m1_history.csv` y `models/metrics.json`. Este último incluye distancia de edición normalizada, CER, coincidencia exacta, errores por tipo de carácter, latencia y número de parámetros. `src/models.py` expone `load_model` y `predict` para la aplicación.

La corrida inicial sobre el conjunto procesado completo usó 37 366 secuencias de entrenamiento y 3 857 de validación. El mejor checkpoint fue el de la época 8: distancia de edición normalizada **0,2435**, CER **0,2425**, coincidencia exacta **13,12 %** y **473 405** parámetros. La primera época se recuperó desde un checkpoint que solo tenía pesos, por lo que AdamW se reinició al comenzar la segunda. El historial conserva la pérdida de validación de esa primera época, pero no su pérdida de entrenamiento.

Pruebas de los contratos de CTC y del flujo de entrenamiento:

```powershell
python -m unittest discover -s tests -v
```

## Aplicación

```powershell
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

Se ejecuta desde la raíz del repositorio. Tiene tres páginas: **Explorar datos** (corpus de frases y una secuencia por dentro), **Predecir** (secuencia cruda de prueba o archivo subido, preprocesamiento invisible y transcripción por modelo) y **Rendimiento** (comparación desde `models/metrics.json`). Los modelos aparecen solos cuando existen sus pesos (`models/m2_*.pt`, `models/m3_*.pt`), su entrada en `metrics.json` y su soporte en `src.models.load_model`. Todo corre en CPU.

Para Streamlit Community Cloud, el archivo principal es `app/app.py` y las dependencias se leen de `app/requirements.txt`. La paleta y la tipografía están justificadas en [`reports/app_paleta.md`](reports/app_paleta.md).
