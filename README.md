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

Se ejecuta desde la raíz del repositorio. Tiene tres páginas: **Explorar datos** (muestras limpias, metadatos del corpus original, calidad por mano y canales normalizados), **Predecir** (secuencia cruda de prueba o archivo subido, preprocesamiento automático, animación y transcripción por uno o varios modelos) y **Rendimiento** (comparación desde `models/metrics.json`, controles para ocultar gráficas y benchmark local). Las predicciones y métricas se pueden descargar en CSV. Todo corre en CPU.

Un checkpoint cargable activa el modelo en **Predecir**. Su entrada en `metrics.json` activa sus resultados en **Rendimiento**; no es necesario tener métricas agregadas para transcribir. El repositorio contiene pesos reales de M1 y métricas de **validación**, no resultados finales de test. TCN y Transformer ya tienen soporte de carga e inferencia, pero requieren sus checkpoints finales. Los resultados de distintos splits o tamaños se muestran por separado.

Para Streamlit Community Cloud, el archivo principal es `app/app.py` y las dependencias se leen de `app/requirements.txt`. La paleta y la tipografía están justificadas en [`reports/app_paleta.md`](reports/app_paleta.md).

### Cargar nuevos datos

Subir un CSV o Parquet con las coordenadas `x_<mano>_<landmark>`, `y_<mano>_<landmark>` y `z_<mano>_<landmark>`, donde `<mano>` es `right_hand` o `left_hand` y `<landmark>` va de 0 a 20. Debe existir una mano completa; pose y la otra mano son opcionales. Los landmarks ausentes pueden tener valores vacíos/NaN. Si no existe `frame`, se usa el orden de las filas. Si hay varios `sequence_id`, la app permite seleccionar uno.

Se rechazan archivos vacíos o ilegibles, coordenadas no numéricas o infinitas, identificadores ausentes, frames repetidos y entradas de más de 10 000 frames. Una secuencia sin mano válida se informa sin ejecutar inferencia. La frase de referencia es opcional y se conserva exactamente como se ingresa para calcular distancia de edición.

### Incorporar checkpoints y métricas

Agregar los checkpoints finales como `models/m2_tcn_ctc.pt` y `models/m3_transformer_ctc.pt` y reiniciar la app. Deben seguir el contrato de `src.models.load_model`: `architecture`, `config` y `state_dict`. Los archivos terminados en `_last.pt` no se usan para la demostración. Conservar un solo checkpoint final por clave de modelo para evitar ambigüedad.

Las entradas `m1`, `m2` y `m3` de `models/metrics.json` deben contener `split` (`val` o `test`), `sequences`, `normalized_edit_distance`, `cer`, `exact_match`, `parameters` y `by_character_type`. Las categorías `letters`, `digits`, `symbols` y `spaces` incluyen `reference` y `error_rate`; `src.evaluate` produce este formato. Los historiales opcionales `models/mN_history.csv` contienen `epoch`, `train_loss` y `val_loss`. No cambiar el split de una evaluación de validación para presentarla como prueba.

El benchmark mide inferencia y decodificación en CPU sobre las muestras, con una llamada de calentamiento por modelo. Excluye lectura, carga y limpieza. No vuelve a calcular CER ni distancia de edición sobre test y no sobrescribe métricas.

### Informe, presentación y demo

- [Sección de la aplicación](reports/05_informe_aplicacion.md), con tecnologías, funcionamiento, capturas y límites actuales.
- [Sección de la aplicación en PDF](reports/aplicacion.pdf).
- [Presentación editable](reports/presentacion/ASL_resultados.pptx) y [versión PDF](reports/presentacion/ASL_resultados.pdf), con 12 diapositivas.
- [Guion y ensayo](reports/guion_demo.md) y [video de respaldo sin audio](reports/demo_app.mp4).

Para regenerar el PDF de la sección de aplicación después de editar su Markdown o las capturas:

```powershell
python -m pip install -r reports/requirements.txt
python reports/generar_pdf_aplicacion.py
```

Para verificar la app y los contratos compartidos, instalar también las dependencias raíz y ejecutar `python -m unittest discover -s tests -v`. Las pruebas de la app usan el checkpoint incluido, ejercitan las tres páginas y revisan entradas inválidas sin necesitar un navegador.
