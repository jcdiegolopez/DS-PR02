# Presentación y demostración

## Preparación

Desde la raíz del repositorio, instalar `app/requirements.txt` y ejecutar `python -m streamlit run app/app.py`. Abrir `http://localhost:8501`. Confirmar que aparece M1 BiGRU como disponible y abrir la presentación en `reports/presentacion/ASL_resultados.pptx`. Mantener descargados el PDF de las diapositivas y `reports/demo_app.mp4`. La ejecución requiere CPU, las muestras incluidas y los checkpoints locales; no necesita Kaggle ni descargar datos grandes.

Antes de exponer, cerrar tareas de cómputo intensivas. La latencia de una demostración no se compara directamente con los tiempos por lotes del entrenamiento. Si se incorporan nuevos pesos o métricas, reiniciar la app y realizar el recorrido otra vez. No reemplazar archivos de modelos mientras se expone.

## Guion de las diapositivas (8 minutos)

| Diapositiva | Tiempo | Explicación |
|---|---|---|
| 1. Reconocimiento de secuencias | 20 s | Presentar el objetivo del proyecto: transcribir deletreo manual de ASL. |
| 2. Secuencias de landmarks a texto | 35 s | Aclarar entrada, salida y falta de alineación por carácter. |
| 3. Limpieza compartida | 45 s | Explicar mano activa, huecos y normalización; destacar que la app usa el mismo código del entrenamiento. |
| 4. Partición por participante | 35 s | Explicar el aislamiento de firmantes y el papel distinto de train, val y test. |
| 5. Arquitecturas | 35 s | Comparar recurrente, convolucional y atención; indicar cuáles tienen artefactos finales disponibles. |
| 6. BiGRU | 40 s | Presentar NED, CER, coincidencia exacta y parámetros como resultados de validación. |
| 7. Error por carácter | 40 s | Señalar el error de espacios y explicar por qué la frecuencia no es una explicación suficiente. |
| 8. Explorar datos | 30 s | Mostrar muestras limpias, corpus original, cobertura y trayectorias. |
| 9. Predicción | 75 s | Hacer la demostración en vivo descrita abajo. |
| 10. Color y legibilidad | 30 s | Justificar paleta categórica, escala secuencial y marcas adicionales al color. |
| 11. Rendimiento y costo | 40 s | Mostrar controles para ocultar gráficas y el benchmark local. |
| 12. Demostración y límites | 35 s | Cerrar con funcionamiento actual, alcance de landmarks y dependencia de los artefactos finales. |

## Recorrido en vivo

1. Abrir **Predecir**, conservar **Muestras de prueba** y elegir `109-580-5359`. Seleccionar M1 BiGRU.
2. Pulsar **Reproducir**. Explicar que el subtítulo se forma mediante decodificación CTC y no por una etiqueta manual por frame.
3. Contrastar `109-580-5359` con `109-580-5354`: una sustitución, 0,0833 de distancia normalizada. Activar la rejilla para identificar el último dígito y descargar el CSV si se desea conservar el resultado.
4. Activar y desactivar **Mostrar el preprocesamiento**. Explicar que es un diagnóstico opcional: la limpieza ya ocurrió automáticamente.
5. Abrir **Explorar datos**. Mostrar el selector de muestras limpias / corpus original. Bajar a **Una secuencia por dentro**, consultar trayectoria, canales normalizados y esqueleto. Aclarar que las 30 muestras no representan toda la distribución del corpus.
6. Abrir **Rendimiento**. Señalar explícitamente que los resultados locales de M1 son de validación. Mostrar la categoría de espacios y alternar los controles de las gráficas.
7. Pulsar **Medir en esta máquina**. Explicar calentamiento y promedio en CPU; distinguirlo de los tiempos por lotes. No presentar una latencia local como una constante universal.

## Respaldo ante una falla en vivo

Abrir `reports/demo_app.mp4`. La grabación muestra el checkpoint real, la animación, la exploración y la medición de CPU. No tiene audio: usar este mismo guion durante su reproducción. Si falla el reproductor, utilizar las capturas de `reports/capturas/` y la versión PDF de las diapositivas.

No intentar instalar dependencias o reentrenar durante la exposición. Indicar qué artefactos están disponibles y cuáles faltan. El ejemplo grabado no equivale a una evaluación final del test ni demuestra la superioridad de una arquitectura frente a las otras dos.

## Ensayo

Verificar las tres páginas, escoger otra muestra, comprobar el mensaje para un archivo inválido y confirmar que los controles ocultan y muestran las gráficas. Si se dispone de varios modelos reales, seleccionar cada uno y después todos. Mantener las entradas de `metrics.json` con su split real. La comparación final requiere los checkpoints de M2/M3 y sus resultados de test; no hay resultados sustitutos en la presentación.
