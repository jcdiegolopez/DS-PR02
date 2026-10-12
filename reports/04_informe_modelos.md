# Informe final: modelos, resultados y conclusiones

Reconocimiento de *fingerspelling* de ASL a partir de landmarks de MediaPipe
CC3084 Data Science, Proyecto 2: Resultados. Reto 15, *Google – American Sign Language Fingerspelling Recognition*.

> Este archivo contiene la metodología de modelado (selección de algoritmos, ajuste de hiperparámetros y evaluación), los resultados de los modelos y las conclusiones. Continúa a `03_informe_datos.md`. Las cifras provienen de `notebooks/Proyecto2_Resultados_Master.ipynb` (Partes II y III), ejecutado sobre el corpus completo en Kaggle; las figuras están en `reports/figures/`.

---

## 4. Metodología: modelos

### 4.6 Selección de los algoritmos

La revisión bibliográfica y el análisis exploratorio llevan a la misma formulación: el problema es de **reconocimiento de secuencias**, no de clasificación. Cada muestra es una serie de unos 160 cuadros de landmarks que debe transcribirse a una frase de longitud variable, y los datos no indican en qué cuadro empieza cada letra. Por eso los tres modelos comparten la salida y la función de pérdida. En cada cuadro emiten una distribución sobre los 59 caracteres del vocabulario oficial más el símbolo *blank*, y se entrenan con CTC (Graves et al., 2006). Lo único que cambia entre ellos es el codificador temporal, que es justo lo que la literatura discute para este tipo de problema:

| Modelo | Codificador | Justificación |
|---|---|---|
| **M1. BiGRU + CTC** | Dos capas GRU bidireccionales | Las redes recurrentes bidireccionales con CTC son la línea base clásica del reconocimiento de voz de extremo a extremo (Graves et al., 2013) y de *fingerspelling* en video (Shi et al., 2018). La GRU (Cho et al., 2014) tiene menos parámetros que la LSTM (Hochreiter y Schmidhuber, 1997). La bidireccionalidad (Schuster y Paliwal, 1997) permite que cada cuadro vea la letra anterior y la siguiente, lo que conviene a la coarticulación observada en el EDA. |
| **M2. TCN + CTC** | Convoluciones temporales dilatadas con conexiones residuales | Lea et al. (2017) y Bai et al. (2018) muestran que las convoluciones temporales igualan a las recurrentes en tareas de secuencias, con cálculo paralelo. El EDA mostró que la información está en transiciones locales entre configuraciones de mano, que una ventana convolucional captura directamente. |
| **M3. Transformer + CTC** | Encoder Transformer con codificación posicional sinusoidal | La autoatención (Vaswani et al., 2017) relaciona cualquier par de cuadros. Es la base de los Sign Language Transformers (Camgoz et al., 2020) y de las arquitecturas Conformer y Squeezeformer (Gulati et al., 2020; Kim et al., 2022) que dominaron la competencia. |

Se descartaron explícitamente dos alternativas:
- **Clasificadores de una letra por cuadro o por ventana** (por ejemplo, *random forest* o SVM sobre características agregadas): requieren saber dónde está cada letra, información que el conjunto de datos no tiene.
- **Modelos *Seq2Seq* con decodificador autoregresivo**: son más costosos de entrenar y de ejecutar en la aplicación, y no resuelven una limitación que CTC tenga en este problema, porque el texto sigue el mismo orden que el deletreo.

### 4.7 Configuración común

Para que la comparación aísle el efecto de la arquitectura, los tres modelos usan:

- **Las mismas entradas:** las 80 características por cuadro de la sección 4.3, y solo los cuadros que la máscara marca como válidos.
- **La misma partición por participante:** `train` actualiza los pesos, `val` elige hiperparámetros y checkpoints, y `test` se usa una sola vez al final.
- **El mismo entrenamiento:** optimizador AdamW, recorte del gradiente a norma 1, lotes de 32 secuencias con relleno dinámico y semilla 42.
- **La misma decodificación:** voraz, es decir, el símbolo más probable de cada cuadro seguido de la función *B* de CTC.
- **El mismo filtro:** se excluyen las secuencias que no cumplen la longitud mínima de CTC, el 3.8% del corpus.

### 4.8 Ajuste de hiperparámetros

El presupuesto de GPU disponible (Kaggle, una NVIDIA T4) no permite una búsqueda exhaustiva sobre 54,000 secuencias. Se usó una **búsqueda reducida y documentada**: cada configuración se entrenó 3 épocas sobre 5,000 secuencias de entrenamiento y se evaluó con la distancia de edición normalizada sobre el mismo conjunto de validación (cobertura ≥ 50%). Se exploraron tres ejes:

1. **Capacidad del codificador**, en los tres modelos: una versión base y una más grande (más unidades o canales y más capas).
2. **Tasa de aprendizaje** en M3 (10⁻³ frente a 5·10⁻⁴). Los Transformers son más sensibles a ella que las redes recurrentes.
3. **Umbral de cobertura** en M1 (20% frente a 50%). Esta decisión quedó pendiente en la preparación de datos: con 20% se entrena con el 90% del corpus, pero con secuencias más incompletas.

La configuración con menor distancia de edición en validación de cada arquitectura se entrenó luego con todos los datos. El entrenamiento final se detiene en la primera de tres condiciones: 15 épocas, 3 épocas seguidas sin mejorar la pérdida de validación (*early stopping*) o 30 minutos de GPU por modelo. Siempre se conserva el checkpoint con menor pérdida de validación.

### 4.9 Evaluación

Cada modelo final se evaluó una vez sobre `test`, con 10 participantes que no intervinieron ni en el entrenamiento ni en la selección de hiperparámetros. Se reportan:

- **Distancia de edición normalizada** (métrica principal; sección 3.6).
- **CER**, y el puntaje de la competencia `1 − CER`.
- **Proporción de frases exactas**.
- **Error por tipo de carácter** (letras, dígitos, símbolos y espacios), para contrastar con el desbalance del vocabulario hallado en el EDA.
- **Error según la longitud de la frase**.
- **Costo:** número de parámetros, tiempo de inferencia y tiempo de entrenamiento.

---

## 5. Resultados de los modelos

*(Pendiente: se completa con las cifras de la ejecución en Kaggle.)*

---

## 6. Conclusiones

*(Pendiente: se completa con las cifras de la ejecución en Kaggle.)*
