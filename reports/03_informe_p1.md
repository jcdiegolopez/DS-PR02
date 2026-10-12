# Informe final, secciones de la Persona 1

Reconocimiento de *fingerspelling* de ASL a partir de landmarks de MediaPipe
CC3084 Data Science, Proyecto 2: Resultados. Reto 15, *Google – American Sign Language Fingerspelling Recognition*.

> Este archivo contiene la introducción, los objetivos, el marco teórico, la metodología de datos y los resultados del conjunto de datos, la limpieza y el análisis exploratorio. La selección de algoritmos, la comparación de modelos, la sección de la aplicación y las conclusiones las escriben las otras personas. Las referencias completas están en `02_referencias.md`.

---

## 1. Introducción

La Organización Mundial de la Salud (2021) estima que más de 430 millones de personas necesitan rehabilitación por pérdida de audición y que la cifra superará los 700 millones en 2050. Para buena parte de la comunidad sorda, la lengua materna es una lengua de señas, y la tecnología de uso diario no la entiende: los asistentes de voz, los buscadores y los teclados predictivos están pensados para quien habla o escribe con soltura una lengua oral. Bragg et al. (2019) señalan que el reconocimiento de lenguas de señas sigue lejos de la madurez del reconocimiento de voz, en parte por la escasez de datos y en parte porque el problema combina visión por computadora, lingüística y accesibilidad.

Dentro de la Lengua de Señas Americana (ASL), el deletreo manual o *fingerspelling* tiene una función propia: representa palabras letra por letra con configuraciones de una sola mano. Se usa para nombres propios, direcciones, números de teléfono, correos, direcciones web y términos que no tienen una seña establecida. Padden y Gunsauls (2003) documentan que ocupa entre el 12 % y el 35 % del discurso en ASL. Es, además, justo el tipo de contenido que una persona necesita introducir en un teléfono, así que un sistema que no lo resuelva bien no sirve para la comunicación cotidiana.

El problema es difícil por tres razones. Primero, las letras no se producen aisladas: la forma de la mano en una letra depende de la anterior y de la siguiente (coarticulación), y los firmantes fluidos deletrean varias letras por segundo. Segundo, los datos no vienen segmentados: se conoce la frase completa de cada video, pero no en qué instante ocurre cada letra. Tercero, hay mucha variación entre firmantes en velocidad, lateralidad y estilo (Shi et al., 2018, 2019).

Este proyecto trabaja con el conjunto de datos del reto de Kaggle *Google – American Sign Language Fingerspelling Recognition* (Chow et al., 2023), recolectado con teléfonos por firmantes sordos y descrito por Georg et al. (2024). Los videos no se distribuyen: cada cuadro llega ya convertido en 543 puntos del cuerpo estimados con MediaPipe (Lugaresi et al., 2019). La tarea es convertir esa secuencia de coordenadas en la frase deletreada.

En la primera fase del proyecto se hizo el análisis exploratorio, que definió cómo limpiar los datos. En esta segunda fase se construye el sistema completo: una función de preprocesamiento única, tres modelos de reconocimiento de secuencias entrenados con la pérdida CTC, su comparación por distancia de edición, y una aplicación que recibe una secuencia cruda y muestra la transcripción de cada modelo.

## 2. Objetivos

### 2.1 Objetivo general

Desarrollar y evaluar un sistema que transcriba a texto secuencias de *fingerspelling* de ASL representadas como landmarks de MediaPipe, comparando tres arquitecturas de reconocimiento de secuencias entrenadas con CTC y poniendo el resultado al alcance de un usuario mediante una aplicación interactiva.

### 2.2 Objetivos específicos

1. **Implementar un preprocesamiento único y reproducible** que aplique las decisiones del análisis exploratorio (selección de canales, mano activa, invalidación de artefactos, recorte, interpolación limitada, normalización y máscara de validez) y que produzca exactamente las mismas características en el entrenamiento y en la aplicación.
2. **Construir los conjuntos de entrenamiento, validación y prueba separando firmantes**, de modo que ningún participante aparezca en más de un conjunto y la evaluación mida el desempeño con personas que el modelo no ha visto.
3. **Entrenar tres modelos con pérdida CTC** (una red recurrente bidireccional, una red convolucional temporal y un codificador basado en atención) sobre el corpus completo de 68 archivos.
4. **Comparar los modelos** con la distancia de edición normalizada como métrica principal y, como métricas secundarias, la tasa de error por carácter, la proporción de frases exactas, el error por tipo de carácter, el tiempo de inferencia y el número de parámetros.
5. **Relacionar los errores de los modelos con los hallazgos del análisis exploratorio**, en particular con el desbalance entre tipos de carácter y con la calidad de la señal de la mano.
6. **Entregar una aplicación** que reciba una secuencia cruda en el formato original, la preprocese sin intervención del usuario y presente la predicción de uno o varios modelos junto con su rendimiento.

## 3. Marco teórico

### 3.1 El deletreo manual en ASL

El alfabeto manual de ASL asigna una configuración de la mano a cada letra del alfabeto latino. Padden y Gunsauls (2003) describen cómo este sistema, tomado de la escritura, se integró a la lengua hasta volverse un recurso habitual y no un simple auxiliar. Para el reconocimiento automático importan tres propiedades. Es unimanual, así que una mano concentra casi toda la información. Es rápido y continuo, por lo que las configuraciones se funden unas con otras. Y su vocabulario es abierto: lo que se deletrea suele ser precisamente lo que no está en un diccionario, como nombres, números o direcciones.

### 3.2 Reconocimiento automático de lenguas de señas

El reconocimiento de señas pasó de sistemas estadísticos con características diseñadas a mano a redes neuronales profundas. Koller et al. (2015) muestran el enfoque estadístico para reconocimiento continuo con varios firmantes y vocabulario grande, y hacen explícito el problema que sigue vigente: la variación entre personas. La revisión de Rastgoo et al. (2021) ordena los trabajos con aprendizaje profundo según la entrada que usan (imagen, profundidad o esqueleto) y el tipo de tarea (señas aisladas o continuas). Camgoz et al. (2018) plantearon la traducción de señas como un problema de secuencia a secuencia, y Camgoz et al. (2020) lo resolvieron con Transformers entrenando de forma conjunta el reconocimiento y la traducción con una pérdida CTC intermedia.

Para *fingerspelling* en condiciones reales, Shi et al. (2018) recopilaron videos de ASL de internet y mostraron que el reconocimiento "en el mundo real" es mucho más difícil que en estudio. Shi et al. (2019) mejoraron el resultado con un mecanismo de atención visual que se acerca iterativamente a la mano, sin depender de un detector de manos aparte. Shi et al. (2021) abordaron después el problema previo de detectar en qué tramos de un video se está deletreando. El conjunto que usa este proyecto es de otra escala: Georg et al. (2024) reportan más de tres millones de caracteres deletreados por firmantes sordos frente a la cámara de un teléfono.

### 3.3 De la imagen a los puntos clave: estimación de pose y MediaPipe

Un video es una entrada muy costosa: cada cuadro tiene cientos de miles de pixeles y casi todos son fondo. La estimación de pose reduce cada cuadro a las coordenadas de unas cuantas articulaciones. OpenPose (Cao et al., 2021) mostró que es posible estimar en tiempo real los puntos del cuerpo, las manos y la cara de varias personas a partir de una imagen común. MediaPipe (Lugaresi et al., 2019) es un marco de trabajo para armar este tipo de flujos de percepción y ejecutarlos en dispositivos móviles. Su modelo de manos (Zhang et al., 2020) primero detecta la palma y luego regresa 21 puntos tridimensionales de la mano.

Los datos del reto se generaron con la solución *Holistic* de MediaPipe, que entrega 543 puntos por cuadro: 468 de la cara, 33 del cuerpo y 21 de cada mano. Cada punto tiene coordenadas *x* e *y* normalizadas al ancho y alto de la imagen, y una *z* de profundidad relativa. Trabajar con puntos tiene dos ventajas: la entrada es pequeña y no contiene la imagen de la persona. Tiene también un costo, que el análisis exploratorio midió: cuando el detector no encuentra una mano, ese cuadro llega vacío, y la calidad de todo lo demás depende de la calidad de esa estimación.

### 3.4 Modelos para secuencias

Las redes neuronales profundas aprenden representaciones en capas a partir de los datos, sin características diseñadas a mano (LeCun et al., 2015). Para secuencias hay tres familias que este proyecto compara.

**Redes recurrentes.** Procesan la secuencia paso a paso y mantienen un estado que resume lo visto. Las redes recurrentes simples olvidan con facilidad las dependencias largas; la LSTM (Hochreiter y Schmidhuber, 1997) lo resuelve con compuertas que controlan qué se guarda y qué se descarta. La GRU (Cho et al., 2014) es una variante con menos compuertas y menos parámetros. Una red bidireccional (Schuster y Paliwal, 1997) recorre la secuencia en ambos sentidos, de modo que la salida en cada instante conoce el contexto anterior y el posterior. Eso encaja con la coarticulación: la forma de una letra depende de las vecinas. Graves et al. (2013) mostraron que las redes recurrentes profundas y bidireccionales entrenadas de extremo a extremo son competitivas en reconocimiento de voz, un problema con la misma estructura que este.

**Redes convolucionales temporales.** Aplican convoluciones de una dimensión a lo largo del tiempo. Lea et al. (2017) las propusieron para segmentar acciones en video y mostraron que capturan patrones de larga duración con un entrenamiento más rápido que el de las recurrentes. Bai et al. (2018) compararon de forma sistemática convoluciones con dilatación contra redes recurrentes y encontraron que las primeras igualan o superan a las segundas en varias tareas de secuencias. Su ventaja práctica es que todos los instantes se calculan en paralelo.

**Atención y Transformers.** El Transformer (Vaswani et al., 2017) elimina la recurrencia: cada posición de la secuencia consulta directamente a todas las demás mediante autoatención. Captura dependencias globales, pero no tiene de por sí una noción de vecindad local. El Conformer (Gulati et al., 2020) combina autoatención con convoluciones para modelar a la vez lo global y lo local, y se volvió una referencia en reconocimiento de voz. Squeezeformer (Kim et al., 2022) reorganiza esa arquitectura para reducir su costo.

### 3.5 Clasificación temporal conexionista (CTC)

Los datos no dicen en qué cuadro ocurre cada letra. La clasificación temporal conexionista (Graves et al., 2006) permite entrenar sin esa alineación. El modelo emite, en cada uno de los *T* cuadros, una distribución sobre el vocabulario más un símbolo especial llamado *blank*. Una secuencia de *T* símbolos es un camino π, y la función *B* lo convierte en texto fusionando los símbolos repetidos consecutivos y eliminando después los *blank*. Muchos caminos distintos producen la misma frase **y**, y su probabilidad es la suma sobre todos ellos:

p(**y** | **x**) = Σ_{π ∈ B⁻¹(**y**)} Π_{t=1}^{T} p(π_t | **x**)

La pérdida es el logaritmo negativo de esa probabilidad, y se calcula de forma eficiente con programación dinámica. De la definición se sigue una restricción que se usó al preparar los datos: la entrada debe tener al menos tantos cuadros como caracteres tiene la frase, más uno por cada par de caracteres iguales consecutivos, porque entre ellos hace falta un *blank*. Para predecir se usa la decodificación voraz: se toma el símbolo más probable de cada cuadro y se aplica *B*.

### 3.6 Métrica de evaluación

La exactitud por clase no aplica cuando la salida es una cadena de longitud variable. La distancia de Levenshtein (1966) cuenta el número mínimo de inserciones, eliminaciones y sustituciones de un carácter que convierten una cadena en otra. La competencia usa la distancia de edición normalizada por la longitud de la frase real (Chow et al., 2023), y esa es la métrica principal del proyecto. La tasa de error por carácter (CER) divide las ediciones totales entre los caracteres totales del conjunto.

## 4. Metodología: datos

### 4.1 Fuente y estructura

Los datos provienen del reto de Kaggle (Chow et al., 2023). Tienen dos niveles unidos por `sequence_id`:

| Archivo | Una fila es | Contenido |
|---|---|---|
| `train.csv` | una secuencia | ruta, `file_id`, `sequence_id`, `participant_id`, `phrase` |
| `train_landmarks/*.parquet` (68 archivos) | un cuadro | `sequence_id`, `frame` y 1,629 coordenadas |
| `character_to_prediction_index.json` | un carácter | índice de cada carácter del vocabulario |

El corpus tiene 67,208 secuencias de 94 participantes y ocupa cerca de 190 GB.

### 4.2 Conjuntos de entrenamiento, validación y prueba

La partición se hizo **por participante**, no por secuencia. El análisis exploratorio encontró que el 38 % de la varianza en la velocidad de deletreo se explica por el firmante; si la misma persona estuviera en entrenamiento y en prueba, el modelo podría aprender su estilo y la evaluación sería optimista. Los 94 participantes se barajaron con una semilla fija (42) y se repartieron 80 % / 10 % / 10 %:

| Conjunto | Participantes | Secuencias | % de secuencias | Longitud media de la frase |
|---|---|---|---|---|
| Entrenamiento | 75 | 54,377 | 80.9 | 17.78 |
| Validación | 9 | 6,894 | 10.3 | 17.84 |
| Prueba | 10 | 5,937 | 8.8 | 18.00 |

La partición se define sobre `train.csv`, así que es la misma sin importar cuántos archivos se procesen. El conjunto de validación se usa para elegir el mejor punto de entrenamiento y para ajustar hiperparámetros. El de prueba se reserva para una sola evaluación final.

### 4.3 Preprocesamiento

Todo el preprocesamiento vive en una función, `preprocess_sequence` (`src/preprocessing.py`), que recibe una secuencia cruda y devuelve una matriz de características y una máscara de validez. El entrenamiento y la aplicación llaman a la misma función. Los pasos, en orden, son:

1. **Selección de canales.** Se leen las dos manos (126 coordenadas) y siete puntos del cuerpo en *x* e *y*: nariz, hombros, codos y muñecas. La malla facial se descarta: es el 86.2 % de las columnas y no participa en el deletreo.
2. **Mano activa.** Se elige la mano con mayor proporción de cuadros detectados. Si es la izquierda, la secuencia se refleja horizontalmente para que todas parezcan diestras.
3. **Invalidación de artefactos.** Se calcula el desplazamiento del centroide de la mano entre cuadros consecutivos. Un cuadro se invalida solo si es atípico por dos criterios a la vez: la puntuación *z* robusta basada en la desviación absoluta mediana supera 3.5 (Iglewicz y Hoaglin, 1993) y el desplazamiento supera 0.15 unidades normalizadas por cuadro, un salto físicamente imposible para una mano.
4. **Recorte.** Se eliminan los tramos sin mano al inicio y al final.
5. **Interpolación limitada.** Los huecos internos de hasta 5 cuadros se rellenan por interpolación lineal. Los más largos se dejan vacíos.
6. **Normalización.** La mano se centra en la muñeca y se divide por la distancia entre la muñeca y la base del dedo medio, lo que elimina la posición y el tamaño aparente. La posición de la muñeca y los puntos del cuerpo se expresan respecto al punto medio de los hombros y se dividen por el ancho de hombros.
7. **Máscara de validez.** Se marca cada cuadro como válido o no. Los no válidos se rellenan con cero solo para que la matriz sea numérica; el modelo los ignora.

El resultado son **80 características por cuadro**: la mano activa normalizada (63), la posición de la muñeca respecto al cuerpo (2), los siete puntos del cuerpo (14) y un indicador de si la otra mano aparece (1).

Dos decisiones se dejaron abiertas a propósito. El umbral de descarte por cobertura (el análisis exploratorio proponía exigir 50 % de cuadros válidos) no se aplica al preparar los datos: se guarda la cobertura de cada secuencia y el umbral se decide al entrenar. Tampoco se descartan aquí las secuencias con menos cuadros que los que exige CTC; se marcan, y el cargador de datos las excluye.

### 4.4 Almacenamiento

Las secuencias tienen longitudes distintas. Se guardan concatenadas en un solo arreglo en `float16`, con un índice que registra dónde empieza y cuánto mide cada una. El corpus procesado ocupa 1.6 GB frente a los 190 GB originales.

### 4.5 Herramientas y cómputo

| Componente | Herramienta |
|---|---|
| Lenguaje | Python 3 |
| Lectura por columnas de los Parquet | PyArrow |
| Cálculo numérico y tablas | NumPy (Harris et al., 2020), pandas, SciPy |
| Modelos y entrenamiento | PyTorch (Paszke et al., 2019) |
| Gráficas del informe | Matplotlib, seaborn |
| Aplicación | Streamlit, Plotly |
| Control de versiones | Git y GitHub |

El preprocesamiento del corpus completo se ejecutó en un notebook de Kaggle, donde la competencia ya está montada como fuente de datos y no hace falta descargar nada. El notebook clona el repositorio para usar el mismo código de `src/`. Como solo se leen 141 de las 1,631 columnas de cada archivo, las 67,208 secuencias (10.6 millones de cuadros) se procesaron en unos 9 minutos de CPU. El primer modelo se entrenó en una GPU local a partir de esas salidas.

*Pendiente de completar por quien entrenó: modelo de la GPU y tiempo de entrenamiento de cada modelo.*

## 5. Resultados: conjunto de datos, limpieza y análisis exploratorio

Las cifras del corpus de frases se calcularon sobre las 67,208 secuencias de `train.csv`. Las de la señal provienen del análisis exploratorio de un archivo oficial completo (1,000 secuencias, 162,699 cuadros), salvo donde se indica que corresponden al corpus completo.

### 5.1 El corpus de frases

- **Tamaño.** 67,208 secuencias, 46,478 frases distintas y cerca de 1.2 millones de caracteres.
- **Longitud.** Las frases miden entre 1 y 31 caracteres, con media 17.8, mediana 17 y desviación estándar 5.7. La mitad central va de 12 a 22.
- **Contenido.** No es lenguaje común. Según una clasificación aproximada por reglas sobre el texto, cerca del 32 % de las frases son direcciones, el 28 % direcciones web y el 24 % números de teléfono; el 16 % restante es otro tipo de texto, como nombres.
- **Vocabulario.** Aparecen 59 caracteres. Las letras son el 61.2 % del texto, los dígitos el 24.6 %, los símbolos el 9.3 % y los espacios el 4.9 %. El desbalance está en la cola: 17 caracteres aparecen en menos de una de cada mil posiciones, y varios símbolos (`$`, `#`, `!`, `*`, `[`, `;`) casi no aparecen.
- **Participantes.** Cada participante aporta en mediana 795 secuencias; el que más, 1,535, y el que menos, 1. El coeficiente de Gini de los aportes es 0.16, es decir, ningún firmante domina el corpus.

El peso de los dígitos es una consecuencia directa del contenido: en este corpus los números no son una clase rara, como sí lo son los símbolos poco usados.

### 5.2 Estructura y calidad de la señal

- **Composición.** De las 1,629 coordenadas por cuadro, 1,404 son de la cara (86.2 %), 99 del cuerpo (6.1 %) y 126 de las manos (7.7 %). Leer solo manos y cuerpo reduce la memoria al 13.8 %.
- **La ausencia es por bloque.** MediaPipe entrega los 21 puntos de una mano o ninguno; no se encontraron detecciones parciales. El cuerpo aparece en todos los cuadros.
- **Una sola mano trabaja.** La mano no dominante está ausente en más del 90 % de los cuadros en el 99.2 % de las secuencias. No es un defecto: es la huella de una actividad unimanual. La mano activa es la derecha en el 79.6 % de las secuencias y la izquierda en el 20.4 %.
- **La mano activa también se pierde.** En promedio falta en el 44 % de los cuadros de una secuencia, y en el 42.3 % de las secuencias falta en más de la mitad. El 41.6 % de los cuadros no tiene ninguna mano detectada. Ninguna secuencia carece por completo de ambas manos.
- **Los huecos son cortos pero pesan poco.** La mediana de un hueco es de 3 cuadros y el 67.9 % mide 5 o menos, pero esos huecos cortos solo suman el 22 % de los cuadros ausentes. Unos pocos huecos largos (hasta 324 cuadros) concentran el resto.

### 5.3 Qué estrategia de relleno usar

Para decidir cómo tratar los huecos se hizo un experimento con respuesta conocida: en tramos donde la mano sí estaba, se ocultaron huecos de 1 a 21 cuadros y se midió el error de cada estrategia al reconstruirlos (2,807 huecos en 401 secuencias).

| Estrategia | Error absoluto medio | Veces peor que la mejor |
|---|---|---|
| Interpolación lineal | 0.0142 | 1.0 |
| Repetir el último valor | 0.0209 | 1.5 |
| Media de la secuencia | 0.0304 | 2.1 |
| Rellenar con ceros | 0.3713 | 26.2 |

Rellenar con ceros es 26 veces peor que interpolar, porque en coordenadas normalizadas el cero es una esquina de la imagen y equivale a teletransportar la mano. El error de la interpolación crece con la longitud del hueco (de 0.007 con un cuadro a 0.025 con 21). De ahí salen las dos reglas del preprocesamiento: interpolar solo huecos de hasta 5 cuadros y marcar el resto como no válido en lugar de inventarlo.

### 5.4 Artefactos de seguimiento

La velocidad mediana del centroide de la mano es de 0.009 unidades por cuadro y el percentil 99.9 es 0.109. El criterio estadístico por sí solo marcaría el 5.3 % de las transiciones, demasiadas para ser errores. El criterio físico (más de 0.15) marca el 0.028 %, y todas coinciden con el estadístico. Exigir ambos deja 26 transiciones en 23 secuencias. Son pocos casos, y por eso se invalida el cuadro en vez de descartar la secuencia.

### 5.5 Dimensión temporal

- **Duración.** Las secuencias miden en mediana 148 cuadros, con un rango de 1 a 672 y asimetría positiva. Rellenar todas hasta la longitud máxima desperdiciaría la mayor parte del cómputo, así que los lotes se rellenan solo hasta la secuencia más larga de cada lote.
- **Relación con el texto.** La duración crece con la longitud de la frase: correlación de Pearson de 0.62 y una pendiente de 9.2 cuadros por carácter (unos 0.3 segundos a 30 cuadros por segundo).
- **El firmante importa.** El firmante más lento usa 3.5 veces más cuadros por carácter que el más rápido (16.0 contra 4.6). La prueba de Kruskal-Wallis rechaza que la velocidad sea igual entre firmantes (H = 398.2, 89 grados de libertad, p < 0.001), y el firmante explica el 38.2 % de la varianza. Este resultado es el que justifica partir los datos por participante.

### 5.6 Resultado de la preparación sobre el corpus completo

Al ejecutar el preprocesamiento sobre los 68 archivos se obtuvo lo siguiente:

1. **La calidad depende del firmante.** La cobertura media de la mano activa por participante va de 33 % a 94 %. El conjunto de validación, con solo 9 participantes, quedó con una cobertura media de 59 %, menor que la de entrenamiento (69 %) y prueba (70 %). La validación es, por lo tanto, algo más difícil que la prueba, y hay que tenerlo presente al comparar sus métricas.
2. **El umbral de descarte tiene un costo alto.** Exigir 50 % de cobertura conserva el 71.1 % del corpus; exigir 20 % conserva el 90.5 %. Como el modelo recibe la máscara de validez, el umbral se trata como un hiperparámetro del entrenamiento.
3. **CTC es viable en casi todo el corpus.** La mediana es de 8.8 cuadros por carácter. Solo 2,527 secuencias (3.8 %) tienen menos cuadros que los que su frase exige y se excluyen del entrenamiento.
4. **Datos efectivos de entrenamiento.** Con cobertura mínima de 50 % y la condición de CTC, el primer modelo se entrenó con 37,366 secuencias y se validó con 3,857.
5. **Consistencia entre entrenamiento y aplicación.** Una secuencia cruda exportada para la aplicación produce exactamente las mismas características que las guardadas para entrenar. Se comprobó en local y en Kaggle.

### 5.7 Limitaciones

- El análisis de la señal de las secciones 5.2 a 5.5 se hizo sobre un archivo de los 68. Las cifras de la sección 5.6 sí corresponden al corpus completo y son coherentes con las del archivo (por ejemplo, 71.3 % de retención en el archivo y 71.1 % en el corpus).
- El conjunto de validación tiene 9 participantes y el de prueba 10. Con tan pocos firmantes, un solo participante de mala calidad mueve las métricas.
- Todo el sistema hereda los errores de MediaPipe: si el detector no encuentra la mano, no hay nada que el modelo pueda recuperar.
- La cara se descartó. En ASL la boca y la expresión aportan información, aunque su papel en el deletreo es secundario; no se midió cuánto se pierde.
