# Reconocimiento Continuo de ASL Fingerspelling: Marco Teórico y Objetivos
## CC3084 – Data Science | Proyecto 2: Análisis Exploratorio
### Desafío: Google – American Sign Language Fingerspelling Recognition

---

## 1. Situación Problemática

### 1.1 Contexto Sociolingüístico y la Brecha de Comunicación
La comunicación interpersonal es un derecho humano fundamental y el eje vertebrador de la inclusión social, educativa y laboral. A nivel mundial, según la Organización Mundial de la Salud (OMS), más de 430 millones de personas requieren servicios de rehabilitación para la pérdida de audición, y se proyecta que esta cifra superará los 700 millones para el año 2050. Para la comunidad sorda e hipoacúsica (*Deaf and Hard of Hearing* - DHH), las lenguas de señas representan sus lenguas maternas y vehículos primarios de identidad cultural.

A pesar de los avances acelerados en tecnologías de reconocimiento automático del habla (*Automatic Speech Recognition* - ASR) y procesamiento de lenguaje natural (NLP) para lenguas orales, existe una asimetría crítica en el desarrollo de interfaces inclusivas basadas en visión computacional para lenguas viso-espaciales. Mientras que un usuario oyente puede interactuar fluidamente mediante voz con asistentes virtuales, motores de búsqueda y dispositivos móviles, las personas señantes enfrentan una persistente **barrera de accesibilidad digital**.

### 1.2 El Rol Crítico del *Fingerspelling* (Deletreo Manual) en ASL
En el Lenguaje de Señas Americano (*American Sign Language* - ASL), el deletreo manual (*fingerspelling*) es un subsistema lingüístico indispensable consistente en la representación ortográfica, letra por letra, del alfabeto latino mediante configuraciones manuales específicas. Estudios lingüísticos evidencian que el *fingerspelling* comprende entre el **12% y el 35% del discurso fluido cotidiano** en ASL, cumpliendo funciones semánticas irremplazables:
* **Préstamos léxicos y neologismos:** Incorporación de conceptos modernos, tecnológicos o científicos que carecen de una seña léxica estandarizada.
* **Nombres propios y toponimia:** Identificación de personas, ciudades, marcas y títulos.
* **Información estructurada y sensible:** Transmisión de direcciones físicas, códigos postales, contraseñas, correos electrónicos, URLs y números de teléfono.
* **Énfasis pragmático y desambiguación:** Aclaración de términos homónimos o especificación exacta de terminología legal y médica.

Por ende, un sistema de traducción de lenguaje de señas que no resuelva con alta fidelidad el *fingerspelling* resulta inviable para aplicaciones de interacción cotidiana y servicios de asistencia en tiempo real.

### 1.3 Desafíos Tecnológicos y de Visión por Computadora
El reconocimiento automático de *fingerspelling* continuo presenta una complejidad técnica significativamente mayor que el reconocimiento de señas aisladas:
1. **Co-articulación Severa:** En el deletreo fluido y rápido (donde señantes nativos alcanzan velocidades de hasta 5 a 6 letras por segundo), las transiciones entre configuraciones manuales provocan que la forma de una letra se solape biomecánicamente con la anterior y la posterior, alterando la morfología canónica de los dedos.
2. **Auto-oclusión y Ángulos de Cámara:** La grabación mediante cámaras frontales de teléfonos inteligentes (*selfie cameras*) genera oclusiones frecuentes donde la palma o dedos cercanos bloquean la visibilidad de las articulaciones traseras.
3. **Variabilidad Inter e Intra-firmante:** Existen notables diferencias biomecánicas asociadas a la lateralidad (señantes zurdos vs. diestros), flexibilidad articular, velocidad de ejecución, pausas idiosincrásicas y condiciones variables de iluminación y encuadre.
4. **Desbalance de Frecuencia de Caracteres:** El vocabulario incluye no solo caracteres alfabéticos de distribución asimétrica (ley de Zipf en lengua inglesa), sino también números y caracteres especiales (`@`, `/`, `-`, `.`), cuya frecuencia relativa es muy baja pero su impacto en el significado es determinante.

Para abordar este desafío, Google, en colaboración con el *Deaf Professional Arts Network* (DPAN), construyó el conjunto de datos de *fingerspelling* más extenso y ecológicamente válido hasta la fecha, extrayendo coordenadas tridimensionales de puntos clave (*landmarks*) mediante el framework de visión **MediaPipe Holistic**. La situación problemática radica en que la vasta dimensionalidad, la variabilidad temporal y el desbalance inherente de este corpus exigen un análisis exploratorio riguroso antes de pretender entrenar arquitecturas neuronales complejas.

---

## 2. Problema Científico

### 2.1 Formulación de la Pregunta de Investigación Central
A partir del diagnóstico de la situación descrita, se formula el siguiente problema científico:

> **¿Cómo varían las distribuciones estadísticas, la longitud y la diversidad léxica de las secuencias de deletreo manual en ASL entre diferentes participantes, y en qué medida la heterogeneidad muestral y el desbalance de frecuencias de los caracteres alfanuméricos y especiales condicionan la viabilidad y el diseño de pipelines de reconocimiento continuo basados en series temporales de coordenadas biomecánicas?**

### 2.2 Delimitación y Descomposición del Problema Científico
El problema científico planteado descompone la incertidumbre en tres vectores analíticos:
1. **Vector de Variabilidad Muestral e Inter-Sujeto:** Determinar el grado de concentración o dispersión del número de secuencias aportadas por los 147 participantes (`participant_id`), identificando si existen sesgos de representación que puedan inducir sobreajuste a estilos de firma particulares.
2. **Vector de Complejidad Textual y Léxica:** Evaluar la distribución de longitudes de frases (`phrase`), el tamaño efectivo del vocabulario (59 caracteres) y el grado de asimetría en la distribución de frecuencias unigramas y transicionales (bigramas/trigramas), comparándola contra la distribución del lenguaje natural escrito.
3. **Vector de Relación Estructural Texto-Secuencia:** Cuantificar la relación entre la extensión léxica de la frase y la variabilidad de tipologías (URLs, teléfonos, direcciones, lenguaje común), estableciendo pautas de tokenización y segmentación indispensables para las funciones de pérdida de alineamiento temporal (como *Connectionist Temporal Classification* - CTC o modelos *Seq2Seq* con atención).

---

## 3. Objetivos de la Investigación

### 3.1 Objetivo General
* **Analizar exhaustivamente las propiedades estadísticas, la distribución léxica, la representatividad muestral y las características estructurales del corpus de metadatos del conjunto de datos *Google - ASL Fingerspelling Recognition*, con el fin de diagnosticar sesgos de datos, caracterizar la complejidad del vocabulario y fundamentar con rigor analítico las decisiones de preprocesamiento, limpieza y modelado predictivo.**

### 3.2 Objetivos Específicos
1. **Caracterizar la variabilidad y distribución de la longitud de las secuencias textuales** (expresada en número de caracteres y palabras) mediante estadística descriptiva unidimensional (media, mediana, desviación estándar, rango intercuartílico, asimetría y curtosis), identificando la prevalencia de valores atípicos y segmentando los registros por tipología semántica (URLs, teléfonos, direcciones y texto general).
2. **Evaluar el balance, diversidad y representatividad de las observaciones registradas por participante (`participant_id`)**, cuantificando la concentración de muestras por señante y analizando la heterogeneidad en la longitud media de las frases para detectar potenciales sesgos de firmante en el dataset.
3. **Determinar el perfil de frecuencias de los 59 caracteres del alfabeto extendido de predicción y modelar las probabilidades de transición ortográfica (bigramas y trigramas)**, contrastando la distribución empírica con la frecuencia del inglés estándar para identificar clases minoritarias críticas y anticipar desafíos biomecánicos de co-articulación.

---

## 4. Referencias Académicas del Dominio
1. **Deaf Professional Arts Network (DPAN) & Google Research (2023).** *FSboard: An American Sign Language Fingerspelling Dataset.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
2. **Lugaresi, C., et al. (2019).** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv preprint arXiv:1906.08172.
3. **Padden, C. A., & Gunsauls, D. C. (2003).** *How the alphabet came to be used in a sign language.* Sign Language Studies, 4(1), 10-33.
4. **Battison, R. (1978).** *Lexical Borrowing in American Sign Language.* Linstok Press.
5. **Graves, A., et al. (2006).** *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks.* ICML.
