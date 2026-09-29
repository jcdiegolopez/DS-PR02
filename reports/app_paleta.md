# Aplicación: paleta de colores y su justificación

La app muestra la predicción de cada modelo como una **línea de subtítulo** (*closed caption*): caja negra, texto blanco y monoespaciado, con la frase real arriba y la leída por el modelo abajo. La idea viene del propio problema: el reconocimiento de *fingerspelling* existe para que una persona sorda pueda comunicarse con quien no sabe señas, y el subtítulo es la forma más conocida de accesibilidad en pantalla para la comunidad sorda.

## Estrategia de color

La paleta es **restringida**: neutros para todo lo que es interfaz y color solo donde transmite información. Hay tres familias de color con funciones distintas que no se mezclan:

| Familia | Tipo de paleta | Uso |
|---|---|---|
| Neutros y subtítulo | Acromática | Fondo, texto, estructura y cajas de subtítulo |
| Modelos (M1, M2, M3) | Categórica | Identifica un modelo en todas las páginas y gráficas |
| Tipos de carácter | Categórica | Letras, dígitos, símbolos y espacios en el análisis del corpus |

A eso se suman dos colores **reservados** con un solo significado cada uno: el amarillo marca lo que está activo y el bermellón marca un error.

### Neutros

| Rol | Color | Por qué |
|---|---|---|
| Fondo | `#EEEFEC` | Gris claro casi acromático. La app se proyecta en un salón con luz encendida; un fondo claro mantiene el contraste en un proyector, donde los fondos oscuros se ven lavados. No es blanco puro para reducir el deslumbramiento. |
| Texto | `#111213` | Casi negro. Contraste 16.3:1 sobre el fondo (WCAG AAA). |
| Texto secundario | `#55585C` | Contraste 6.2:1 (AA para texto normal). |
| Caja de subtítulo | `#0B0B0C` con texto `#F5F5F0` | Es la convención visual de los subtítulos. Contraste 18:1. |

La barra lateral usa el mismo negro del subtítulo para que la navegación y las transcripciones pertenezcan al mismo sistema.

### Color categórico por modelo: Okabe-Ito

| Modelo | Color |
|---|---|
| M1 BiGRU | `#E69F00` naranja |
| M2 TCN | `#56B4E9` azul cielo |
| M3 Transformer | `#009E73` verde azulado |

La paleta de Okabe e Ito (2008) se diseñó para distinguirse con los tres tipos principales de daltonismo (protanopía, deuteranopía y tritanopía). Una paleta categórica tiene que diferenciar grupos sin sugerir un orden, y por eso se eligieron tonos distintos con luminosidad parecida en lugar de una escala de un solo color. Un modelo conserva su color en todas las páginas: si M1 es naranja en la tabla, también lo es en las curvas, en las barras y en la animación.

Como estos tonos tienen poco contraste contra el fondo claro (2.1:1 a 3.2:1), nunca se usan como color de texto. Las barras llevan un contorno oscuro y el nombre del modelo siempre aparece escrito junto al color, así que la información no depende solo del tono (WCAG 1.4.1).

### Tipos de carácter

| Tipo | Color |
|---|---|
| Letras | `#0072B2` azul |
| Dígitos | `#CC79A7` rosa violáceo |
| Símbolos | `#8C6D1F` ocre oscuro |
| Espacios | `#8A8D91` gris |

Los tres primeros también salen de Okabe-Ito, pero son tonos distintos a los de los modelos para que las dos familias no se confundan. El espacio va en gris porque no es un signo visible.

### Colores reservados

- **Amarillo caption `#F0E442`**: marca únicamente lo activo, como el firmante resaltado o el control deslizante. Sobre negro tiene un contraste de 14.9:1. No se usa para decorar.
- **Bermellón `#D55E00`**: marca únicamente los errores de transcripción (sustituciones y caracteres omitidos). El texto blanco sobre bermellón tiene 3.9:1, que cumple AA para texto grande; las letras del subtítulo miden 24 px. Además del color, los errores llevan subrayado o celda propia, para que se distingan también en escala de grises.

## Tipografía

**Atkinson Hyperlegible Next** para la interfaz y **Atkinson Hyperlegible Mono** para los subtítulos y los números. El Braille Institute diseñó esta familia para lectores con baja visión: formas de letra muy diferenciadas (`l`, `1` e `I` no se confunden; el `0` lleva una marca que lo separa de la `O`). Eso es justo lo que exige una transcripción letra por letra. Las fuentes van incluidas en `app/static/fonts/`, así que la demo funciona sin internet.

## Referencias

- Okabe, M., & Ito, K. (2008). *Color Universal Design (CUD): How to make figures and presentations that are friendly to colorblind people*. J*FLY. https://jfly.uni-koeln.de/color/
- World Wide Web Consortium. (2023). *Web Content Accessibility Guidelines (WCAG) 2.2*. https://www.w3.org/TR/WCAG22/
- Braille Institute of America. (2025). *Atkinson Hyperlegible Next*. https://www.brailleinstitute.org/freefont/
