"""Línea de subtítulo: frase real contra la leída por el modelo, carácter por carácter."""

from __future__ import annotations

from html import escape

from src.evaluate import edit_operations

_KIND = {"correct": "ok", "substitution": "sub", "deletion": "del", "insertion": "ins"}


def _cell(char: str, in_error: bool) -> str:
    if char == " ":
        return "␣" if in_error else "&nbsp;"
    return escape(char) if char else "&nbsp;"


def diff_caption(reference: str, prediction: str, tag_left: str, tag_right: str, grid: bool = False) -> str:
    """Rejilla de dos filas (REAL / MODELO) alineada por la distancia de edición."""
    ops = edit_operations(reference, prediction)
    words, current = [], []
    for kind, ref_char, hyp_char in ops:
        error = kind != "correct"
        current.append(
            f'<div class="cap-col {_KIND[kind]}"><span class="ref">{_cell(ref_char, error)}</span>'
            f'<span class="hyp">{_cell(hyp_char, error) if kind != "deletion" else ""}</span></div>'
        )
        if ref_char == " " or hyp_char == " ":
            words.append(current)
            current = []
    if current:
        words.append(current)
    body = "".join(f'<div style="display:flex">{"".join(w)}</div>' for w in words)
    return (
        f'<div class="cap{" grid" if grid else ""}" role="figure" aria-label="Frase real: {escape(reference)}. '
        f'Predicción: {escape(prediction)}">'
        f'<div class="cap-tag"><span>{tag_left}</span><span>{tag_right}</span></div>'
        f'<div style="display:flex"><div class="cap-rowlabels"><span>REAL</span><span>LEÍDO</span></div>'
        f'<div class="cap-grid">{body}</div></div></div>'
    )


def legend() -> str:
    return (
        '<div class="cap-legend">'
        '<span><i style="background:#D55E00"></i>sustitución</span>'
        '<span><i style="background:#0B0B0C;box-shadow:inset 0 -3px 0 #D55E00"></i>carácter omitido</span>'
        '<span><i style="background:#0B0B0C;outline:1px dashed #0B0B0C;outline-offset:2px"></i>carácter insertado</span>'
        '<span>␣ espacio con error</span></div>'
    )


def edit_distance(reference: str, prediction: str) -> int:
    return sum(kind != "correct" for kind, _, _ in edit_operations(reference, prediction))
