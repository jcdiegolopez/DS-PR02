"""Exporta la sección de aplicación a PDF desde su fuente Markdown y capturas."""

from html import escape
from pathlib import Path
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "aplicacion.pdf"


def inline(text):
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C9CBC6"))
    canvas.line(20 * mm, 18 * mm, A4[0] - 20 * mm, 18 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#55585C"))
    canvas.drawString(20 * mm, 13 * mm, "CC3084 Data Science | Aplicación de reconocimiento de ASL")
    canvas.drawRightString(A4[0] - 20 * mm, 13 * mm, str(doc.page))
    canvas.restoreState()


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyApp", fontName="Helvetica", fontSize=10.5,
                              leading=15.5, spaceAfter=9, alignment=TA_LEFT))
    styles.add(ParagraphStyle("CaptionApp", parent=styles["BodyApp"], fontSize=9,
                              leading=12, textColor=colors.HexColor("#55585C")))
    styles["Heading1"].fontName = "Helvetica-Bold"
    styles["Heading1"].fontSize = 23
    styles["Heading1"].leading = 28
    styles["Heading1"].spaceAfter = 18
    styles["Heading2"].fontName = "Helvetica-Bold"
    styles["Heading2"].fontSize = 14
    styles["Heading2"].leading = 19
    styles["Heading2"].spaceBefore = 15
    styles["Heading2"].spaceAfter = 9
    styles["Heading2"].keepWithNext = True
    story = []
    for block in (ROOT / "05_informe_aplicacion.md").read_text(encoding="utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            story.append(Paragraph(inline(block[2:]), styles["Heading1"]))
        elif block.startswith("## "):
            story.append(Paragraph(inline(block[3:]), styles["Heading2"]))
        elif block.startswith("!["):
            file = ROOT / re.search(r"\]\(([^)]+)\)", block).group(1)
            with PILImage.open(file) as source:
                width, height = source.size
            figure = Image(str(file), width=170 * mm, height=170 * mm * height / width)
            figure.keepWithNext = True
            story.extend([Spacer(1, 5 * mm), figure])
        elif block.startswith("- "):
            for line in block.splitlines():
                story.append(Paragraph(inline(line[2:]), styles["BodyApp"], bulletText="•"))
        else:
            style = styles["CaptionApp"] if block.startswith("**Figura") else styles["BodyApp"]
            story.append(Paragraph(inline(block.replace("\n", " ")), style))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=20 * mm,
                           rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=24 * mm,
                           title="Aplicación de reconocimiento de deletreo manual", author="")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
