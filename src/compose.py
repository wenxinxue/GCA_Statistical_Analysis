from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS = (4536, 1890)


def _panel_label_font(size: int = 40) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("Arial Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _draw_panel_labels(canvas: Image.Image, labels: dict[str, tuple[int, int]]) -> None:
    draw = ImageDraw.Draw(canvas)
    font = _panel_label_font()
    for label, position in labels.items():
        draw.text(position, label, font=font, fill="black", anchor="la")


def _crop_fraction(image: Image.Image, left=0.0, top=0.0, right=0.0, bottom=0.0) -> Image.Image:
    width, height = image.size
    return image.crop((
        round(width * left),
        round(height * top),
        round(width * (1 - right)),
        round(height * (1 - bottom)),
    ))


def _paste_contain(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, width, height = box
    image = image.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = left + (width - image.width) // 2
    y = top + (height - image.height) // 2
    canvas.paste(image, (x, y))


def _save(canvas: Image.Image, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / f"{stem}.png", dpi=(300, 300), optimize=True)
    canvas.save(output / f"{stem}.pdf", resolution=300.0)


def compose_rct_outcomes(panel_dir: Path, output: Path) -> None:
    canvas = Image.new("RGB", CANVAS, "white")
    panels_a_f = Image.open(panel_dir / "rct_outcome_panels.png")
    panels_g_h = Image.open(panel_dir / "rct_effect_panels.png")
    panel_i = Image.open(panel_dir / "rct_pdqi_domain_panel.png")

    _paste_contain(canvas, panels_a_f, (685, 36, 1521, 1831))
    panel_g = _crop_fraction(panels_g_h, top=0.11429, right=0.25836)
    panel_h = _crop_fraction(panels_g_h, left=0.76664, top=0.07786)
    _paste_contain(canvas, panel_g, (2260, 23, 1598, 889))
    _paste_contain(canvas, panel_h, (2296, 968, 512, 899))
    _paste_contain(canvas, panel_i, (2808, 968, 1050, 899))
    _draw_panel_labels(canvas, {"g": (2220, 28), "h": (2220, 974), "i": (2768, 974)})
    _save(canvas, output, "rct_outcomes")


def compose_rct_patient_outcomes(panel_dir: Path, output: Path) -> None:
    canvas = Image.new("RGB", CANVAS, "white")
    panels_a_b = Image.open(panel_dir / "rct_attention_panels.png")
    panels_c_d = Image.open(panel_dir / "rct_editing_panels.png")
    _paste_contain(canvas, panels_a_b, (898, 37, 2150, 896))
    _paste_contain(canvas, panels_c_d, (898, 974, 2150, 792))
    _draw_panel_labels(canvas, {"c": (850, 974), "d": (1930, 974)})
    _save(canvas, output, "rct_patient_outcomes")


def compose_single_panel(
    panel_dir: Path,
    output: Path,
    source_stem: str,
    final_stem: str,
    box: tuple[int, int, int, int],
) -> None:
    canvas = Image.new("RGB", CANVAS, "white")
    panel = Image.open(panel_dir / f"{source_stem}.png")
    _paste_contain(canvas, panel, box)
    _save(canvas, output, final_stem)
