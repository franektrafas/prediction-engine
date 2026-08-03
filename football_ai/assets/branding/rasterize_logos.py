"""Rasterize the StatsDesk SVG brand assets to transparent PNGs, once.

Not a runtime step: the logo never changes, so re-rendering it on every PDF
export or page load would be pure waste. Run this after editing any SVG here
and commit the PNGs alongside them.

    python football_ai/assets/branding/rasterize_logos.py

Why a hand-rolled renderer instead of a library: every off-the-shelf SVG
rasteriser available here needs a native cairo build that this machine
doesn't have (`cairosvg`, and `reportlab`'s renderPM via `rlPyCairo`; no
prebuilt `pycairo` wheel exists for this platform either). Going
SVG -> PDF -> PNG through `sips` does work, but flattens onto an opaque white
background, which would put a white box behind a logo meant for a dark page.

So this reads the real SVG files and redraws them with Pillow onto an RGBA
canvas. It deliberately supports only the small subset those files use --
`<rect>` and `<text>`/`<tspan>` -- and raises on anything else rather than
silently dropping it, so a future edit to the artwork can't quietly render
wrong. Geometry always comes from the SVG; nothing about the design is
duplicated here.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BRANDING_DIRECTORY = Path(__file__).resolve().parent
SCALE = 3
_SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
# The artwork asks for Georgia; DejaVuSans (shipped with matplotlib, already a
# project dependency) is the portable stand-in when Georgia isn't installed.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/Library/Fonts/Georgia.ttf",
)


def _resolve_font(size: int) -> ImageFont.FreeTypeFont:
    """Load Georgia if the machine has it, else matplotlib's bundled DejaVuSans."""
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    import matplotlib

    fallback = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return ImageFont.truetype(str(fallback), size)


def _parse_colour(
    value: str | None, opacity: float
) -> tuple[int, int, int, int] | None:
    """Turn a `#rrggbb` fill plus an inherited opacity into an RGBA tuple."""
    if not value or value == "none":
        return None
    match = re.fullmatch(r"#([0-9a-fA-F]{6})", value.strip())
    if not match:
        raise ValueError(
            f"Unsupported fill {value!r} -- this renderer only takes #rrggbb"
        )
    digits = match.group(1)
    red, green, blue = (int(digits[index : index + 2], 16) for index in (0, 2, 4))
    return (red, green, blue, round(255 * opacity))


def _draw_text(
    draw: ImageDraw.ImageDraw,
    element: ElementTree.Element,
    scale: int,
    inherited_fill: str | None,
    opacity: float,
) -> None:
    """Draw a `<text>` run, honouring per-`<tspan>` fills and letter-spacing.

    Pillow has no letter-spacing, so runs are drawn character by character --
    which the watermark needs (it sets 1.5) and the lockup benefits from.
    """
    x = float(element.get("x", 0)) * scale
    y = float(element.get("y", 0)) * scale
    font_size = round(float(element.get("font-size", 16)) * scale)
    spacing = float(element.get("letter-spacing", 0)) * scale
    font = _resolve_font(font_size)

    runs: list[tuple[str, str | None]] = []
    if element.text and element.text.strip():
        runs.append((element.text.strip(), element.get("fill", inherited_fill)))
    for child in element:
        if child.tag != f"{_SVG_NAMESPACE}tspan":
            raise ValueError(f"Unsupported child of <text>: {child.tag}")
        if child.text:
            runs.append((child.text, child.get("fill", inherited_fill)))

    cursor = x
    for content, fill in runs:
        colour = _parse_colour(fill, opacity)
        for character in content:
            # SVG's y on <text> is the baseline, which is Pillow's "ls" anchor.
            draw.text((cursor, y), character, font=font, fill=colour, anchor="ls")
            cursor += draw.textlength(character, font=font) + spacing


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    element: ElementTree.Element,
    scale: int,
    inherited_fill: str | None,
    opacity: float,
) -> None:
    """Draw a `<rect>`, rounded when it carries an `rx`."""
    x = float(element.get("x", 0)) * scale
    y = float(element.get("y", 0)) * scale
    width = float(element.get("width", 0)) * scale
    height = float(element.get("height", 0)) * scale
    radius = float(element.get("rx", 0)) * scale
    colour = _parse_colour(element.get("fill", inherited_fill), opacity)
    box = (x, y, x + width, y + height)
    if radius > 0:
        draw.rounded_rectangle(box, radius=radius, fill=colour)
    else:
        draw.rectangle(box, fill=colour)


def _draw_element(
    draw: ImageDraw.ImageDraw,
    element: ElementTree.Element,
    scale: int,
    inherited_fill: str | None,
    opacity: float,
) -> None:
    """Dispatch one SVG element, recursing into groups."""
    tag = element.tag.replace(_SVG_NAMESPACE, "")
    local_opacity = opacity * float(element.get("opacity", 1))
    local_fill = element.get("fill", inherited_fill)
    if tag == "g":
        for child in element:
            _draw_element(draw, child, scale, local_fill, local_opacity)
    elif tag == "rect":
        _draw_rect(draw, element, scale, inherited_fill, local_opacity)
    elif tag == "text":
        _draw_text(draw, element, scale, inherited_fill, local_opacity)
    elif tag in {"title", "desc"}:
        return
    else:
        raise ValueError(f"Unsupported SVG element <{tag}> -- extend this renderer")


def rasterize(svg_path: Path, scale: int = SCALE) -> Path:
    """Render one SVG to a transparent PNG of the same stem, `scale` times its size."""
    root = ElementTree.parse(svg_path).getroot()
    view_box = root.get("viewBox")
    if not view_box:
        raise ValueError(f"{svg_path.name} has no viewBox")
    _, _, view_width, view_height = (float(part) for part in view_box.split())
    canvas = Image.new(
        "RGBA", (round(view_width * scale), round(view_height * scale)), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(canvas)
    root_opacity = float(root.get("opacity", 1))
    for element in root:
        _draw_element(draw, element, scale, root.get("fill"), root_opacity)
    output_path = svg_path.with_suffix(".png")
    canvas.save(output_path)
    return output_path


def main() -> None:
    """Rasterize every SVG in this directory."""
    for svg_path in sorted(BRANDING_DIRECTORY.glob("*.svg")):
        output_path = rasterize(svg_path)
        print(f"{svg_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
