from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path


WIDTH = 2100
HEIGHT = 1400


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height
        self.items: list[str] = []

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1, rx=0, opacity=1):
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke="#222", sw=2, dash=None, opacity=1):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{dash_attr}/>'
        )

    def text(self, x, y, value, size=28, weight=400, anchor="start", fill="#202124", italic=False):
        style = "italic" if italic else "normal"
        self.items.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}" font-style="{style}">{esc(value)}</text>'
        )

    def circle(self, x, y, r, fill, stroke="#ffffff", sw=3, opacity=1):
        self.items.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def polygon(self, points, fill, stroke="#ffffff", sw=3, opacity=1):
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.items.append(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def marker(self, x, y, shape, fill, size=12, opacity=1):
        if shape == "circle":
            self.circle(x, y, size, fill, opacity=opacity)
        elif shape == "square":
            self.rect(x - size, y - size, size * 2, size * 2, fill, "#ffffff", 3, opacity=opacity)
        else:
            self.polygon([(x, y - size - 2), (x - size - 2, y + size), (x + size + 2, y + size)], fill, opacity=opacity)

    def save(self, path: Path):
        header = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            '<style>text{font-family:Arial,Helvetica,sans-serif}</style>\n'
        )
        path.write_text(header + "\n".join(self.items) + "\n</svg>\n", encoding="utf-8")


def axis(svg: SVG, x0, y0, w, h, xmin, xmax, ymin, ymax, xticks, yticks=None, xfmt=lambda x: f"{x:.2f}"):
    for value in xticks:
        x = x0 + (value - xmin) / (xmax - xmin) * w
        svg.line(x, y0, x, y0 + h, "#dfe3e8", 2)
        svg.text(x, y0 + h + 42, xfmt(value), 23, anchor="middle", fill="#4b5563")
    if yticks:
        for value in yticks:
            y = y0 + h - (value - ymin) / (ymax - ymin) * h
            svg.line(x0, y, x0 + w, y, "#eef0f2", 2)
            svg.text(x0 - 14, y + 8, f"{value:.2f}", 22, anchor="end", fill="#4b5563")
    svg.line(x0, y0 + h, x0 + w, y0 + h, "#495057", 2)
    svg.line(x0, y0, x0, y0 + h, "#495057", 2)


def sx(value, x0, w, xmin, xmax):
    return x0 + (value - xmin) / (xmax - xmin) * w


def sy(value, y0, h, ymin, ymax):
    return y0 + h - (value - ymin) / (ymax - ymin) * h


def render_png(svg_path: Path, png_path: Path, width: int = WIDTH, height: int = HEIGHT):
    node = shutil.which("node")
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    if bundled.exists():
        node = str(bundled)
    if not node:
        raise RuntimeError("Node.js is required to rasterize the SVG to PNG.")
    renderer = Path(__file__).with_name("render_png.js")
    subprocess.run([node, str(renderer), str(svg_path), str(png_path), str(width), str(height)], check=True)


SOURCE_LABEL = {"iscx_url2016_binary": "ISCX-URL2016", "phiusiil": "PhiUSIIL"}
SOURCE_COLOR = {"iscx_url2016_binary": "#2B6CB0", "phiusiil": "#C2417B"}
MODEL_LABEL = {"lr": "LR", "rf": "RF", "xgb": "XGBoost"}
MODEL_SHAPE = {"lr": "circle", "rf": "square", "xgb": "triangle"}
