"""Render synthetic scenes with KNOWN content.

Having ground-truth content (shape types, colors, counts, which is largest) lets
us build *content-aware* injected instructions whose correct answer we already
know -- essential for teaching and scoring context-grounded compliance without a
human-labeled image set. Scenes are deliberately simple and synthetic (no real
people or scraped imagery), which keeps the dataset self-contained and ethical.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (220, 50, 50), "blue": (50, 90, 220), "green": (40, 170, 70),
    "yellow": (230, 200, 40), "purple": (150, 60, 200),
}


def render_scene(rng: np.random.Generator, width: int = 384, height: int = 288,
                 min_shapes: int = 3, max_shapes: int = 6):
    """Return (PIL.Image, metadata dict) for a random scene."""
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    n = int(rng.integers(min_shapes, max_shapes + 1))
    shapes = []
    for _ in range(n):
        kind = SHAPES[int(rng.integers(len(SHAPES)))]
        cname = list(COLORS)[int(rng.integers(len(COLORS)))]
        color = COLORS[cname]
        size = int(rng.integers(40, 90))
        cx = int(rng.integers(size, width - size))
        cy = int(rng.integers(size, height - size))
        box = [cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2]
        if kind == "circle":
            draw.ellipse(box, fill=color)
            area = np.pi * (size / 2) ** 2
        elif kind == "square":
            draw.rectangle(box, fill=color)
            area = size ** 2
        else:  # triangle
            draw.polygon([(cx, box[1]), (box[0], box[3]), (box[2], box[3])], fill=color)
            area = 0.5 * size * size
        shapes.append({"kind": kind, "color": cname, "size": size, "area": float(area)})

    largest = max(shapes, key=lambda s: s["area"])
    color_counts = {c: sum(s["color"] == c for s in shapes) for c in COLORS}
    kind_counts = {k: sum(s["kind"] == k for s in shapes) for k in SHAPES}
    meta = {
        "n_shapes": n,
        "shapes": shapes,
        "largest_kind": largest["kind"],
        "largest_color": largest["color"],
        "color_counts": color_counts,
        "kind_counts": kind_counts,
    }
    return img, meta
