#!/usr/bin/env python3
"""Generate transparent, animated colour dot-matrix portraits for the profile README."""
from __future__ import annotations

from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def square_crop(img: Image.Image, fx: float = 0.5, fy: float = 0.43) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = min(max(fx * w - side / 2, 0), w - side)
    top = min(max(fy * h - side / 2, 0), h - side)
    return img.crop((round(left), round(top), round(left) + side, round(top) + side))


def isolate_subject(img: Image.Image) -> Image.Image:
    """Use GrabCut + morphology to isolate the central person from the photo background."""
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    rect = (
        max(2, int(w * 0.035)),
        max(2, int(h * 0.02)),
        max(2, int(w * 0.93)),
        max(2, int(h * 0.965)),
    )
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 7, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if count > 1:
        cx0, cy0 = w / 2, h * 0.48
        best_label, best_score = 1, -1.0
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            cx, cy = centroids[label]
            dist = ((cx - cx0) / w) ** 2 + ((cy - cy0) / h) ** 2
            score = area * (1.0 / (1.0 + 4.0 * dist))
            if score > best_score:
                best_label, best_score = label, score
        fg = np.where(labels == best_label, 255, 0).astype("uint8")

    kernel = np.ones((7, 7), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    fg = cv2.GaussianBlur(
        fg,
        (0, 0),
        sigmaX=max(1.5, w / 360),
        sigmaY=max(1.5, h / 360),
    )

    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray(fg, mode="L"))
    return rgba


def preserve_theme_colour(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Use identical portrait colours in both GitHub themes.

    A small shadow lift is retained from the approved dark-mode portrait so facial,
    hair and clothing detail survive the dot conversion. Light-mode contrast is
    handled by a silhouette underlay, not by recolouring the photograph.
    """
    lift = 0.10
    return (
        int(r + (255 - r) * lift),
        int(g + (255 - g) * lift),
        int(b + (255 - b) * lift),
    )


def generate_svg(src: Path, out: Path, cols: int = 100, theme: str = "dark") -> None:
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img = square_crop(img)
    img = isolate_subject(img)

    # The source photo, crop, subject mask and dot geometry are identical for both themes.
    rgb_img = img.convert("RGB")
    rgb_img = ImageEnhance.Contrast(rgb_img).enhance(1.14)
    rgb_img = ImageEnhance.Color(rgb_img).enhance(1.04)
    alpha = img.getchannel("A")

    gray = rgb_img.convert("L")
    hard_mask = alpha.point(lambda v: 255 if v > 35 else 0)
    gray = ImageOps.equalize(gray, mask=hard_mask)
    gray = gray.filter(
        ImageFilter.UnsharpMask(
            radius=max(2, img.size[0] // 80),
            percent=60,
            threshold=0,
        )
    )

    rows = cols
    small_rgb = rgb_img.resize((cols, rows), Image.Resampling.LANCZOS)
    small_gray = gray.resize((cols, rows), Image.Resampling.LANCZOS)
    small_alpha = alpha.resize((cols, rows), Image.Resampling.LANCZOS)

    cell = 10.0
    pad = 8.0
    max_r = cell * 0.46
    width = cols * cell + pad * 2
    height = rows * cell + pad * 2

    css = [
        "@keyframes rv{from{opacity:0}to{opacity:1}}",
        ".rw{animation:rv .45s ease-out both}",
    ]
    step = 2.5 / max(rows - 1, 1)
    css.extend(f".r{y}{{animation-delay:{y * step:.3f}s}}" for y in range(rows))

    rp = small_rgb.load()
    gp = small_gray.load()
    ap = small_alpha.load()
    groups: list[str] = []

    for y in range(rows):
        underlay: list[str] = []
        circles: list[str] = []
        for x in range(cols):
            a = ap[x, y] / 255.0
            if a < 0.10:
                continue

            lum = gp[x, y] / 255.0
            radius = max_r * (0.18 + 0.82 * (lum ** 0.78)) * min(1.0, a * 1.35)
            if radius < 0.35:
                continue

            cx = pad + x * cell + cell / 2
            cy = pad + y * cell + cell / 2

            # On GitHub light mode the transparent gaps between portrait dots inherit
            # a white background and wash out light clothing/skin. A near-contiguous
            # dark underlay is drawn only inside the isolated subject silhouette.
            # It animates row-for-row with the portrait, so the reveal is unchanged.
            if theme == "light":
                under_r = max_r * 0.97 * min(1.0, a * 1.25)
                if under_r >= 0.35:
                    underlay.append(
                        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{under_r:.2f}" '
                        f'fill="#0d1117" opacity="0.92"/>'
                    )

            r, g, b = rp[x, y]
            r, g, b = preserve_theme_colour(r, g, b)
            fill = f"#{r:02x}{g:02x}{b:02x}"
            circles.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" fill="{fill}"/>'
            )

        if circles:
            # Underlay comes first, then the unchanged coloured portrait dots.
            groups.append(
                f'<g class="rw r{y}">' + "".join(underlay) + "".join(circles) + "</g>"
            )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" '
        f'aria-label="Ansh Dutta dot-matrix portrait">'
        f'<style>{"".join(css)}</style>'
        + "".join(groups)
        + "</svg>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    args = parser.parse_args()
    generate_svg(args.source, args.output, args.cols, args.theme)
