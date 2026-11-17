"""Publication-style pipeline diagram (vector PDF) with real thumbnails produced by the released model.

Panel (a): inference on a full-disk image. Panel (b): offline training on filament-centred crops.
Used by ``make_report_figures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from data import SolarFilamentCropDataset  # noqa: E402
from model import make_3channel_image  # noqa: E402
from pipeline import extract_instances, predict_probability  # noqa: E402

INK = "#2b2b2b"
ENC, DEC, HEAD = "#9ecae1", "#fdd0a2", "#a1d99b"
STAGE = "#eef2f7"
GREEN, RED = (40, 220, 60), (255, 60, 60)


def _u8(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def build_assets(model, device, gray: np.ndarray, train_items: list[dict[str, Any]], tau: float, min_area: int) -> dict[str, Any]:
    """Thumbnails for the diagram, all computed with the released code and weights."""
    x3 = make_3channel_image(cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA))
    channels = [
        _u8((x3[..., 0] * 255).astype(np.uint8), 256),
        _u8((x3[..., 1] * 255).astype(np.uint8), 256),
        _u8((np.clip(x3[..., 2] * 4.0, 0, 1) * 255).astype(np.uint8), 256),  # contrast-stretched for display only
    ]
    prob = predict_probability(model, gray, device, 1024, 4)
    prob_rgb = (plt.get_cmap("magma")(cv2.resize(prob, (256, 256), interpolation=cv2.INTER_AREA))[..., :3] * 255).astype(np.uint8)

    instances = extract_instances(prob, tau, min_area, False)
    overlay = cv2.cvtColor(_u8(gray, 256), cv2.COLOR_GRAY2RGB)
    palette = plt.get_cmap("tab10")
    for k, m in enumerate(instances):
        sel = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
        overlay[sel] = (0.35 * overlay[sel] + 0.65 * np.array(palette(k % 10)[:3]) * 255).astype(np.uint8)

    crops = []
    for item in train_items:
        img, msk = SolarFilamentCropDataset([item], 512, False)[0]
        g = _u8((img[0].numpy() * 255).astype(np.uint8), 160)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        mask = cv2.resize(msk[0].numpy().astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, GREEN, 1)
        crops.append(rgb)
    return {"raw": _u8(gray, 256), "channels": channels, "prob": prob_rgb, "instances": overlay, "crops": crops}


def draw_pipeline(path: Path, assets: dict[str, Any], tau: float, min_area: int) -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 3.43))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 144)
    ax.set_ylim(9.5, 78)
    ax.set_aspect("equal")
    ax.axis("off")

    def thumb(img, x, y, s, cmap=None, zorder=2):
        ax.imshow(img, extent=[x, x + s, y, y + s], cmap=cmap, zorder=zorder, interpolation="bilinear", vmin=0, vmax=255 if img.ndim == 2 else None)
        ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=INK, lw=0.6, zorder=zorder + 1))

    def label(x, y, text, size=5.2, ha="center", va="top", **kw):
        ax.text(x, y, text, fontsize=size, ha=ha, va=va, color=INK, linespacing=1.15, **kw)

    def arrow(x0, y0, x1, y1, dashed=False, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK,
                                     linestyle=(0, (3, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=5))

    def box(x, y, w, h, text, fc, size=5.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=INK, lw=0.7, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, fontsize=size, ha="center", va="center", color=INK, linespacing=1.2, zorder=3)

    # ------------------------------------------------------------------ panel (a): inference
    label(1.5, 77.2, "(a)  Inference on a full-disk image", size=7, ha="left", fontweight="bold")
    ya, s = 52.0, 13.0  # thumbnail baseline and size
    thumb(assets["raw"], 1.5, ya, s, cmap="gray")
    label(8.0, ya - 1.2, "H-$\\alpha$ image\n$2048\\times2048$")
    arrow(15.5, ya + s / 2, 20.5, ya + s / 2)
    label(18.0, ya + s / 2 + 1.4, "resize\n$1024^2$", size=4.8, va="bottom")

    # three-channel stack (back to front: Sobel, CLAHE, raw)
    cs, off, base = 9.5, 2.6, ya + 5.2
    for i in (2, 1, 0):  # draw back to front: Sobel, CLAHE, raw
        thumb(assets["channels"][i], 22.0 + i * off, base - i * off, cs, cmap="gray", zorder=2 + (3 - i) * 2)
    for i, name in enumerate(["raw", "CLAHE", "Sobel"]):  # staircase labels next to each visible layer edge
        label(22.0 + i * off + cs + 0.6, base - i * off + cs - 1.0, name, size=4.6, ha="left")
    label(29.4, ya - 1.2, "3-channel input\n$3\\times1024^2$")
    arrow(42.8, ya + s / 2, 46.4, ya + s / 2)

    # U-Net feature-map schematic (block height ~ spatial size, width ~ log channels)
    blocks = [("stem", 64, 12.0, ENC), ("enc1", 64, 10.0, ENC), ("enc2", 128, 8.0, ENC), ("enc3", 256, 6.2, ENC), ("enc4", 512, 4.6, ENC),
              ("dec4", 256, 6.2, DEC), ("dec3", 128, 8.0, DEC), ("dec2", 64, 10.0, DEC), ("dec1", 64, 12.0, DEC), ("dec0", 32, 14.0, DEC),
              ("head", 1, 14.0, HEAD)]
    yc, x0, pitch = ya + s / 2 + 0.2, 48.0, 4.0
    centres, tops = [], []
    for i, (name, ch, h, col) in enumerate(blocks):
        w = 0.6 + 0.28 * np.log2(ch)
        xc = x0 + i * pitch
        ax.add_patch(Rectangle((xc - w / 2, yc - h / 2), w, h, fc=col, ec=INK, lw=0.6, zorder=3))
        label(xc, ya - 1.0, str(ch), size=4.2)
        centres.append(xc)
        tops.append(yc + h / 2)
    for src, dst, level in [(0, 8, 71.0), (1, 7, 69.6), (2, 6, 68.2), (3, 5, 66.8)]:  # encoder stage -> decoder stage, as in model.py
        ax.plot([centres[src], centres[src], centres[dst]], [tops[src], level, level], color="#666666", lw=0.6, ls=(0, (2, 1.5)), zorder=1)
        arrow(centres[dst], level, centres[dst], tops[dst] + 0.3, dashed=True)
    label((centres[0] + centres[8]) / 2, 72.4, "skip connections (concatenation)", size=4.6, va="bottom", style="italic")
    yb = ya - 3.6
    ax.plot([centres[0] - 1.6, centres[4] + 1.6], [yb, yb], color=INK, lw=0.7)
    ax.plot([centres[5] - 1.6, centres[10] + 1.0], [yb, yb], color=INK, lw=0.7)
    label((centres[0] + centres[4]) / 2, yb - 0.8, "ResNet-34 encoder\n(ImageNet init.)", size=5)
    label((centres[5] + centres[10]) / 2, yb - 0.8, "decoder: $2\\times2$ transposed conv\n+ 2 conv-BN-ReLU per stage", size=5)
    arrow(centres[10] + 2.6, ya + s / 2, 98.0, ya + s / 2)

    thumb(assets["prob"], 98.5, ya, s)
    label(105.0, ya - 1.2, "probability map $p$\n(mean of 4 flips)")
    arrow(112.0, ya + s / 2, 116.5, ya + s / 2)
    label(114.3, ya + s / 2 + 1.4, f"$p\\geq{tau:g}$", size=4.8, va="bottom")
    thumb(assets["instances"], 117.0, ya, s)
    label(123.5, ya - 1.2, f"8-connected comp.,\n$A\\geq{min_area}$ px")
    arrow(130.5, ya + s / 2, 133.5, ya + s / 2)
    box(133.8, ya + 0.3, 9.4, s - 0.6, "submission\n.csv\n<stem>_k,\nRLE", "#ffffff", size=4.6)

    # ------------------------------------------------------------------ panel (b): training
    label(1.5, 35.0, "(b)  Offline training on filament-centred crops", size=7, ha="left", fontweight="bold")
    yb0, cs2 = 21.0, 9.0
    for i, crop in enumerate(assets["crops"]):
        thumb(crop, 1.5 + i * 10.0, yb0, cs2)
    label(1.5 + 14.5, yb0 - 1.0, "$512^2$ crops around each annotated\nfilament (7,522; green = ground truth)")
    arrow(31.0, yb0 + cs2 / 2, 37.0, yb0 + cs2 / 2)
    label(34.0, yb0 + cs2 / 2 + 1.4, "flip,\nrot $90^\\circ$", size=4.8, va="bottom")
    box(37.5, yb0 - 1.0, 24.0, cs2 + 2.0, "ResNet-34 U-Net\n24.45 M parameters\n(same network as in (a))", STAGE)
    arrow(62.2, yb0 + cs2 / 2, 67.0, yb0 + cs2 / 2)
    label(64.6, yb0 + cs2 / 2 + 1.4, "logits", size=4.8, va="bottom")
    mid = yb0 + cs2 / 2
    for txt, yy in [("$0.40\\,\\mathcal{L}_{\\mathrm{BCE}}$", mid + 2.9), ("$0.35\\,\\mathcal{L}_{\\mathrm{Dice}}$", mid - 1.7), ("$0.25\\,\\mathcal{L}_{\\mathrm{clDice}}$", mid - 6.3)]:
        box(67.5, yy, 22.0, 3.4, txt, "#ffffff", size=5.2)
        arrow(90.0, yy + 1.7, 96.2, mid)
    ax.add_patch(Circle((98.0, yb0 + cs2 / 2), 1.7, fc="#ffffff", ec=INK, lw=0.7, zorder=3))
    ax.text(98.0, yb0 + cs2 / 2, "$\\Sigma$", fontsize=6, ha="center", va="center", color=INK, zorder=4)
    arrow(99.9, yb0 + cs2 / 2, 104.0, yb0 + cs2 / 2)
    box(104.5, yb0 - 0.6, 17.0, cs2 + 1.2, "AdamW, cosine\n18 epochs, AMP\nbatch 16", STAGE)
    arrow(122.2, yb0 + cs2 / 2, 126.0, yb0 + cs2 / 2)
    box(126.5, yb0 - 0.6, 16.7, cs2 + 1.2, "best checkpoint\n(crop-level val\nDice 0.7422)", HEAD)
    label(87.5, 13.4, "train/val split by image: 566 / 141 images (competition train split only)", size=4.8, ha="center", style="italic")

    # trained weights feed the network of panel (a), entering between encoder and decoder
    xw, xt = 134.8, (centres[4] + centres[5]) / 2
    ax.plot([xw, xw], [yb0 + cs2 + 0.9, 39.0], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.plot([xw, xt], [39.0, 39.0], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    arrow(xt, 39.0, xt, 42.6, dashed=True)
    label(100.0, 39.6, "trained weights (loaded without re-training)", size=4.8, va="bottom", style="italic")

    fig.savefig(path, dpi=300)
    plt.close(fig)
