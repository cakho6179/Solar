"""Create the figures and uncertainty estimates used in the technical report.

    python make_report_figures.py --ann <train.json> --images <train_images> --out figures

Writes ``pipeline.pdf`` (schematic), ``examples.pdf`` (qualitative validation examples) and
``val_metrics.json`` (point estimates and 95 % bootstrap intervals over the validation images).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data import decode_coco_segm, load_gt_rles, load_instances_by_stem, split_stems  # noqa: E402
from figure_pipeline import build_assets, draw_pipeline  # noqa: E402
from pipeline import evaluate, extract_instances, load_trained_model, predict_probability, rles_from_masks  # noqa: E402


# --------------------------------------------------------------------------- bootstrap
def bootstrap(per_image: list[dict], n_boot: int = 2000, seed: int = 0) -> dict[str, dict[str, float]]:
    tp = np.array([p["TP"] for p in per_image], float)
    fp = np.array([p["FP"] for p in per_image], float)
    fn = np.array([p["FN"] for p in per_image], float)
    siou = np.array([p["SQ"] * p["TP"] for p in per_image], float)
    dice = np.array([p["mean_dice"] for p in per_image], float)

    def stats(idx):
        t, f, n, s = tp[idx].sum(), fp[idx].sum(), fn[idx].sum(), siou[idx].sum()
        den = t + 0.5 * f + 0.5 * n
        return {
            "PQ": s / den if den else 0.0,
            "SQ": s / t if t else 0.0,
            "RQ": t / den if den else 0.0,
            "Dice": dice[idx].mean(),
        }

    point = stats(np.arange(len(per_image)))
    rng = np.random.default_rng(seed)
    draws = [stats(rng.integers(0, len(per_image), len(per_image))) for _ in range(n_boot)]
    out = {}
    for k in point:
        v = np.array([d[k] for d in draws])
        out[k] = {"value": float(point[k]), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    return out


# --------------------------------------------------------------------------- examples
def contours(mask: np.ndarray, size: int) -> list[np.ndarray]:
    small = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(cs)


def draw_examples(path: Path, items: list[tuple[str, np.ndarray, list[np.ndarray], list[np.ndarray], float]], disp: int = 720) -> None:
    # one row: ground truth (green) and prediction (red) contours drawn on the same image
    fig, axes = plt.subplots(1, len(items), figsize=(7.2, 2.55))
    for j, (stem, gray, gt_masks, pred_masks, dice) in enumerate(items):
        img = cv2.cvtColor(cv2.resize(gray, (disp, disp), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
        for masks, color in ((gt_masks, (40, 220, 60)), (pred_masks, (255, 60, 60))):
            for m in masks:
                cv2.drawContours(img, contours(m, disp), -1, color, 2)
        axes[j].imshow(img)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        axes[j].set_title(f"{stem[:8]}.. (image Dice {dice:.2f})", fontsize=6.5)
    plt.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=Path("models/best_model.pth"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--min-area", type=int, nargs="+", default=[1000])
    ap.add_argument("--final", default="auto", help="'auto' (best validation PQ) or infer_size,threshold,min_area for the example figure")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, _ = load_trained_model(args.weights, device)
    by_stem = load_instances_by_stem(args.ann, args.images)
    _, val_stems = split_stems(list(by_stem))
    gt_all = load_gt_rles(args.ann)
    gt = {s: gt_all[s] for s in val_stems}

    results = []
    per_image_by_variant: dict[tuple, dict[str, dict]] = {}

    for size in args.sizes:
        variants = [(size, thr, ma) for thr in args.thresholds for ma in args.min_area]
        preds: dict[tuple, dict[str, list]] = {v: {} for v in variants}
        for s in tqdm(val_stems, desc=f"size={size}", unit="img"):
            gray = cv2.imread(str(by_stem[s][0]["image_path"]), cv2.IMREAD_GRAYSCALE)
            prob = predict_probability(model, gray, device, size, 4)
            for v in variants:
                preds[v][s] = rles_from_masks({s: extract_instances(prob, v[1], v[2], False)})[s]
        for v in variants:
            per_image = [evaluate(gt, preds[v], [s]) for s in val_stems]
            ci = bootstrap(per_image)
            tot = evaluate(gt, preds[v], val_stems)
            results.append({"infer_size": v[0], "threshold": v[1], "min_area": v[2], "TP": tot["TP"], "FP": tot["FP"], "FN": tot["FN"],
                            "one_to_many": tot["one_to_many"], "many_to_one": tot["many_to_one"], "ci95": ci})
            print(v, {k: f"{c['value']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" for k, c in ci.items()}, tot["TP"], tot["FP"], tot["FN"], flush=True)
            per_image_by_variant[v] = {s: p for s, p in zip(val_stems, per_image)}

    (args.out / "val_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.final == "auto":  # configuration with the best validation PQ
        best = max(results, key=lambda r: r["ci95"]["PQ"]["value"])
        final_key = (best["infer_size"], best["threshold"], best["min_area"])
    else:
        f_size, f_thr, f_min = args.final.split(",")
        final_key = (int(f_size), float(f_thr), int(f_min))
    print("example configuration:", final_key)
    per_image_final = per_image_by_variant.get(final_key, {})

    if per_image_final:
        order = sorted(val_stems, key=lambda s: per_image_final[s]["mean_dice"])
        picks = [order[int(0.9 * (len(order) - 1))], order[len(order) // 2], order[int(0.1 * (len(order) - 1))]]
        items = []
        for s in picks:
            gray = cv2.imread(str(by_stem[s][0]["image_path"]), cv2.IMREAD_GRAYSCALE)
            prob = predict_probability(model, gray, device, final_key[0], 4)
            pred = extract_instances(prob, final_key[1], final_key[2], False)
            gt_masks = [mask_utils.decode(r) for r in gt[s]]
            items.append((s, gray, gt_masks, pred, per_image_final[s]["mean_dice"]))
        draw_examples(args.out / "examples.pdf", items)
        print("examples:", picks)

    # pipeline diagram with real thumbnails: a validation disk (90th-percentile example) and three training crops
    demo_stem = picks[0] if per_image_final else val_stems[0]
    demo_gray = cv2.imread(str(by_stem[demo_stem][0]["image_path"]), cv2.IMREAD_GRAYSCALE)
    train_stems, _ = split_stems(list(by_stem))
    crop_items = []
    for s in (train_stems[60], train_stems[260], train_stems[460]):
        areas = [int(decode_coco_segm(it["segm"], *demo_gray.shape).sum()) for it in by_stem[s]]
        crop_items.append(by_stem[s][int(np.argmax(areas))])
    assets = build_assets(model, device, demo_gray, crop_items, final_key[1], final_key[2])
    draw_pipeline(args.out / "pipeline.pdf", assets, final_key[1], final_key[2])
    print("pipeline figure written; disk shown:", demo_stem)


if __name__ == "__main__":
    main()
