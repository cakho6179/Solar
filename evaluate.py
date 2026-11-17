"""Evaluate the model on the held-out validation images (the 20% split that was never trained on).

    python evaluate.py --ann <MAGFiLO_1.0_Annotations_kaggle2026_train.json> --images <train_images_dir>

Reports pixel Dice, Panoptic Quality and fragmentation counts for one or several configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from data import load_gt_rles, load_instances_by_stem, split_stems
from pipeline import InferenceConfig, evaluate, load_trained_model, predict_probability, extract_instances, rles_from_masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=Path("models/best_model.pth"))
    ap.add_argument("--infer-size", type=int, nargs="+", default=[1024])
    ap.add_argument("--tta", type=int, nargs="+", default=[4])
    ap.add_argument("--threshold", type=float, nargs="+", default=[InferenceConfig.threshold])
    ap.add_argument("--min-area", type=int, nargs="+", default=[InferenceConfig.min_area])
    ap.add_argument("--opening", type=int, nargs="+", default=[0], help="0 = off, 1 = on (several values = ablation)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N validation images")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, _ = load_trained_model(args.weights, device)

    by_stem = load_instances_by_stem(args.ann, args.images)
    _, val_stems = split_stems(list(by_stem))
    if args.limit:
        val_stems = val_stems[: args.limit]
    gt_all = load_gt_rles(args.ann)
    gt = {s: gt_all[s] for s in val_stems}
    print(f"{len(val_stems)} validation images, {sum(len(v) for v in gt.values())} GT instances")

    # One forward pass per (image, infer_size, tta); every post-processing variant reuses it.
    # Instances are RLE-encoded immediately so that memory stays small.
    results = []
    for size in args.infer_size:
        for tta in args.tta:
            variants = [(thr, ma, op) for thr in args.threshold for ma in args.min_area for op in args.opening]
            preds = {v: {} for v in variants}
            for s in tqdm(val_stems, desc=f"size={size} tta={tta}", unit="img"):
                gray = cv2.imread(str(by_stem[s][0]["image_path"]), cv2.IMREAD_GRAYSCALE)
                prob = predict_probability(model, gray, device, size, tta)
                for thr, ma, op in variants:
                    preds[(thr, ma, op)][s] = rles_from_masks({s: extract_instances(prob, thr, ma, bool(op))})[s]
            for thr, ma, op in variants:
                cfg = InferenceConfig(size, tta, thr, ma, bool(op))
                res = evaluate(gt, preds[(thr, ma, op)], val_stems)
                res["config"] = vars(cfg)
                results.append(res)
                print(
                    f"size={size} tta={tta} thr={thr} min_area={ma} opening={op} | "
                    f"PQ={res['PQ']:.4f} SQ={res['SQ']:.4f} RQ={res['RQ']:.4f} Dice={res['mean_dice']:.4f} "
                    f"TP/FP/FN={res['TP']}/{res['FP']}/{res['FN']} 1toN={res['one_to_many']} Nto1={res['many_to_one']}",
                    flush=True,
                )
    if args.out_json:
        args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
