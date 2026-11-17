"""Inference pipeline, RLE encoding, evaluation (Dice / Panoptic Quality) and submission validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from model import FilamentResNet34UNet, make_3channel_image

IMAGE_SIZE = 2048


# --------------------------------------------------------------------------- model I/O
def load_trained_model(weights_path: str | Path, device: torch.device) -> tuple[FilamentResNet34UNet, dict]:
    """Load a checkpoint (``weights_only=True``, ``strict=True``). Returns ``(model, metadata)``."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model = FilamentResNet34UNet(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "model"}
    return model, meta


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    gray: np.ndarray,
    device: torch.device,
    infer_size: int = 1024,
    tta: int = 4,
) -> np.ndarray:
    """Full-disk probability map at the native resolution of ``gray``.

    The image is resized to ``infer_size`` (area interpolation), pushed through the network with
    ``tta`` flip views (1 = none, 2 = +H-flip, 4 = +V-flip and HV-flip), averaged, and resized back.
    """
    h, w = gray.shape
    small = gray if infer_size == h == w else cv2.resize(gray, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(make_3channel_image(small).transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)

    flips: list[tuple[int, ...]] = [(), (3,), (2,), (2, 3)][: {1: 1, 2: 2, 4: 4}[tta]]
    acc = torch.zeros((1, 1, infer_size, infer_size), device=device)
    for dims in flips:
        xi = torch.flip(x, dims) if dims else x
        p = torch.sigmoid(model(xi))
        acc += torch.flip(p, dims) if dims else p
    prob = (acc / len(flips)).squeeze().cpu().numpy()
    if prob.shape != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob


def guarded_opening(mask: np.ndarray, area_guard: float = 0.85, min_area: int = 100) -> np.ndarray:
    """3x3 cross opening, rejected if it removes more than ``1 - area_guard`` of the mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    area = int(smoothed.sum())
    if area >= area_guard * int(mask.sum()) and area >= min_area:
        return smoothed
    return mask


def extract_instances(
    prob: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 60,
    opening: bool = False,
) -> list[np.ndarray]:
    """Threshold, split into 8-connected components, drop small ones, optionally smooth.

    Connected components are pixel-disjoint by construction, so no two instances overlap.
    """
    binary = (prob >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    instances = []
    height, width = binary.shape
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        # Work inside the (zero-padded) bounding box: much cheaper than comparing the full frame.
        pad = 2
        y0, y1, x0, x1 = max(0, y - pad), min(height, y + h + pad), max(0, x - pad), min(width, x + w + pad)
        sub = (labels[y0:y1, x0:x1] == lbl).astype(np.uint8)
        if opening:
            sub = guarded_opening(sub)
        m = np.zeros((height, width), dtype=np.uint8)
        m[y0:y1, x0:x1] = sub
        instances.append(m)
    return instances


def encode_rle(mask: np.ndarray) -> str:
    """Compressed COCO RLE counts string (what the submission CSV expects)."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def decode_rle(counts: str, size: int = IMAGE_SIZE) -> np.ndarray:
    return mask_utils.decode({"size": [size, size], "counts": counts.encode("ascii")})


@dataclass
class InferenceConfig:
    """Default = the configuration with the best validation PQ (see ``figures/val_metrics.json``)."""

    infer_size: int = 1024
    tta: int = 4
    threshold: float = 0.98
    min_area: int = 1000
    opening: bool = False


def predict_image(model: torch.nn.Module, gray: np.ndarray, device: torch.device, cfg: InferenceConfig) -> list[np.ndarray]:
    prob = predict_probability(model, gray, device, cfg.infer_size, cfg.tta)
    return extract_instances(prob, cfg.threshold, cfg.min_area, cfg.opening)


def run_inference(
    model: torch.nn.Module,
    image_paths: Iterable[Path],
    device: torch.device,
    cfg: InferenceConfig,
    progress: Any = None,
) -> pd.DataFrame:
    """Run inference over images and return the submission DataFrame (filament_id, segmentation_rle)."""
    rows: list[tuple[str, str]] = []
    paths = list(image_paths)
    iterator = progress(paths) if progress is not None else paths
    for path in iterator:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Cannot read image: {path}")
        for k, m in enumerate(predict_image(model, gray, device, cfg), start=1):
            rows.append((f"{path.stem}_{k}", encode_rle(m)))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


# --------------------------------------------------------------------------- evaluation
def rles_from_dataframe(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for fid, counts in zip(df["filament_id"], df["segmentation_rle"]):
        stem = fid.rsplit("_", 1)[0]
        out.setdefault(stem, []).append({"size": [IMAGE_SIZE, IMAGE_SIZE], "counts": counts.encode("ascii")})
    return out


def rles_from_masks(masks_by_stem: dict[str, list[np.ndarray]]) -> dict[str, list[dict[str, Any]]]:
    return {s: [mask_utils.encode(np.asfortranarray(m)) for m in ms] for s, ms in masks_by_stem.items()}


def evaluate(
    gt: dict[str, list[dict[str, Any]]],
    pred: dict[str, list[dict[str, Any]]],
    stems: Iterable[str] | None = None,
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Panoptic Quality (Kirillov et al., CVPR 2019) plus pixel Dice and fragmentation counts.

    * PQ = sum(IoU of matched pairs) / (TP + FP/2 + FN/2), IoU > ``iou_thresh``, one-to-one matching,
      pooled over all images (SQ = mean matched IoU, RQ = TP / (TP + FP/2 + FN/2)).
    * Dice = mean over images of the pixel Dice between the union of GT and the union of predictions.
    * ``one_to_many`` = GT instances where >= 2 predictions have >= 50% of their area inside that GT
      (fragmentation); ``many_to_one`` = predictions that cover >= 50% of >= 2 GT instances (merging).
      These two use our own definition and may differ from the organizers' one.
    This is our implementation of the published formulas; it may differ in details from the
    official scorer, so treat absolute values as approximate.
    """
    stems = sorted(stems if stems is not None else gt.keys())
    tp_ious: list[float] = []
    fp = fn = one_to_many = many_to_one = 0
    dices: list[float] = []

    for stem in stems:
        g, p = gt.get(stem, []), pred.get(stem, [])
        if not g and not p:
            continue
        if not g or not p:
            fp += len(p)
            fn += len(g)
            dices.append(0.0)
            continue

        ious = mask_utils.iou(g, p, [0] * len(p))  # (n_gt, n_pred)
        ag, ap = mask_utils.area(g).astype(float), mask_utils.area(p).astype(float)
        inter = ious * (ag[:, None] + ap[None, :]) / (1.0 + ious)

        pairs = np.argwhere(ious > iou_thresh)
        order = np.argsort(-ious[pairs[:, 0], pairs[:, 1]]) if len(pairs) else []
        used_g: set[int] = set()
        used_p: set[int] = set()
        for k in order:
            gi, pi = int(pairs[k, 0]), int(pairs[k, 1])
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            tp_ious.append(float(ious[gi, pi]))
        fp += len(p) - len(used_p)
        fn += len(g) - len(used_g)

        pred_mostly_in_gt = (inter / np.maximum(ap[None, :], 1)) >= 0.5  # (n_gt, n_pred)
        pred_covers_gt = (inter / np.maximum(ag[:, None], 1)) >= 0.5  # (n_gt, n_pred)
        one_to_many += int((pred_mostly_in_gt.sum(axis=1) >= 2).sum())
        many_to_one += int((pred_covers_gt.sum(axis=0) >= 2).sum())

        ug, up = mask_utils.merge(g), mask_utils.merge(p)
        i_area = float(mask_utils.area(mask_utils.merge([ug, up], intersect=True)))
        a, b = float(mask_utils.area(ug)), float(mask_utils.area(up))
        dices.append(2 * i_area / (a + b) if (a + b) > 0 else 1.0)

    tp = len(tp_ious)
    den = tp + 0.5 * fp + 0.5 * fn
    return {
        "PQ": sum(tp_ious) / den if den > 0 else 0.0,
        "SQ": float(np.mean(tp_ious)) if tp_ious else 0.0,
        "RQ": tp / den if den > 0 else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "one_to_many": one_to_many,
        "many_to_one": many_to_one,
        "n_images": len(stems),
    }


# --------------------------------------------------------------------------- validation
@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def validate_submission(df: pd.DataFrame, expected_stems: Iterable[str] | None = None) -> list[Check]:
    """Structural checks on a submission DataFrame. Returns one ``Check`` per rule (10 in total)."""
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), detail))

    add("columns are exactly filament_id, segmentation_rle", list(df.columns) == ["filament_id", "segmentation_rle"], str(list(df.columns)))
    add("non-empty", len(df) > 0, f"{len(df)} rows")
    if len(df) == 0 or not {"filament_id", "segmentation_rle"} <= set(df.columns):
        return checks

    rle = df["segmentation_rle"]
    add("no null / empty RLE", bool(rle.notna().all() and (rle.astype(str).str.len() > 0).all()))
    add("filament_id unique", df["filament_id"].is_unique, f"{df['filament_id'].duplicated().sum()} duplicates")
    add("filament_id format <stem>_<k>", bool(df["filament_id"].astype(str).str.contains(r"^.+_\d+$").all()))
    add("no quote characters in RLE", not rle.astype(str).str.contains("['\"]").any())

    stems = df["filament_id"].astype(str).str.rsplit("_", n=1).str[0]
    decode_fail = empty = overlap_images = 0
    for stem, group in df.groupby(stems, sort=False):
        occupied = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for counts in group["segmentation_rle"]:
            try:
                m = decode_rle(str(counts))
            except Exception:
                decode_fail += 1
                continue
            if m.sum() == 0:
                empty += 1
            occupied += m
        overlap_images += int((occupied > 1).any())
    add("every RLE decodes at 2048x2048", decode_fail == 0, f"{decode_fail} failures")
    add("no empty masks", empty == 0, f"{empty} empty")
    add("no pixel shared by two instances of one image", overlap_images == 0, f"{overlap_images} images with overlap")

    if expected_stems is not None:
        expected, got = set(expected_stems), set(stems)
        # Images without any predicted filament are legitimate (they simply have no rows); only ids that
        # do not belong to the test set are an error. The number of uncovered images is reported.
        add(
            "no unknown image ids (coverage reported)",
            got <= expected,
            f"{len(got)}/{len(expected)} test images have predictions, unknown ids={len(got - expected)}, "
            f"images without prediction={len(expected - got)}",
        )
    else:
        add("no unknown image ids (coverage reported)", True, "skipped (no expected stems given)")
    return checks


def print_checks(checks: list[Check]) -> bool:
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f"  ({c.detail})" if c.detail else ""))
    ok = all(c.passed for c in checks)
    print(f"=> {sum(c.passed for c in checks)}/{len(checks)} checks passed")
    return ok
