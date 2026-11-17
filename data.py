"""Data access: MAGFiLO COCO annotations, train/validation split and the crop dataset."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from model import make_3channel_image

H = W = 2048
CROP_SIZE = 512
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")


def decode_coco_segm(segm: Any, h: int = H, w: int = W) -> np.ndarray:
    """Decode a COCO polygon list or RLE dict into a (h, w) uint8 mask (training-time decoder)."""
    if isinstance(segm, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in segm:
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
        return mask
    if isinstance(segm, dict):
        return mask_utils.decode(segm).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def load_instances_by_stem(ann_path: str | Path, images_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Return {image stem: [{"image_path", "segm"}, ...]} for annotated images that exist on disk."""
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_name = {im["id"]: im["file_name"] for im in coco["images"]}
    images_dir = Path(images_dir)
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        fname = id_to_name.get(ann["image_id"], "")
        path = images_dir / fname
        if path.is_file():
            by_stem[Path(fname).stem].append({"image_path": path, "segm": ann["segmentation"]})
    return by_stem


def split_stems(stems: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic grouped split used for the released checkpoint.

    Stems (image timestamp + station code) are sorted and the last ``len // 5`` form the
    validation set, so all annotations of one image stay on the same side of the split.
    Returns ``(train_stems, val_stems)``.
    """
    stems = sorted(stems)
    n_val = max(1, len(stems) // 5)
    return stems[:-n_val], stems[-n_val:]


def load_gt_rles(ann_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Ground-truth instances as compressed COCO RLEs, keyed by image stem (for evaluation)."""
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        stem = id_to_stem.get(ann["image_id"])
        if stem is None:
            continue
        seg = ann["segmentation"]
        if isinstance(seg, list):
            polys = [p for p in seg if len(p) >= 6]
            if not polys:
                continue
            rle = mask_utils.merge(mask_utils.frPyObjects(polys, H, W))
        elif isinstance(seg, dict):
            rle = mask_utils.frPyObjects(seg, H, W) if isinstance(seg.get("counts"), list) else seg
        else:
            continue
        gt[stem].append(rle)
    return dict(gt)


class SolarFilamentCropDataset(Dataset):
    """512x512 crops centred on single annotated filament instances (jittered at train time)."""

    def __init__(self, items: list[dict[str, Any]], crop_size: int = CROP_SIZE, is_train: bool = True) -> None:
        self.items = items
        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        img_gray = cv2.imread(str(item["image_path"]), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            img_gray = np.zeros((H, W), dtype=np.uint8)
        mask = decode_coco_segm(item["segm"], H, W)

        ys, xs = np.nonzero(mask)
        if len(ys) > 0:
            ymin, ymax, xmin, xmax = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
            bw, bh = max(64, xmax - xmin + 1), max(64, ymax - ymin + 1)
            side = max(int(max(bw, bh) * 1.4), self.crop_size)
            if self.is_train:
                cx += random.randint(-bw // 6, bw // 6)
                cy += random.randint(-bh // 6, bh // 6)
            x0 = max(0, min(cx - side // 2, W - side))
            y0 = max(0, min(cy - side // 2, H - side))
            side = min(side, W - x0, H - y0)
        else:
            x0, y0, side = 0, 0, self.crop_size

        crop_gray = img_gray[y0 : y0 + side, x0 : x0 + side]
        crop_mask = mask[y0 : y0 + side, x0 : x0 + side]
        if crop_gray.shape[0] != self.crop_size or crop_gray.shape[1] != self.crop_size:
            crop_gray = cv2.resize(crop_gray, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            crop_mask = cv2.resize(crop_mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

        img_3ch = make_3channel_image(crop_gray)

        if self.is_train:  # flips and 90-degree rotations only
            if random.random() > 0.5:
                img_3ch = np.ascontiguousarray(img_3ch[:, ::-1, :])
                crop_mask = np.ascontiguousarray(crop_mask[:, ::-1])
            if random.random() > 0.5:
                img_3ch = np.ascontiguousarray(img_3ch[::-1, :, :])
                crop_mask = np.ascontiguousarray(crop_mask[::-1, :])
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                img_3ch = np.ascontiguousarray(np.rot90(img_3ch, k, (0, 1)))
                crop_mask = np.ascontiguousarray(np.rot90(crop_mask, k, (0, 1)))

        t_img = torch.from_numpy(img_3ch.transpose(2, 0, 1).copy()).float()
        t_mask = torch.from_numpy(crop_mask.copy()).float().unsqueeze(0)
        return t_img, t_mask
