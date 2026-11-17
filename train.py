"""Train the ResNet-34 U-Net with the BCE + Dice + clDice objective on instance-centred 512x512 crops.

The released checkpoint ``models/best_model.pth`` was produced on a Kaggle GPU with exactly these
defaults (seed 2026, batch 16, 18 epochs, AdamW lr 3e-4 / wd 1e-4, cosine schedule, AMP):

    python train.py --ann <MAGFiLO_1.0_Annotations_kaggle2026_train.json> --images <train_images_dir> --out runs/exp

Training data is the official competition *train* split only. The model-selection metric ("val Dice")
is the pixel Dice over validation crops that are centred on annotated filaments; it is not a
full-disk metric. Use ``evaluate.py`` for full-disk Dice / Panoptic Quality.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CROP_SIZE, SolarFilamentCropDataset, load_instances_by_stem, split_stems
from losses import FilamentLoss
from model import FilamentResNet34UNet

SEED = 2026


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, type=Path, help="COCO annotation JSON (competition train split)")
    ap.add_argument("--images", required=True, type=Path, help="directory with the training images")
    ap.add_argument("--out", type=Path, default=Path("train_output"))
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="1 epoch on a handful of crops (smoke test)")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    by_stem = load_instances_by_stem(args.ann, args.images)
    train_stems, val_stems = split_stems(list(by_stem))
    train_items = [it for s in train_stems for it in by_stem[s]]
    val_items = [it for s in val_stems for it in by_stem[s]]
    if args.dry_run:
        train_items, val_items, args.epochs, args.batch_size, args.workers = train_items[:4], val_items[:2], 1, 2, 0
    print(f"{len(train_items)} train crops ({len(train_stems)} images) | {len(val_items)} val crops ({len(val_stems)} images) | device={device}")

    pin = device.type == "cuda"
    train_loader = DataLoader(SolarFilamentCropDataset(train_items, CROP_SIZE, True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(SolarFilamentCropDataset(val_items, CROP_SIZE, False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = FilamentResNet34UNet(pretrained=True).to(device)  # ImageNet-pretrained encoder
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FilamentLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(imgs), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs), masks)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= max(1, len(train_items))
        scheduler.step()

        model.eval()
        val_loss = inter = union = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(imgs)
                        loss = criterion(logits, masks)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                val_loss += loss.item() * len(imgs)
                pred = (torch.sigmoid(logits) >= 0.5).float()
                inter += torch.sum(pred * masks).item()
                union += torch.sum(pred).item() + torch.sum(masks).item()
        val_loss /= max(1, len(val_items))
        val_dice = (2.0 * inter + 1.0) / (union + 1.0)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{time.time() - t0:.1f}s] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice, "lr": optimizer.param_groups[0]["lr"]})
        if val_dice > best:
            best = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val_dice": best, "arch": "ResNet34_UNet_3Ch_clDice"}, args.out / "best_model.pth")

    (args.out / "training_report.json").write_text(json.dumps({"history": history, "best_val_dice": best}, indent=2), encoding="utf-8")
    print(f"Done. best val_dice={best:.4f} -> {args.out / 'best_model.pth'}")


if __name__ == "__main__":
    main()
