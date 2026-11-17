"""Generate a submission CSV from a directory of H-alpha test images.

    python predict.py --images <test_images_dir> --weights models/best_model.pth --output submission.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # library modules live in src/

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data import list_images  # noqa: E402
from pipeline import InferenceConfig, load_trained_model, print_checks, run_inference, validate_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="directory with the test images (jpeg/jpg/png)")
    ap.add_argument("--weights", type=Path, default=Path("models/best_model.pth"))
    ap.add_argument("--output", type=Path, default=Path("submission.csv"))
    ap.add_argument("--infer-size", type=int, default=InferenceConfig.infer_size, help="network input resolution")
    ap.add_argument("--tta", type=int, choices=(1, 2, 4), default=InferenceConfig.tta, help="number of flip views")
    ap.add_argument("--threshold", type=float, default=InferenceConfig.threshold)
    ap.add_argument("--min-area", type=int, default=InferenceConfig.min_area, help="minimum instance area in pixels (2048x2048 frame)")
    ap.add_argument("--opening", action="store_true", help="apply the area-guarded 3x3 cross opening to each instance")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N images (smoke test)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, meta = load_trained_model(args.weights, device)
    print(f"Loaded {args.weights} | epoch={meta.get('epoch')} val_dice(crop-level)={meta.get('best_val_dice', float('nan')):.4f} | device={device}")

    images = list_images(args.images)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    cfg = InferenceConfig(args.infer_size, args.tta, args.threshold, args.min_area, args.opening)
    t0 = time.time()
    df = run_inference(model, images, device, cfg, progress=lambda xs: tqdm(xs, unit="img"))
    dt = time.time() - t0
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} instances for {len(images)} images to {args.output} ({dt / len(images):.2f} s/image, config={cfg})")

    ok = print_checks(validate_submission(df, expected_stems=[p.stem for p in images]))
    if not ok:
        raise SystemExit("Submission validation failed")


if __name__ == "__main__":
    main()
