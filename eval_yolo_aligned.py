"""Evaluate YOLO depth model on NYU with multiple alignment methods.

Usage:
    cd ~/ultralytics_depth_anything
    /home/rick/miniconda/envs/pytorch/bin/python eval_yolo_aligned.py \
        --checkpoint /home/rick/ultralytics/runs/depth/runs/depth/exp_l3/weights/best.pt \
        --device 0
"""

import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.expanduser("~/ultralytics_depth_anything"))
from ultralytics.nn.tasks import DepthModel

EIGEN_CROP = (45, 471, 41, 601)


def load_nyu_eigen(mat_path="/data/depth_anything/nyu_depth_v2_labeled.mat"):
    with h5py.File(mat_path, "r") as f:
        images = np.transpose(np.array(f["images"]), (0, 3, 2, 1))  # (N, H, W, 3) RGB
        depths = np.transpose(np.array(f["depths"]), (0, 2, 1))      # (N, H, W)
    indices = np.load("/data/depth_anything/eigen_test_indices.npy")
    return images[indices], depths[indices]


def load_yolo_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Try to get the model config from checkpoint
    ema = ckpt.get("ema") or ckpt.get("model")
    if hasattr(ema, "eval"):
        # Full model object saved
        model = ema.float()
    else:
        raise RuntimeError("Cannot load model from checkpoint - expected full model object")
    model = model.to(device).eval()
    return model


def eigen_crop(pred, gt):
    t, b, l, r = EIGEN_CROP
    return pred[t:b, l:r], gt[t:b, l:r]


def align_log_ls(pred, gt, mask, robust=True, n_iter=5, trim_pct=0.012):
    """Log-space least-squares alignment (same as DA V2 eval)."""
    lp = np.log(np.clip(pred[mask], 1e-8, None)).flatten()
    lg = np.log(gt[mask]).flatten()
    if robust:
        for _ in range(n_iter):
            A = np.stack([lp, np.ones_like(lp)], axis=1)
            s, t = np.linalg.lstsq(A, lg, rcond=None)[0]
            residuals = np.abs(s * lp + t - lg)
            threshold = np.percentile(residuals, (1 - trim_pct) * 100)
            keep = residuals < threshold
            lp, lg = lp[keep], lg[keep]
    else:
        A = np.stack([lp, np.ones_like(lp)], axis=1)
        s, t = np.linalg.lstsq(A, lg, rcond=None)[0]
    return np.exp(s * np.log(np.clip(pred, 1e-8, None)) + t)


def align_ls(pred, gt, mask):
    """Linear least-squares alignment."""
    p = pred[mask].flatten()
    g = gt[mask].flatten()
    A = np.stack([p, np.ones_like(p)], axis=1)
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]
    return s * pred + t


def align_median(pred, gt, mask):
    return pred * (np.median(gt[mask]) / (np.median(pred[mask]) + 1e-8))


def compute_metrics(pred, gt, min_depth=1e-3, max_depth=10.0):
    mask = (gt > min_depth) & (gt < max_depth)
    p, g = pred[mask], gt[mask]
    if len(g) == 0:
        return None
    thresh = np.maximum(p / g, g / p)
    return {
        "delta1": float((thresh < 1.25).mean()),
        "delta2": float((thresh < 1.25**2).mean()),
        "delta3": float((thresh < 1.25**3).mean()),
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "silog": float(np.sqrt(np.mean((np.log(p) - np.log(g))**2) - np.mean(np.log(p) - np.log(g))**2) * 100),
    }


@torch.no_grad()
def predict_yolo(model, images, device, imgsz=480):
    preds = []
    for i, img_rgb in enumerate(images):
        h, w = img_rgb.shape[:2]
        # BGR conversion + normalize
        x = torch.from_numpy(img_rgb[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = F.interpolate(x, size=(imgsz, imgsz), mode="bilinear", align_corners=True).to(device)
        pred = model(x)
        if isinstance(pred, dict):
            pred = pred["depth"]
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)
        preds.append(pred.squeeze().cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(images)}")
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.checkpoint}")
    model = load_yolo_model(args.checkpoint, device)
    print("Model loaded")

    print("Loading NYU Eigen test split...")
    images, depths = load_nyu_eigen()
    print(f"Loaded {len(images)} images")

    print(f"Running inference (imgsz={args.imgsz})...")
    t0 = time.time()
    preds = predict_yolo(model, images, device, args.imgsz)
    print(f"Inference: {time.time()-t0:.1f}s")

    # Evaluate with different alignment methods
    methods = {
        "none": lambda p, g, m: p,
        "median": align_median,
        "linear_ls": align_ls,
        "log_ls": lambda p, g, m: align_log_ls(p, g, m, robust=False),
        "log_ls_robust": lambda p, g, m: align_log_ls(p, g, m, robust=True),
    }

    for method_name, align_fn in methods.items():
        all_metrics = []
        for i in range(len(images)):
            pc, gc = eigen_crop(preds[i], depths[i])
            mask = (gc > 1e-3) & (gc < 10.0) & (pc > 1e-6)
            if mask.sum() < 100:
                continue
            aligned = align_fn(pc, gc, mask)
            aligned = np.clip(aligned, 1e-3, 10.0)
            m = compute_metrics(aligned, gc)
            if m:
                all_metrics.append(m)

        agg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n{'='*60}")
        print(f"Alignment: {method_name} ({len(all_metrics)} images)")
        print(f"  delta1={agg['delta1']:.4f}  delta2={agg['delta2']:.4f}  delta3={agg['delta3']:.4f}")
        print(f"  abs_rel={agg['abs_rel']:.4f}  rmse={agg['rmse']:.4f}  silog={agg['silog']:.4f}")


if __name__ == "__main__":
    main()
