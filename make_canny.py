from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-image SmartControl quick comparison. "
            "Runs pretrain(c_fix) and ours(c_fix/c_ada) with the same seed."
        )
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quick_compare_outputs"),
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("models/cldm_v15.yaml"),
        help="Model config path.",
    )
    parser.add_argument(
        "--control-ckpt",
        type=Path,
        default=Path("models/control_v11p_sd15_canny.pth"),
        help="ControlNet checkpoint path.",
    )
    parser.add_argument(
        "--sd-ckpt",
        type=Path,
        default=Path("models/v1-5-pruned.ckpt"),
        help="Stable Diffusion checkpoint path.",
    )
    parser.add_argument(
        "--smart-ckpt",
        type=Path,
        required=True,
        help="Your trained SmartControl checkpoint path.",
    )

    parser.add_argument("--width", type=int, default=512, help="Resize width.")
    parser.add_argument("--height", type=int, default=512, help="Resize height.")
    parser.add_argument(
        "--low-threshold",
        type=int,
        default=100,
        help="Canny low threshold.",
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=200,
        help="Canny high threshold.",
    )

    parser.add_argument("--steps", type=int, default=30, help="DDIM steps.")
    parser.add_argument("--cfg", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    return parser.parse_args()


def read_and_resize_rgb(path: Path, width: int, height: int) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return img_rgb


def to_canny_rgb(img_rgb: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def rgb_to_cond_tensor(img_rgb: np.ndarray, device: str) -> torch.Tensor:
    arr = img_rgb.astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    return ten.to(device=device, dtype=torch.float32)


def load_into_model(
    model: torch.nn.Module,
    ckpt_path: Path,
    strict: bool = False,
) -> Tuple[int, int]:
    sd = load_state_dict(str(ckpt_path), location="cpu")
    ret = model.load_state_dict(sd, strict=strict)
    return len(ret.missing_keys), len(ret.unexpected_keys)


def build_model(args: argparse.Namespace, use_smart: bool) -> torch.nn.Module:
    model = create_model(str(args.config)).cpu()
    load_into_model(model, args.control_ckpt, strict=False)
    load_into_model(model, args.sd_ckpt, strict=False)
    if use_smart:
        missing, unexpected = load_into_model(model, args.smart_ckpt, strict=False)
        print(f"[smart-ckpt] missing_keys={missing}, unexpected_keys={unexpected}")
    model = model.to(args.device).eval()
    return model


@torch.no_grad()
def infer_one(
    model: torch.nn.Module,
    canny_rgb: np.ndarray,
    prompt: str,
    steps: int,
    cfg_scale: float,
    seed: int,
    mode: str,
    device: str,
) -> np.ndarray:
    cond = rgb_to_cond_tensor(canny_rgb, device=device)
    c = model.get_learned_conditioning([prompt])
    uc = model.get_unconditional_conditioning(1)
    cond_dict = {"c_concat": [cond], "c_crossattn": [c]}
    uc_dict = {"c_concat": [cond], "c_crossattn": [uc]}

    sampler = DDIMSampler(model)
    _, _, h, w = cond.shape
    shape = (model.channels, h // 8, w // 8)

    orig_apply = model.apply_model
    model.apply_model = lambda x, t, cc, *a, **k: orig_apply(
        x, t, cc, mode=mode, *a, **k
    )
    try:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        samples, _ = sampler.sample(
            S=steps,
            batch_size=1,
            shape=shape,
            conditioning=cond_dict,
            eta=0.0,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uc_dict,
            verbose=False,
        )
    finally:
        model.apply_model = orig_apply

    x = model.decode_first_stage(samples)
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)[0]
    x = x.permute(1, 2, 0).detach().cpu().numpy()
    return (x * 255).astype(np.uint8)


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def pixel_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def print_cpre_stats(model: torch.nn.Module) -> None:
    if not hasattr(model, "c_pre_list"):
        print("[c_pre] model has no c_pre_list")
        return
    print("[c_pre] parameter stats:")
    for name, p in model.c_pre_list.named_parameters():
        mean_abs = float(p.abs().mean())
        std = float(p.std(unbiased=False))
        print(f"  {name}: mean_abs={mean_abs:.8f}, std={std:.8f}")


def main() -> None:
    args = parse_args()
    if args.low_threshold < 0 or args.high_threshold < 0:
        raise ValueError("Canny thresholds must be >= 0.")
    if args.low_threshold > args.high_threshold:
        raise ValueError("low-threshold must be <= high-threshold.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    image = read_and_resize_rgb(args.image, width=args.width, height=args.height)
    canny = to_canny_rgb(
        image,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    model_pre = build_model(args, use_smart=False)
    pred_pre_fix = infer_one(
        model_pre,
        canny_rgb=canny,
        prompt=args.prompt,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        mode="c_fix",
        device=args.device,
    )
    del model_pre
    if args.device == "cuda":
        torch.cuda.empty_cache()

    model_ours = build_model(args, use_smart=True)
    print_cpre_stats(model_ours)
    pred_ours_fix = infer_one(
        model_ours,
        canny_rgb=canny,
        prompt=args.prompt,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        mode="c_fix",
        device=args.device,
    )
    pred_ours_ada = infer_one(
        model_ours,
        canny_rgb=canny,
        prompt=args.prompt,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        mode="c_ada",
        device=args.device,
    )

    mae_pre_vs_ours_fix = pixel_mae(pred_pre_fix, pred_ours_fix)
    mae_pre_vs_ours_ada = pixel_mae(pred_pre_fix, pred_ours_ada)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(out_dir / "input_rgb.png", image)
    save_rgb(out_dir / "input_canny.png", canny)
    save_rgb(out_dir / "pred_pretrain_fix.png", pred_pre_fix)
    save_rgb(out_dir / "pred_ours_fix.png", pred_ours_fix)
    save_rgb(out_dir / "pred_ours_ada.png", pred_ours_ada)

    panel4 = np.concatenate([image, canny, pred_pre_fix, pred_ours_ada], axis=1)
    panel5 = np.concatenate([image, canny, pred_pre_fix, pred_ours_fix, pred_ours_ada], axis=1)
    save_rgb(out_dir / "panel_4col.png", panel4)
    save_rgb(out_dir / "panel_5col.png", panel5)

    print("Saved outputs to:", out_dir)
    print(f"MAE(pre_fix vs ours_fix): {mae_pre_vs_ours_fix:.6f}")
    print(f"MAE(pre_fix vs ours_ada): {mae_pre_vs_ours_ada:.6f}")


if __name__ == "__main__":
    main()

