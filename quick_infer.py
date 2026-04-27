from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler


DEFAULT_NEGATIVE_PROMPT = (
    'lowres, blurry, bad anatomy, bad hands, cropped, worst quality, low quality'
)
DEFAULT_COMPARE_SMART_CKPT = Path('lightning_logs/version_7/checkpoints/csp-epochepoch=199.ckpt')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Single-image SmartControl inference entrypoint. '
            'Supports either single-model sampling or pretrain-vs-smart comparison '
            'on the same canny condition.'
        )
    )
    parser.add_argument('--image', type=Path, required=True, help='Input image path.')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('quick_infer_outputs'),
        help='Directory to save outputs.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('models/cldm_v15.yaml'),
        help='Model config path.',
    )
    parser.add_argument(
        '--control-ckpt',
        type=Path,
        default=Path('models/control_v11p_sd15_canny.pth'),
        help='ControlNet checkpoint path.',
    )
    parser.add_argument(
        '--sd-ckpt',
        type=Path,
        default=Path('models/v1-5-pruned.ckpt'),
        help='Stable Diffusion checkpoint path.',
    )
    parser.add_argument(
        '--smart-ckpt',
        type=Path,
        default=Path('models/canny.ckpt'),
        help='SmartControl checkpoint path. Used by single-model c_ada inference.',
    )
    parser.add_argument(
        '--compare-pretrain',
        action='store_true',
        help='Compare pretrain(c_fix) against SmartControl(c_ada) and save one output for each.',
    )
    parser.add_argument(
        '--compare-smart-ckpt',
        type=Path,
        default=DEFAULT_COMPARE_SMART_CKPT,
        help='SmartControl checkpoint path used when --compare-pretrain is enabled.',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='c_ada',
        choices=['c_fix', 'c_ada'],
        help='Inference mode for single-model path. c_ada uses SmartControl local adaptive control.',
    )
    parser.add_argument(
        '--crop',
        type=str,
        default='full',
        choices=['full', 'left', 'right'],
        help='Crop strategy before resizing. Default is full-image inference; left/right is only for compatibility with panel inputs.',
    )
    parser.add_argument('--width', type=int, default=512, help='Resize width.')
    parser.add_argument('--height', type=int, default=512, help='Resize height.')
    parser.add_argument(
        '--low-threshold',
        type=int,
        default=100,
        help='Canny low threshold.',
    )
    parser.add_argument(
        '--high-threshold',
        type=int,
        default=200,
        help='Canny high threshold.',
    )
    parser.add_argument('--steps', type=int, default=60, help='DDIM steps.')
    parser.add_argument('--cfg', type=float, default=8.5, help='Classifier-free guidance scale.')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta.')
    parser.add_argument(
        '--control-scale',
        type=float,
        default=1.0,
        help='Global control strength multiplier before c_fix/c_ada fusion.',
    )
    parser.add_argument('--seed', type=int, default=12345, help='Base random seed.')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to draw using sequential seeds. Compare mode requires 1.',
    )
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='Negative prompt. Use empty string to disable.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Inference device.',
    )
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def crop_rgb(img_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'full':
        return img_rgb
    width = img_rgb.shape[1]
    mid = width // 2
    if mode == 'left':
        return img_rgb[:, :mid, :]
    if mode == 'right':
        return img_rgb[:, mid:, :]
    raise ValueError(f'Unsupported crop mode: {mode}')


def resize_rgb(img_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if img_rgb.shape[1] == width and img_rgb.shape[0] == height:
        return img_rgb
    interpolation = cv2.INTER_AREA
    if img_rgb.shape[1] < width or img_rgb.shape[0] < height:
        interpolation = cv2.INTER_LANCZOS4
    return cv2.resize(img_rgb, (width, height), interpolation=interpolation)


def read_crop_and_resize_rgb(path: Path, crop: str, width: int, height: int) -> np.ndarray:
    img_rgb = read_rgb(path)
    img_rgb = crop_rgb(img_rgb, crop)
    return resize_rgb(img_rgb, width=width, height=height)


def to_canny_rgb(img_rgb: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def rgb_to_cond_tensor(img_rgb: np.ndarray, device: str) -> torch.Tensor:
    arr = img_rgb.astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    return ten.to(device=device, dtype=torch.float32)


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f'Failed to write image: {path}')


def load_into_model(model: torch.nn.Module, ckpt_path: Path, strict: bool = False) -> Tuple[int, int]:
    state_dict = load_state_dict(str(ckpt_path), location='cpu')
    ret = model.load_state_dict(state_dict, strict=strict)
    return len(ret.missing_keys), len(ret.unexpected_keys)


def build_model(
    config: Path,
    control_ckpt: Path,
    sd_ckpt: Path,
    smart_ckpt: Optional[Path],
    device: str,
) -> torch.nn.Module:
    if smart_ckpt is not None and not smart_ckpt.exists():
        raise FileNotFoundError(f'Smart checkpoint not found: {smart_ckpt}')

    model = create_model(str(config)).cpu()
    load_into_model(model, control_ckpt, strict=False)
    load_into_model(model, sd_ckpt, strict=False)

    if smart_ckpt is not None:
        missing, unexpected = load_into_model(model, smart_ckpt, strict=False)
        print(f'[smart-ckpt] {smart_ckpt}: missing_keys={missing}, unexpected_keys={unexpected}')

    model = model.to(device).eval()
    return model


def apply_model_with_mode_and_scale(
    model: torch.nn.Module,
    x_noisy: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, Any],
    mode: str,
    control_scale: float,
) -> torch.Tensor:
    assert isinstance(cond, dict), 'cond must be dict'
    diffusion_model = model.model.diffusion_model
    cond_txt = torch.cat(cond['c_crossattn'], dim=1)

    if cond['c_concat'] is None:
        return diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=None,
            only_mid_control=model.only_mid_control,
        )

    hint = torch.cat(cond['c_concat'], dim=1)
    control = model.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt)
    if control_scale != 1.0:
        control = [c * float(control_scale) for c in control]

    return diffusion_model(
        x=x_noisy,
        timesteps=t,
        context=cond_txt,
        control=control,
        only_mid_control=model.only_mid_control,
        mode=mode,
        impath=None,
        c_pre=(None if mode == 'c_fix' else model.c_pre_list),
    )


@torch.no_grad()
def infer_many(
    model: torch.nn.Module,
    canny_rgb: np.ndarray,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg_scale: float,
    eta: float,
    base_seed: int,
    num_samples: int,
    mode: str,
    device: str,
    control_scale: float,
) -> Tuple[List[np.ndarray], List[int]]:
    cond = rgb_to_cond_tensor(canny_rgb, device=device)
    cond = cond.repeat(num_samples, 1, 1, 1)

    prompts = [prompt] * num_samples
    c = model.get_learned_conditioning(prompts)
    if negative_prompt:
        uc = model.get_learned_conditioning([negative_prompt] * num_samples)
    else:
        uc = model.get_unconditional_conditioning(num_samples)

    cond_dict = {'c_concat': [cond], 'c_crossattn': [c]}
    uc_dict = {'c_concat': [cond], 'c_crossattn': [uc]}

    sampler = DDIMSampler(model)
    _, _, h, w = cond.shape
    shape = (model.channels, h // 8, w // 8)

    seeds = [int(base_seed) + i for i in range(num_samples)]
    noise_list: List[torch.Tensor] = []
    for seed in seeds:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        noise_list.append(noise)
    x_t = torch.stack(noise_list, dim=0).to(device)

    original_apply = model.apply_model
    model.apply_model = lambda x, t, cc, *a, **k: apply_model_with_mode_and_scale(
        model,
        x_noisy=x,
        t=t,
        cond=cc,
        mode=mode,
        control_scale=control_scale,
    )
    try:
        samples, _ = sampler.sample(
            S=steps,
            batch_size=num_samples,
            shape=shape,
            conditioning=cond_dict,
            eta=eta,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uc_dict,
            verbose=False,
            x_T=x_t,
        )
    finally:
        model.apply_model = original_apply

    decoded = model.decode_first_stage(samples)
    decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

    outputs: List[np.ndarray] = []
    for idx in range(decoded.shape[0]):
        image = decoded[idx].permute(1, 2, 0).detach().cpu().numpy()
        outputs.append((image * 255.0).astype(np.uint8))
    return outputs, seeds


def annotate_tile(img_rgb: np.ndarray, label: str) -> np.ndarray:
    canvas = np.full((img_rgb.shape[0] + 28, img_rgb.shape[1], 3), 255, dtype=np.uint8)
    canvas[28:, :, :] = img_rgb
    cv2.rectangle(canvas, (0, 0), (img_rgb.shape[1] - 1, 27), (220, 220, 220), thickness=1)
    cv2.putText(
        canvas,
        label,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_overview_panel(input_rgb: np.ndarray, canny_rgb: np.ndarray, outputs: Sequence[np.ndarray], seeds: Sequence[int]) -> np.ndarray:
    tiles = [annotate_tile(input_rgb, 'input'), annotate_tile(canny_rgb, 'canny')]
    for idx, (img, seed) in enumerate(zip(outputs, seeds), start=1):
        tiles.append(annotate_tile(img, f'out_{idx:02d} seed={seed}'))
    return np.concatenate(tiles, axis=1)


def build_compare_panel(input_rgb: np.ndarray, canny_rgb: np.ndarray, pred_pre: np.ndarray, pred_ours: np.ndarray, seed: int) -> np.ndarray:
    tiles = [
        annotate_tile(input_rgb, 'input'),
        annotate_tile(canny_rgb, 'canny'),
        annotate_tile(pred_pre, f'pre_train c_fix seed={seed}'),
        annotate_tile(pred_ours, f'ours c_ada seed={seed}'),
    ]
    return np.concatenate(tiles, axis=1)


def pixel_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def validate_args(args: argparse.Namespace) -> None:
    if args.low_threshold < 0 or args.high_threshold < 0:
        raise ValueError('Canny thresholds must be >= 0.')
    if args.low_threshold > args.high_threshold:
        raise ValueError('low-threshold must be <= high-threshold.')
    if args.width <= 0 or args.height <= 0:
        raise ValueError('width and height must be positive.')
    if args.steps <= 0:
        raise ValueError('steps must be positive.')
    if args.cfg <= 0:
        raise ValueError('cfg must be positive.')
    if args.num_samples <= 0:
        raise ValueError('num-samples must be positive.')
    if args.control_scale <= 0:
        raise ValueError('control-scale must be positive.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    for path in (args.image, args.config, args.control_ckpt, args.sd_ckpt):
        if not path.exists():
            raise FileNotFoundError(f'Required file not found: {path}')

    if args.compare_pretrain:
        if args.num_samples != 1:
            raise ValueError('Compare mode requires --num-samples 1.')
        if not args.compare_smart_ckpt.exists():
            raise FileNotFoundError(f'Compare smart checkpoint not found: {args.compare_smart_ckpt}')
    elif args.mode == 'c_ada' and not args.smart_ckpt.exists():
        raise FileNotFoundError(f'Smart checkpoint not found: {args.smart_ckpt}')


def run_compare_mode(args: argparse.Namespace, image: np.ndarray, canny: np.ndarray, out_dir: Path) -> None:
    model_pre = build_model(
        config=args.config,
        control_ckpt=args.control_ckpt,
        sd_ckpt=args.sd_ckpt,
        smart_ckpt=None,
        device=args.device,
    )
    pre_outputs, pre_seeds = infer_many(
        model=model_pre,
        canny_rgb=canny,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        cfg_scale=args.cfg,
        eta=args.eta,
        base_seed=args.seed,
        num_samples=1,
        mode='c_fix',
        device=args.device,
        control_scale=args.control_scale,
    )
    del model_pre
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    model_ours = build_model(
        config=args.config,
        control_ckpt=args.control_ckpt,
        sd_ckpt=args.sd_ckpt,
        smart_ckpt=args.compare_smart_ckpt,
        device=args.device,
    )
    ours_outputs, ours_seeds = infer_many(
        model=model_ours,
        canny_rgb=canny,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        cfg_scale=args.cfg,
        eta=args.eta,
        base_seed=args.seed,
        num_samples=1,
        mode='c_ada',
        device=args.device,
        control_scale=args.control_scale,
    )
    del model_ours
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    pred_pre = pre_outputs[0]
    pred_ours = ours_outputs[0]
    seed = pre_seeds[0]

    save_rgb(out_dir / 'input_rgb.png', image)
    save_rgb(out_dir / 'input_canny.png', canny)
    save_rgb(out_dir / 'pred_pretrain_fix.png', pred_pre)
    save_rgb(out_dir / 'pred_ours_ada.png', pred_ours)
    save_rgb(out_dir / 'panel_compare_4col.png', build_compare_panel(image, canny, pred_pre, pred_ours, seed))

    meta = {
        'image': str(args.image),
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'compare_pretrain': True,
        'crop': args.crop,
        'width': args.width,
        'height': args.height,
        'low_threshold': args.low_threshold,
        'high_threshold': args.high_threshold,
        'steps': args.steps,
        'cfg': args.cfg,
        'eta': args.eta,
        'control_scale': args.control_scale,
        'seed': seed,
        'pretrain_mode': 'c_fix',
        'ours_mode': 'c_ada',
        'compare_smart_ckpt': str(args.compare_smart_ckpt),
        'control_ckpt': str(args.control_ckpt),
        'sd_ckpt': str(args.sd_ckpt),
        'mae_pre_vs_ours': pixel_mae(pred_pre, pred_ours),
        'pretrain_seed': seed,
        'ours_seed': ours_seeds[0],
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Saved outputs to:', out_dir)
    print(f'  pred_pretrain_fix.png seed={seed}')
    print(f'  pred_ours_ada.png seed={ours_seeds[0]}')
    print(f'  MAE(pretrain vs ours)={meta["mae_pre_vs_ours"]:.6f}')


def run_single_mode(args: argparse.Namespace, image: np.ndarray, canny: np.ndarray, out_dir: Path) -> None:
    model = build_model(
        config=args.config,
        control_ckpt=args.control_ckpt,
        sd_ckpt=args.sd_ckpt,
        smart_ckpt=(args.smart_ckpt if args.mode == 'c_ada' else None),
        device=args.device,
    )
    outputs, seeds = infer_many(
        model=model,
        canny_rgb=canny,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        cfg_scale=args.cfg,
        eta=args.eta,
        base_seed=args.seed,
        num_samples=args.num_samples,
        mode=args.mode,
        device=args.device,
        control_scale=args.control_scale,
    )
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    save_rgb(out_dir / 'input_rgb.png', image)
    save_rgb(out_dir / 'input_canny.png', canny)
    for idx, (img_rgb, seed) in enumerate(zip(outputs, seeds), start=1):
        save_rgb(out_dir / f'sample_{idx:02d}_seed{seed}.png', img_rgb)

    panel = build_overview_panel(image, canny, outputs, seeds)
    save_rgb(out_dir / 'panel_overview.png', panel)

    meta = {
        'image': str(args.image),
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'compare_pretrain': False,
        'mode': args.mode,
        'crop': args.crop,
        'width': args.width,
        'height': args.height,
        'low_threshold': args.low_threshold,
        'high_threshold': args.high_threshold,
        'steps': args.steps,
        'cfg': args.cfg,
        'eta': args.eta,
        'control_scale': args.control_scale,
        'seed': args.seed,
        'num_samples': args.num_samples,
        'seeds': seeds,
        'smart_ckpt': str(args.smart_ckpt),
        'control_ckpt': str(args.control_ckpt),
        'sd_ckpt': str(args.sd_ckpt),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Saved outputs to:', out_dir)
    for idx, seed in enumerate(seeds, start=1):
        print(f'  sample_{idx:02d}: seed={seed}')


def main() -> None:
    args = parse_args()
    validate_args(args)

    image = read_crop_and_resize_rgb(
        args.image,
        crop=args.crop,
        width=args.width,
        height=args.height,
    )
    canny = to_canny_rgb(
        image,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_pretrain:
        run_compare_mode(args=args, image=image, canny=canny, out_dir=out_dir)
    else:
        run_single_mode(args=args, image=image, canny=canny, out_dir=out_dir)


if __name__ == '__main__':
    main()
