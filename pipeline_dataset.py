from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


THIS_FILE = Path(__file__).resolve()
TRAIN_ROOT = THIS_FILE.parents[2]
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_DEFAULT_CONFIG_DIR_CANDIDATES = [
    THIS_FILE.parent / "data" / "get_image" / "configs",
    THIS_FILE.parent / "configs",
]
for _candidate in _DEFAULT_CONFIG_DIR_CANDIDATES:
    if _candidate.exists():
        DEFAULT_CONFIG_DIR = _candidate
        break
else:
    DEFAULT_CONFIG_DIR = _DEFAULT_CONFIG_DIR_CANDIDATES[0]
DEFAULT_PATHS_CONFIG = DEFAULT_CONFIG_DIR / "paths.json"
DEFAULT_INFERENCE_CONFIG = DEFAULT_CONFIG_DIR / "inference.json"
DEFAULT_CLASS_MAP_CONFIG = DEFAULT_CONFIG_DIR / "class_map.json"
DEFAULT_ALPHA_LIST_CONFIG = DEFAULT_CONFIG_DIR / "alpha_list.json"


def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: cv2. Please install opencv-python in this environment."
        ) from exc
    return cv2


def require_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: torch. Please install torch in this environment."
        ) from exc
    return torch


@dataclass
class PathsConfig:
    repo_root: Path
    openimages_root: Path
    work_root: Path
    prompt_json_root: Path
    model_config: Path
    control_ckpt: Path
    sd_ckpt: Path
    smart_ckpt: Optional[Path]


@dataclass
class InferenceConfig:
    device: str
    width: int
    height: int
    canny_low_threshold: int
    canny_high_threshold: int
    ddim_steps: int
    cfg_scale: float
    eta: float
    batch_size: int
    seed: int
    mode: str
    load_smart_ckpt: bool
    negative_prompt: str


@dataclass
class PromptIndex:
    by_path: Dict[str, str]
    by_filename: Dict[str, str]
    by_stem: Dict[str, str]


def infer_repo_root(start: Path) -> Path:
    probe = start.resolve()
    candidates = [probe] + list(probe.parents)
    for candidate in candidates:
        if (candidate / "project_plan.md").exists():
            return candidate
    if len(probe.parents) >= 5:
        return probe.parents[4]
    return probe


def normalize_path_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def stable_seed(pair_id: str, base_seed: int) -> int:
    seed_delta = int(hashlib.sha1(pair_id.encode("utf-8")).hexdigest()[:8], 16)
    return int((base_seed + seed_delta) % (2**31 - 1))


def alpha_tag(alpha: float) -> str:
    alpha_x10 = round(alpha * 10.0)
    if abs(alpha * 10.0 - alpha_x10) < 1e-9 and 0 <= alpha_x10 <= 99:
        return f"a{alpha_x10:02d}"
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    return f"a{text}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            records.append(obj)
    return records


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_repo_relative_path(value: str, repo_root: Path) -> Optional[Path]:
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("TODO"):
        return None
    p = Path(text).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def resolve_cli_path(path_arg: Optional[str], default: Path, repo_root: Path) -> Path:
    if path_arg is None:
        return default.resolve()
    p = Path(path_arg).expanduser()
    if p.is_absolute():
        return p.resolve()
    if str(p).startswith("SmartControl") or str(p).startswith("SemanticControl"):
        return (repo_root / p).resolve()
    return (Path.cwd() / p).resolve()


def load_paths_config(config_path: Path, repo_root: Path) -> PathsConfig:
    cfg = load_json(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid paths config: {config_path}")

    required_keys = [
        "openimages_root",
        "work_root",
        "prompt_json_root",
        "model_config",
        "control_ckpt",
        "sd_ckpt",
    ]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing key '{key}' in {config_path}")

    openimages_root = resolve_repo_relative_path(str(cfg["openimages_root"]), repo_root)
    work_root = resolve_repo_relative_path(str(cfg["work_root"]), repo_root)
    prompt_json_root = resolve_repo_relative_path(str(cfg["prompt_json_root"]), repo_root)
    model_config = resolve_repo_relative_path(str(cfg["model_config"]), repo_root)
    control_ckpt = resolve_repo_relative_path(str(cfg["control_ckpt"]), repo_root)
    sd_ckpt = resolve_repo_relative_path(str(cfg["sd_ckpt"]), repo_root)
    smart_ckpt = resolve_repo_relative_path(str(cfg.get("smart_ckpt", "")), repo_root)

    assert openimages_root is not None
    assert work_root is not None
    assert prompt_json_root is not None
    assert model_config is not None
    assert control_ckpt is not None
    assert sd_ckpt is not None

    return PathsConfig(
        repo_root=repo_root,
        openimages_root=openimages_root,
        work_root=work_root,
        prompt_json_root=prompt_json_root,
        model_config=model_config,
        control_ckpt=control_ckpt,
        sd_ckpt=sd_ckpt,
        smart_ckpt=smart_ckpt,
    )


def load_inference_config(config_path: Path) -> InferenceConfig:
    cfg = load_json(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid inference config: {config_path}")

    def get_int(name: str, default: int) -> int:
        return int(cfg.get(name, default))

    def get_float(name: str, default: float) -> float:
        return float(cfg.get(name, default))

    device = str(cfg.get("device", "cuda")).strip().lower()
    mode = str(cfg.get("mode", "c_fix")).strip().lower()
    if mode not in {"c_fix", "c_ada"}:
        raise ValueError(f"mode must be c_fix or c_ada, got: {mode}")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"device must be cuda or cpu, got: {device}")

    inf = InferenceConfig(
        device=device,
        width=get_int("width", 512),
        height=get_int("height", 512),
        canny_low_threshold=get_int("canny_low_threshold", 100),
        canny_high_threshold=get_int("canny_high_threshold", 200),
        ddim_steps=get_int("ddim_steps", 30),
        cfg_scale=get_float("cfg_scale", 7.5),
        eta=get_float("eta", 0.0),
        batch_size=max(get_int("batch_size", 1), 1),
        seed=get_int("seed", 12345),
        mode=mode,
        load_smart_ckpt=bool(cfg.get("load_smart_ckpt", False)),
        negative_prompt=str(cfg.get("negative_prompt", "")),
    )

    if inf.width <= 0 or inf.height <= 0:
        raise ValueError("width and height must be positive")
    if inf.canny_low_threshold < 0 or inf.canny_high_threshold < 0:
        raise ValueError("canny thresholds must be non-negative")
    if inf.canny_low_threshold > inf.canny_high_threshold:
        raise ValueError("canny_low_threshold must be <= canny_high_threshold")
    if inf.ddim_steps <= 0:
        raise ValueError("ddim_steps must be positive")
    if inf.cfg_scale <= 0:
        raise ValueError("cfg_scale must be positive")
    return inf


def load_class_map(config_path: Path) -> Dict[str, List[str]]:
    cfg = load_json(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid class map config: {config_path}")
    output: Dict[str, List[str]] = {}
    for key, value in cfg.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, list):
            raise ValueError(f"class_map[{key}] must be a list")
        targets: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"class_map[{key}] contains non-string target")
            token = item.strip()
            if token:
                targets.append(token)
        output[key.strip()] = targets
    return output


def load_alpha_list(config_path: Path) -> List[float]:
    cfg = load_json(config_path)
    if not isinstance(cfg, dict) or "alphas" not in cfg:
        raise ValueError(f"Invalid alpha list config: {config_path}")
    alphas_raw = cfg["alphas"]
    if not isinstance(alphas_raw, list) or not alphas_raw:
        raise ValueError(f"'alphas' must be a non-empty list in {config_path}")
    alphas = [float(v) for v in alphas_raw]
    if any((v < 0.0 or v > 1.0) for v in alphas):
        raise ValueError("All alpha values must be within [0.0, 1.0]")
    uniq = sorted({float(v) for v in alphas}, reverse=True)
    if not uniq:
        raise ValueError("No valid alpha values found")
    return uniq


def iter_image_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_image_rgb(path: Path) -> np.ndarray:
    cv2 = require_cv2()
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: Path, img_rgb: np.ndarray) -> None:
    cv2 = require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    cv2 = require_cv2()
    if img.shape[1] == width and img.shape[0] == height:
        return img
    interpolation = cv2.INTER_AREA if (img.shape[1] > width or img.shape[0] > height) else cv2.INTER_LINEAR
    return cv2.resize(img, (width, height), interpolation=interpolation)


def make_canny_rgb(img_rgb: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    cv2 = require_cv2()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def load_prompt_index(class_name: str, prompt_json_root: Path) -> PromptIndex:
    prompt_path = prompt_json_root / f"{class_name}.json"
    if not prompt_path.exists():
        return PromptIndex(by_path={}, by_filename={}, by_stem={})
    data = load_json(prompt_path)
    if not isinstance(data, list):
        raise ValueError(f"{prompt_path} must be a JSON array")

    by_path: Dict[str, str] = {}
    by_filename: Dict[str, str] = {}
    by_stem: Dict[str, str] = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        source = str(obj.get("source", "")).strip()
        prompt = str(obj.get("prompt", "")).strip()
        if not source or not prompt:
            continue
        src_path = Path(source)
        by_filename.setdefault(src_path.name.lower(), prompt)
        by_stem.setdefault(src_path.stem.lower(), prompt)
        by_path.setdefault(normalize_path_key(source), prompt)
        if src_path.is_absolute():
            by_path.setdefault(normalize_path_key(str(src_path.resolve())), prompt)
        else:
            by_path.setdefault(
                normalize_path_key(str((prompt_json_root / src_path).resolve())),
                prompt,
            )
    return PromptIndex(by_path=by_path, by_filename=by_filename, by_stem=by_stem)


def resolve_prompt(img_path: Path, prompt_index: PromptIndex, clsinit: str) -> str:
    candidates = [
        normalize_path_key(str(img_path.resolve())),
        normalize_path_key(str(img_path)),
        img_path.name.lower(),
        img_path.stem.lower(),
    ]
    for key in candidates[:2]:
        if key in prompt_index.by_path:
            return prompt_index.by_path[key]
    if candidates[2] in prompt_index.by_filename:
        return prompt_index.by_filename[candidates[2]]
    if candidates[3] in prompt_index.by_stem:
        return prompt_index.by_stem[candidates[3]]
    return f"a photo of a {clsinit.lower()}"


def rewrite_prompt(original_prompt: str, clsinit: str, clsalt: str) -> Tuple[str, bool]:
    text = str(original_prompt).strip()
    if not text:
        return f"a photo of a {clsalt.lower()}", False

    replaced = False
    if re.fullmatch(r"[A-Za-z0-9_]+", clsinit):
        pattern = re.compile(rf"\b{re.escape(clsinit)}\b", flags=re.IGNORECASE)
        text, count = pattern.subn(clsalt, text)
        replaced = count > 0

    if not replaced:
        pattern = re.compile(re.escape(clsinit), flags=re.IGNORECASE)
        text, count = pattern.subn(clsalt, text)
        replaced = count > 0

    if not replaced:
        return f"a photo of a {clsalt.lower()}", False
    return text, True


def load_state_into_model(model: torch.nn.Module, ckpt_path: Path, strict: bool = False) -> Tuple[int, int]:
    from cldm.model import load_state_dict  # type: ignore

    state_dict = load_state_dict(str(ckpt_path), location="cpu")
    ret = model.load_state_dict(state_dict, strict=strict)
    return len(ret.missing_keys), len(ret.unexpected_keys)


def apply_model_with_mode_and_alpha(
    model: torch.nn.Module,
    x_noisy: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, Any],
    mode: str,
    alpha: float,
) -> torch.Tensor:
    torch = require_torch()
    assert isinstance(cond, dict), "cond must be dict"
    diffusion_model = model.model.diffusion_model
    cond_txt = torch.cat(cond["c_crossattn"], dim=1)

    if cond["c_concat"] is None:
        return diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=None,
            only_mid_control=model.only_mid_control,
        )

    hint = torch.cat(cond["c_concat"], dim=1)
    control = model.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt)
    if alpha != 1.0:
        control = [c * float(alpha) for c in control]

    return diffusion_model(
        x=x_noisy,
        timesteps=t,
        context=cond_txt,
        control=control,
        only_mid_control=model.only_mid_control,
        mode=mode,
        impath=None,
        c_pre=(None if mode == "c_fix" else model.c_pre_list),
    )


class SmartControlInferencer:
    def __init__(self, paths_cfg: PathsConfig, inf_cfg: InferenceConfig) -> None:
        torch = require_torch()
        from cldm.model import create_model  # type: ignore
        from ldm.models.diffusion.ddim import DDIMSampler  # type: ignore

        if inf_cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

        self.paths_cfg = paths_cfg
        self.inf_cfg = inf_cfg
        self.device = inf_cfg.device
        self.model = create_model(str(paths_cfg.model_config)).cpu()

        print(f"[model] loading control ckpt: {paths_cfg.control_ckpt}")
        load_state_into_model(self.model, paths_cfg.control_ckpt, strict=False)
        print(f"[model] loading sd ckpt: {paths_cfg.sd_ckpt}")
        load_state_into_model(self.model, paths_cfg.sd_ckpt, strict=False)

        need_smart = inf_cfg.mode == "c_ada" or inf_cfg.load_smart_ckpt
        if need_smart:
            if paths_cfg.smart_ckpt is None:
                raise RuntimeError(
                    "Smart checkpoint is required for c_ada/load_smart_ckpt, "
                    "but paths.json has TODO/empty smart_ckpt."
                )
            if not paths_cfg.smart_ckpt.exists():
                raise FileNotFoundError(f"smart_ckpt not found: {paths_cfg.smart_ckpt}")
            print(f"[model] loading smart ckpt: {paths_cfg.smart_ckpt}")
            missing, unexpected = load_state_into_model(self.model, paths_cfg.smart_ckpt, strict=False)
            print(f"[model] smart ckpt load summary: missing_keys={missing}, unexpected_keys={unexpected}")

        self.model = self.model.to(self.device).eval()
        self.sampler = DDIMSampler(self.model)

    def generate_batch(
        self,
        prompts: Sequence[str],
        condition_rgbs: Sequence[np.ndarray],
        alpha: float,
        seeds: Sequence[int],
    ) -> List[np.ndarray]:
        torch = require_torch()
        with torch.no_grad():
            if len(prompts) == 0:
                return []
            if len(prompts) != len(condition_rgbs) or len(prompts) != len(seeds):
                raise ValueError("prompts/condition_rgbs/seeds length mismatch")

            width = self.inf_cfg.width
            height = self.inf_cfg.height
            cond_batch = np.stack(
                [resize_rgb(img, width=width, height=height).astype(np.float32) / 255.0 for img in condition_rgbs],
                axis=0,
            )
            cond_tensor = torch.from_numpy(cond_batch).permute(0, 3, 1, 2).contiguous()
            cond_tensor = cond_tensor.to(device=self.device, dtype=torch.float32)

            batch_size = len(prompts)
            c = self.model.get_learned_conditioning(list(prompts))

            if self.inf_cfg.negative_prompt:
                uc = self.model.get_learned_conditioning([self.inf_cfg.negative_prompt] * batch_size)
            else:
                uc = self.model.get_unconditional_conditioning(batch_size)

            cond_dict = {"c_concat": [cond_tensor], "c_crossattn": [c]}
            uc_dict = {"c_concat": [cond_tensor], "c_crossattn": [uc]}

            shape = (self.model.channels, height // 8, width // 8)
            noise_list: List[torch.Tensor] = []
            for seed in seeds:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed))
                noise = torch.randn(shape, generator=generator, dtype=torch.float32)
                noise_list.append(noise)
            x_t = torch.stack(noise_list, dim=0).to(self.device)

            original_apply = self.model.apply_model
            self.model.apply_model = lambda x, t, cond, *a, **k: apply_model_with_mode_and_alpha(
                self.model,
                x_noisy=x,
                t=t,
                cond=cond,
                mode=self.inf_cfg.mode,
                alpha=alpha,
            )
            try:
                samples, _ = self.sampler.sample(
                    S=self.inf_cfg.ddim_steps,
                    batch_size=batch_size,
                    shape=shape,
                    conditioning=cond_dict,
                    eta=self.inf_cfg.eta,
                    unconditional_guidance_scale=self.inf_cfg.cfg_scale,
                    unconditional_conditioning=uc_dict,
                    verbose=False,
                    x_T=x_t,
                )
            finally:
                self.model.apply_model = original_apply

            decoded = self.model.decode_first_stage(samples)
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            out: List[np.ndarray] = []
            for idx in range(decoded.shape[0]):
                img = decoded[idx].permute(1, 2, 0).detach().cpu().numpy()
                out.append((img * 255.0).astype(np.uint8))
            return out


def build_tile(
    img_rgb: np.ndarray,
    label: str,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
    label_height: int = 28,
) -> Image.Image:
    resized = resize_rgb(img_rgb, width=width, height=height)
    tile = Image.new("RGB", (width, height + label_height), color=(255, 255, 255))
    tile.paste(Image.fromarray(resized), (0, label_height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width - 1, label_height - 1), outline=(210, 210, 210), fill=(245, 245, 245))
    draw.text((8, 6), label, fill=(20, 20, 20), font=font)
    return tile


def compose_contact_sheet(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    width: int,
    height: int,
) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels length mismatch")
    font = ImageFont.load_default()
    tiles = [build_tile(img, label, width, height, font) for img, label in zip(images, labels)]
    tile_w, tile_h = tiles[0].size
    panel = Image.new("RGB", (tile_w * len(tiles), tile_h), color=(255, 255, 255))
    for i, tile in enumerate(tiles):
        panel.paste(tile, (i * tile_w, 0))
    return panel


def compose_paper_panel(
    source_rgb: np.ndarray,
    condition_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    caption: str,
    width: int,
    height: int,
) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 26
    caption_height = 56
    tiles = [
        build_tile(source_rgb, "source", width, height, font, label_height=label_height),
        build_tile(condition_rgb, "canny", width, height, font, label_height=label_height),
        build_tile(generated_rgb, "generated", width, height, font, label_height=label_height),
    ]
    tile_w, tile_h = tiles[0].size
    canvas = Image.new(
        "RGB",
        (tile_w * 3, tile_h + caption_height),
        color=(255, 255, 255),
    )
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * tile_w, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (0, tile_h, tile_w * 3 - 1, tile_h + caption_height - 1),
        outline=(210, 210, 210),
        fill=(250, 250, 250),
    )
    wrapped = textwrap.fill(caption, width=100)
    draw.text((12, tile_h + 8), wrapped, fill=(30, 30, 30), font=font)
    return canvas


def default_manifest_paths(work_root: Path) -> Dict[str, Path]:
    manifests_root = work_root / "manifests"
    return {
        "source_manifest": manifests_root / "source_manifest.jsonl",
        "candidates_manifest": manifests_root / "candidates.jsonl",
        "selection_template": manifests_root / "selection_template.csv",
        "selection_csv": manifests_root / "selection.csv",
        "final_pairs": manifests_root / "final_pairs.jsonl",
        "prepare_errors": manifests_root / "prepare_errors.log",
        "generate_errors": manifests_root / "generate_errors.log",
        "finalize_errors": manifests_root / "finalize_errors.log",
        "build_train_manifest": manifests_root / "train_with_masks.jsonl",
        "build_train_errors": manifests_root / "build_train_data_errors.log",
    }


def write_error_log(path: Path, errors: Sequence[str]) -> None:
    if not errors:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in errors:
            f.write(line + "\n")


def cmd_prepare(args: argparse.Namespace, paths_cfg: PathsConfig, inf_cfg: InferenceConfig) -> None:
    openimages_root = paths_cfg.openimages_root
    work_root = paths_cfg.work_root
    manifests = default_manifest_paths(work_root)
    source_manifest_path = resolve_cli_path(args.source_manifest, manifests["source_manifest"], paths_cfg.repo_root)
    errors_path = resolve_cli_path(args.error_log, manifests["prepare_errors"], paths_cfg.repo_root)

    if not openimages_root.exists():
        raise FileNotFoundError(f"openimages_root not found: {openimages_root}")

    class_dirs = [p for p in sorted(openimages_root.iterdir()) if p.is_dir()]
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        class_dirs = [p for p in class_dirs if p.name in wanted]

    print(f"[prepare] openimages_root={openimages_root}")
    print(f"[prepare] work_root={work_root}")
    print(f"[prepare] classes={len(class_dirs)}")

    prompt_cache: Dict[str, PromptIndex] = {}
    manifest_rows: List[dict] = []
    errors: List[str] = []
    existing_condition_targets: Dict[str, Path] = {}

    total_images = 0
    generated_canny = 0
    reused_canny = 0
    fallback_prompt_count = 0

    for class_dir in class_dirs:
        clsinit = class_dir.name
        prompt_cache[clsinit] = load_prompt_index(clsinit, paths_cfg.prompt_json_root)
        class_images = list(iter_image_files(class_dir))
        print(f"[prepare] class={clsinit}, images={len(class_images)}")

        for img_path in class_images:
            total_images += 1
            try:
                source_rgb = read_image_rgb(img_path)
            except Exception as exc:
                errors.append(f"[read-failed] {img_path}: {exc}")
                continue

            source_rel = img_path.relative_to(openimages_root).as_posix()
            sample_id = stable_hash(f"{clsinit}|{source_rel}")
            source_stem = img_path.stem
            condition_path = (work_root / "conditions" / "canny" / clsinit / f"{source_stem}.png").resolve()

            collision_key = normalize_path_key(str(condition_path))
            if collision_key in existing_condition_targets and existing_condition_targets[collision_key] != img_path:
                errors.append(
                    f"[stem-collision] condition={condition_path} reused by "
                    f"{existing_condition_targets[collision_key]} and {img_path}"
                )
                continue
            existing_condition_targets[collision_key] = img_path

            try:
                if condition_path.exists() and not args.overwrite:
                    reused_canny += 1
                else:
                    canny_rgb = make_canny_rgb(
                        source_rgb,
                        low_threshold=inf_cfg.canny_low_threshold,
                        high_threshold=inf_cfg.canny_high_threshold,
                    )
                    save_image_rgb(condition_path, canny_rgb)
                    generated_canny += 1
            except Exception as exc:
                errors.append(f"[canny-failed] {img_path}: {exc}")
                continue

            prompt = resolve_prompt(img_path, prompt_cache[clsinit], clsinit=clsinit)
            if prompt == f"a photo of a {clsinit.lower()}":
                fallback_prompt_count += 1

            manifest_rows.append(
                {
                    "id": sample_id,
                    "clsinit": clsinit,
                    "source": str(img_path.resolve()),
                    "source_filename": img_path.name,
                    "source_stem": source_stem,
                    "condition": str(condition_path),
                    "original_prompt": prompt,
                }
            )

    write_jsonl(source_manifest_path, manifest_rows)
    write_error_log(errors_path, errors)

    print(f"[prepare] total_images={total_images}")
    print(f"[prepare] source_manifest_rows={len(manifest_rows)}")
    print(f"[prepare] generated_canny={generated_canny}, reused_canny={reused_canny}")
    print(f"[prepare] fallback_prompt_count={fallback_prompt_count}")
    print(f"[prepare] source_manifest={source_manifest_path}")
    if errors:
        print(f"[prepare] errors={len(errors)} -> {errors_path}")
        if args.strict:
            raise RuntimeError("prepare encountered errors; see error log")


def build_candidate_rows(
    source_rows: Sequence[dict],
    class_map: Dict[str, List[str]],
    alphas: Sequence[float],
    work_root: Path,
    base_seed: int,
) -> List[dict]:
    rows: List[dict] = []
    for src in source_rows:
        clsinit = str(src["clsinit"])
        targets = class_map.get(clsinit, [])
        if not targets:
            continue
        source_stem = str(src["source_stem"])
        for clsalt in targets:
            if clsalt == clsinit:
                continue
            prompt, replaced = rewrite_prompt(
                original_prompt=str(src["original_prompt"]),
                clsinit=clsinit,
                clsalt=clsalt,
            )
            pair_id = stable_hash(f"{src['id']}|{clsalt}")
            candidate_seed = stable_seed(pair_id, base_seed=base_seed)
            for alpha in alphas:
                tag = alpha_tag(alpha)
                output_path = (
                    work_root
                    / "generated"
                    / f"{clsinit}_to_{clsalt}"
                    / source_stem
                    / f"{source_stem}__{clsinit}__{clsalt}__{tag}.png"
                ).resolve()
                rows.append(
                    {
                        "id": pair_id,
                        "candidate_id": stable_hash(f"{pair_id}|{alpha}"),
                        "clsinit": clsinit,
                        "clsalt": clsalt,
                        "source": str(Path(str(src["source"])).resolve()),
                        "condition": str(Path(str(src["condition"])).resolve()),
                        "original_prompt": str(src["original_prompt"]),
                        "prompt": prompt,
                        "prompt_replaced": bool(replaced),
                        "alpha": float(alpha),
                        "seed": int(candidate_seed),
                        "output": str(output_path),
                        "source_stem": source_stem,
                    }
                )
    return rows


def chunked(items: Sequence[dict], chunk_size: int) -> Iterable[List[dict]]:
    size = max(chunk_size, 1)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def save_selection_template(path: Path, candidate_rows: Sequence[dict]) -> None:
    grouped: Dict[str, dict] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["id"]), row)

    rows_sorted = sorted(
        grouped.values(),
        key=lambda r: (str(r["clsinit"]), str(r["clsalt"]), str(r["source_stem"])),
    )
    fieldnames = [
        "id",
        "clsinit",
        "clsalt",
        "source",
        "condition",
        "original_prompt",
        "prompt",
        "alpha_selected",
        "selected_output",
        "plausible",
        "keep",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(
                {
                    "id": row["id"],
                    "clsinit": row["clsinit"],
                    "clsalt": row["clsalt"],
                    "source": row["source"],
                    "condition": row["condition"],
                    "original_prompt": row["original_prompt"],
                    "prompt": row["prompt"],
                    "alpha_selected": "",
                    "selected_output": "",
                    "plausible": "",
                    "keep": "",
                    "notes": "",
                }
            )


def build_previews(
    candidate_rows: Sequence[dict],
    alpha_order: Sequence[float],
    work_root: Path,
    width: int,
    height: int,
    overwrite: bool,
) -> int:
    cv2 = require_cv2()
    grouped: Dict[Tuple[str, str, str, str], List[dict]] = {}
    for row in candidate_rows:
        key = (str(row["id"]), str(row["clsinit"]), str(row["clsalt"]), str(row["source_stem"]))
        grouped.setdefault(key, []).append(row)

    built_count = 0
    missing_placeholder = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(
        missing_placeholder,
        "MISSING",
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (180, 20, 20),
        2,
        cv2.LINE_AA,
    )

    for (_, clsinit, clsalt, source_stem), rows in grouped.items():
        preview_path = (
            work_root / "previews" / f"{clsinit}_to_{clsalt}" / f"{source_stem}.png"
        ).resolve()
        if preview_path.exists() and not overwrite:
            continue

        rows_by_alpha = {float(r["alpha"]): r for r in rows}
        base = rows[0]
        source_rgb = resize_rgb(read_image_rgb(Path(base["source"])), width=width, height=height)
        condition_rgb = resize_rgb(read_image_rgb(Path(base["condition"])), width=width, height=height)

        images: List[np.ndarray] = [source_rgb, condition_rgb]
        labels: List[str] = ["source", "canny"]
        for alpha in alpha_order:
            row = rows_by_alpha.get(float(alpha))
            labels.append(f"alpha={alpha:g}")
            if row is None:
                images.append(missing_placeholder.copy())
                continue
            output_path = Path(str(row["output"]))
            if output_path.exists():
                images.append(resize_rgb(read_image_rgb(output_path), width=width, height=height))
            else:
                images.append(missing_placeholder.copy())

        panel = compose_contact_sheet(images=images, labels=labels, width=width, height=height)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(preview_path)
        built_count += 1
    return built_count


def cmd_generate(
    args: argparse.Namespace,
    paths_cfg: PathsConfig,
    inf_cfg: InferenceConfig,
    class_map: Dict[str, List[str]],
    alphas: List[float],
) -> None:
    manifests = default_manifest_paths(paths_cfg.work_root)
    source_manifest_path = resolve_cli_path(args.source_manifest, manifests["source_manifest"], paths_cfg.repo_root)
    candidates_manifest_path = resolve_cli_path(args.candidates_manifest, manifests["candidates_manifest"], paths_cfg.repo_root)
    selection_template_path = resolve_cli_path(args.selection_template, manifests["selection_template"], paths_cfg.repo_root)
    errors_path = resolve_cli_path(args.error_log, manifests["generate_errors"], paths_cfg.repo_root)

    source_rows = load_jsonl(source_manifest_path)
    if not source_rows:
        raise RuntimeError(f"source manifest is empty: {source_manifest_path}")

    candidate_rows = build_candidate_rows(
        source_rows=source_rows,
        class_map=class_map,
        alphas=alphas,
        work_root=paths_cfg.work_root,
        base_seed=inf_cfg.seed,
    )
    if not candidate_rows:
        raise RuntimeError("No candidate rows were built. Check class_map.json")

    pending = [
        row
        for row in candidate_rows
        if args.overwrite or (not Path(str(row["output"])).exists())
    ]

    print(f"[generate] source_rows={len(source_rows)}")
    print(f"[generate] candidate_rows_total={len(candidate_rows)}")
    print(f"[generate] candidate_rows_pending={len(pending)}")
    print(f"[generate] mode={inf_cfg.mode}, batch_size={inf_cfg.batch_size}, ddim_steps={inf_cfg.ddim_steps}")

    errors: List[str] = []
    if pending:
        inferencer = SmartControlInferencer(paths_cfg=paths_cfg, inf_cfg=inf_cfg)
        pending_by_alpha: Dict[float, List[dict]] = {}
        for row in pending:
            pending_by_alpha.setdefault(float(row["alpha"]), []).append(row)

        for alpha in alphas:
            rows_alpha = pending_by_alpha.get(float(alpha), [])
            if not rows_alpha:
                continue
            print(f"[generate] alpha={alpha:g}, pending={len(rows_alpha)}")
            done_alpha = 0
            for batch in chunked(rows_alpha, inf_cfg.batch_size):
                valid_rows: List[dict] = []
                cond_images: List[np.ndarray] = []
                prompts: List[str] = []
                seeds: List[int] = []
                for row in batch:
                    condition_path = Path(str(row["condition"]))
                    if not condition_path.exists():
                        errors.append(f"[missing-condition] {condition_path}")
                        continue
                    try:
                        cond_rgb = read_image_rgb(condition_path)
                    except Exception as exc:
                        errors.append(f"[bad-condition] {condition_path}: {exc}")
                        continue
                    valid_rows.append(row)
                    cond_images.append(cond_rgb)
                    prompts.append(str(row["prompt"]))
                    seeds.append(int(row["seed"]))

                if not valid_rows:
                    continue

                try:
                    generated_images = inferencer.generate_batch(
                        prompts=prompts,
                        condition_rgbs=cond_images,
                        alpha=float(alpha),
                        seeds=seeds,
                    )
                except Exception as exc:
                    for row in valid_rows:
                        errors.append(
                            f"[batch-generate-failed] alpha={alpha}, output={row['output']}: {exc}"
                        )
                    continue

                for row, image_rgb in zip(valid_rows, generated_images):
                    output_path = Path(str(row["output"]))
                    try:
                        save_image_rgb(output_path, image_rgb)
                        done_alpha += 1
                    except Exception as exc:
                        errors.append(f"[write-failed] {output_path}: {exc}")
                if done_alpha and done_alpha % 20 == 0:
                    print(f"[generate] alpha={alpha:g}, done={done_alpha}/{len(rows_alpha)}")
            print(f"[generate] alpha={alpha:g}, finished={done_alpha}/{len(rows_alpha)}")

    missing_outputs = [row for row in candidate_rows if not Path(str(row["output"])).exists()]
    if missing_outputs:
        for row in missing_outputs[:200]:
            errors.append(f"[missing-output] {row['output']}")
        if len(missing_outputs) > 200:
            errors.append(f"[missing-output] ... +{len(missing_outputs) - 200} more")

    write_jsonl(candidates_manifest_path, candidate_rows)
    save_selection_template(selection_template_path, candidate_rows)

    preview_count = build_previews(
        candidate_rows=candidate_rows,
        alpha_order=alphas,
        work_root=paths_cfg.work_root,
        width=inf_cfg.width,
        height=inf_cfg.height,
        overwrite=args.overwrite_previews or args.overwrite,
    )
    write_error_log(errors_path, errors)

    print(f"[generate] candidates_manifest={candidates_manifest_path}")
    print(f"[generate] selection_template={selection_template_path}")
    print(f"[generate] preview_count={preview_count}")
    if errors:
        print(f"[generate] errors={len(errors)} -> {errors_path}")
        if args.strict:
            raise RuntimeError("generate encountered errors; see error log")


def parse_keep_flag(value: str) -> bool:
    token = str(value).strip().lower()
    return token in {"1", "true", "yes", "y"}


def resolve_existing_path(value: str, repo_root: Path, work_root: Path) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidates = [
        (Path.cwd() / p).resolve(),
        (work_root / p).resolve(),
        (repo_root / p).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def cmd_finalize(args: argparse.Namespace, paths_cfg: PathsConfig) -> None:
    manifests = default_manifest_paths(paths_cfg.work_root)
    selection_csv_path = resolve_cli_path(args.selection_csv, manifests["selection_csv"], paths_cfg.repo_root)
    candidates_manifest_path = resolve_cli_path(args.candidates_manifest, manifests["candidates_manifest"], paths_cfg.repo_root)
    final_pairs_path = resolve_cli_path(args.final_pairs, manifests["final_pairs"], paths_cfg.repo_root)
    errors_path = resolve_cli_path(args.error_log, manifests["finalize_errors"], paths_cfg.repo_root)

    if not selection_csv_path.exists():
        raise FileNotFoundError(f"selection.csv not found: {selection_csv_path}")
    candidate_rows = load_jsonl(candidates_manifest_path)
    candidate_by_id: Dict[str, List[dict]] = {}
    candidate_output_keys: Dict[Tuple[str, str], dict] = {}
    for row in candidate_rows:
        rid = str(row["id"])
        candidate_by_id.setdefault(rid, []).append(row)
        key = (rid, normalize_path_key(str(Path(str(row["output"])).resolve())))
        candidate_output_keys[key] = row

    output_rows: List[dict] = []
    errors: List[str] = []
    kept_count = 0

    with selection_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        base_required_cols = {
            "id",
            "clsinit",
            "clsalt",
            "source",
            "condition",
            "original_prompt",
            "prompt",
            "alpha_selected",
            "selected_output",
            "keep",
            "notes",
        }
        if reader.fieldnames is None:
            raise ValueError(f"selection.csv has no header: {selection_csv_path}")
        field_set = set(reader.fieldnames)
        if not base_required_cols.issubset(field_set):
            raise ValueError(f"selection.csv columns mismatch: {selection_csv_path}")
        has_plausible_col = "plausible" in field_set
        if args.require_plausible and (not has_plausible_col):
            raise ValueError(
                f"selection.csv missing required column 'plausible': {selection_csv_path}. "
                "Please regenerate selection_template.csv and refill selection.csv."
            )

        for line_no, row in enumerate(reader, start=2):
            keep = parse_keep_flag(row.get("keep", ""))
            if not keep:
                continue
            kept_count += 1
            plausible = True
            if has_plausible_col:
                plausible = parse_keep_flag(row.get("plausible", ""))
            if args.require_plausible and not plausible:
                errors.append(f"[line {line_no}] keep=1 but plausible is not set to 1/true")
                continue
            rid = str(row.get("id", "")).strip()
            if not rid:
                errors.append(f"[line {line_no}] missing id")
                continue

            alpha_selected_raw = str(row.get("alpha_selected", "")).strip()
            selected_output_raw = str(row.get("selected_output", "")).strip()
            if not alpha_selected_raw:
                errors.append(f"[line {line_no}] keep=1 but alpha_selected is empty")
                continue
            if not selected_output_raw:
                errors.append(f"[line {line_no}] keep=1 but selected_output is empty")
                continue

            try:
                alpha_selected = float(alpha_selected_raw)
            except ValueError:
                errors.append(f"[line {line_no}] invalid alpha_selected: {alpha_selected_raw}")
                continue

            selected_output_path = resolve_existing_path(
                selected_output_raw,
                repo_root=paths_cfg.repo_root,
                work_root=paths_cfg.work_root,
            )
            if not selected_output_path.exists():
                errors.append(f"[line {line_no}] selected_output missing: {selected_output_path}")
                continue

            id_candidates = candidate_by_id.get(rid, [])
            if not id_candidates:
                errors.append(f"[line {line_no}] id not found in candidates manifest: {rid}")
                continue

            key = (rid, normalize_path_key(str(selected_output_path.resolve())))
            matched = candidate_output_keys.get(key)
            if matched is None:
                errors.append(
                    f"[line {line_no}] selected_output does not belong to id={rid}: {selected_output_path}"
                )
                continue

            matched_alpha = float(matched["alpha"])
            if abs(matched_alpha - alpha_selected) > 1e-6:
                errors.append(
                    f"[line {line_no}] alpha_selected({alpha_selected}) mismatches "
                    f"candidate alpha({matched_alpha}) for {selected_output_path}"
                )
                continue

            output_rows.append(
                {
                    "id": rid,
                    "clsinit": str(row.get("clsinit", "")).strip(),
                    "clsalt": str(row.get("clsalt", "")).strip(),
                    "source": str(resolve_existing_path(str(row.get("source", "")), paths_cfg.repo_root, paths_cfg.work_root)),
                    "condition": str(resolve_existing_path(str(row.get("condition", "")), paths_cfg.repo_root, paths_cfg.work_root)),
                    "original_prompt": str(row.get("original_prompt", "")).strip(),
                    "prompt": str(row.get("prompt", "")).strip(),
                    "alpha_selected": float(alpha_selected),
                    "paired_image": str(selected_output_path.resolve()),
                    "plausible": bool(plausible),
                }
            )

    write_error_log(errors_path, errors)
    if errors and args.strict:
        raise RuntimeError("finalize encountered errors; see error log")

    write_jsonl(final_pairs_path, output_rows)
    print(f"[finalize] selection_rows_keep=1: {kept_count}")
    print(f"[finalize] final_pairs_rows={len(output_rows)}")
    print(f"[finalize] final_pairs={final_pairs_path}")
    if errors:
        print(f"[finalize] errors={len(errors)} -> {errors_path}")


def parse_bool_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    return parse_keep_flag(str(value))


def build_diff_proxy_masks(
    source_rgb: np.ndarray,
    paired_rgb: np.ndarray,
    diff_threshold: int,
    blur_kernel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cv2 = require_cv2()
    src_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    paired_gray = cv2.cvtColor(paired_rgb, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(src_gray, paired_gray)
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    m_conflict = (diff >= int(diff_threshold)).astype(np.uint8) * 255
    m_bg = np.where(m_conflict > 0, 0, 255).astype(np.uint8)
    return m_conflict, m_bg


def cmd_build_train_data(args: argparse.Namespace, paths_cfg: PathsConfig, inf_cfg: InferenceConfig) -> None:
    manifests = default_manifest_paths(paths_cfg.work_root)
    final_pairs_path = resolve_cli_path(args.final_pairs, manifests["final_pairs"], paths_cfg.repo_root)
    output_manifest_path = resolve_cli_path(args.output_manifest, manifests["build_train_manifest"], paths_cfg.repo_root)
    errors_path = resolve_cli_path(args.error_log, manifests["build_train_errors"], paths_cfg.repo_root)

    if not final_pairs_path.exists():
        raise FileNotFoundError(f"final_pairs not found: {final_pairs_path}")
    rows = load_jsonl(final_pairs_path)
    if not rows:
        raise RuntimeError(f"final_pairs is empty: {final_pairs_path}")

    train_root = resolve_cli_path(args.train_data_root, paths_cfg.work_root / "train_data_pack", paths_cfg.repo_root)
    control_dir = (train_root / "control_canny").resolve()
    image_dir = (train_root / "image_canny").resolve()
    conflict_dir = (train_root / "m_conflict").resolve()
    bg_dir = (train_root / "m_bg").resolve()

    mask_mode = str(args.mask_mode).strip().lower()
    if mask_mode not in {"none", "diff_proxy"}:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    errors: List[str] = []
    output_rows: List[dict] = []
    built_rows = 0
    skipped_rows = 0

    for row in rows:
        rid = str(row.get("id", "")).strip()
        clsinit = str(row.get("clsinit", "")).strip()
        clsalt = str(row.get("clsalt", "")).strip()
        if not rid or not clsinit or not clsalt:
            errors.append(f"[bad-row] missing id/clsinit/clsalt: {row}")
            skipped_rows += 1
            continue

        if args.require_plausible:
            if "plausible" not in row:
                errors.append(
                    f"[missing-plausible] id={rid} has no plausible field in final_pairs. "
                    "Re-run finalize with latest selection_template.csv."
                )
                skipped_rows += 1
                continue
            if not parse_bool_any(row.get("plausible", False)):
                errors.append(f"[not-plausible] id={rid} has plausible=0 and is skipped")
                skipped_rows += 1
                continue

        source_path = resolve_existing_path(str(row.get("source", "")), paths_cfg.repo_root, paths_cfg.work_root)
        condition_path = resolve_existing_path(str(row.get("condition", "")), paths_cfg.repo_root, paths_cfg.work_root)
        paired_path = resolve_existing_path(str(row.get("paired_image", "")), paths_cfg.repo_root, paths_cfg.work_root)
        if not source_path.exists() or not condition_path.exists() or not paired_path.exists():
            errors.append(
                f"[missing-input] id={rid}, source={source_path.exists()}, "
                f"condition={condition_path.exists()}, paired={paired_path.exists()}"
            )
            skipped_rows += 1
            continue

        try:
            source_rgb = resize_rgb(read_image_rgb(source_path), width=inf_cfg.width, height=inf_cfg.height)
            condition_rgb = resize_rgb(read_image_rgb(condition_path), width=inf_cfg.width, height=inf_cfg.height)
            paired_rgb = resize_rgb(read_image_rgb(paired_path), width=inf_cfg.width, height=inf_cfg.height)
        except Exception as exc:
            errors.append(f"[read-failed] id={rid}: {exc}")
            skipped_rows += 1
            continue

        try:
            alt_condition_rgb = make_canny_rgb(
                paired_rgb,
                low_threshold=inf_cfg.canny_low_threshold,
                high_threshold=inf_cfg.canny_high_threshold,
            )
        except Exception as exc:
            errors.append(f"[canny-failed] id={rid}: {exc}")
            skipped_rows += 1
            continue

        source_panel = np.concatenate([condition_rgb, alt_condition_rgb], axis=1)
        target_panel = np.concatenate([source_rgb, paired_rgb], axis=1)

        source_stem = source_path.stem
        name_hash = stable_hash(rid, length=8)
        panel_name = f"{source_stem}__{clsinit}__{clsalt}__{name_hash}.png"
        control_out = (control_dir / panel_name).resolve()
        image_out = (image_dir / panel_name).resolve()

        if (not args.overwrite) and control_out.exists() and image_out.exists():
            pass
        else:
            try:
                save_image_rgb(control_out, source_panel)
                save_image_rgb(image_out, target_panel)
            except Exception as exc:
                errors.append(f"[write-panel-failed] id={rid}: {exc}")
                skipped_rows += 1
                continue

        out_row = {
            "id": rid,
            "clsinit": clsinit,
            "clsalt": clsalt,
            "source": str(control_out),
            "target": str(image_out),
            "prompt": str(row.get("prompt", "")).strip(),
            "alpha_selected": float(row.get("alpha_selected", 0.0)),
            "source_image": str(source_path),
            "paired_image": str(paired_path),
            "condition_init": str(condition_path),
            "condition_alt": "generated_from_paired_image_canny",
            "mask_mode": mask_mode,
        }

        if mask_mode == "diff_proxy":
            try:
                m_conflict, m_bg = build_diff_proxy_masks(
                    source_rgb=source_rgb,
                    paired_rgb=paired_rgb,
                    diff_threshold=int(args.diff_threshold),
                    blur_kernel=int(args.diff_blur_kernel),
                )
                conf_out = (conflict_dir / panel_name).resolve()
                bg_out = (bg_dir / panel_name).resolve()
                save_image_rgb(conf_out, np.stack([m_conflict] * 3, axis=-1))
                save_image_rgb(bg_out, np.stack([m_bg] * 3, axis=-1))
                out_row["m_conflict"] = str(conf_out)
                out_row["m_bg"] = str(bg_out)
            except Exception as exc:
                errors.append(f"[mask-failed] id={rid}: {exc}")
                skipped_rows += 1
                continue

        output_rows.append(out_row)
        built_rows += 1

    write_jsonl(output_manifest_path, output_rows)
    write_error_log(errors_path, errors)

    print(f"[build-train-data] final_pairs_rows={len(rows)}")
    print(f"[build-train-data] built_rows={built_rows}")
    print(f"[build-train-data] skipped_rows={skipped_rows}")
    print(f"[build-train-data] output_manifest={output_manifest_path}")
    print(f"[build-train-data] train_data_root={train_root}")
    print(f"[build-train-data] mask_mode={mask_mode}")
    if mask_mode == "diff_proxy":
        print(
            "[build-train-data] NOTE: diff_proxy masks are heuristic proxies, "
            "not paper-ground-truth m_conflict/m_bg."
        )
    if errors:
        print(f"[build-train-data] errors={len(errors)} -> {errors_path}")
        if args.strict:
            raise RuntimeError("build-train-data encountered errors; see error log")


def cmd_panels(args: argparse.Namespace, paths_cfg: PathsConfig, inf_cfg: InferenceConfig) -> None:
    manifests = default_manifest_paths(paths_cfg.work_root)
    final_pairs_path = resolve_cli_path(args.final_pairs, manifests["final_pairs"], paths_cfg.repo_root)
    if not final_pairs_path.exists():
        raise FileNotFoundError(f"final_pairs not found: {final_pairs_path}")

    rows = load_jsonl(final_pairs_path)
    if not rows:
        raise RuntimeError(f"final_pairs is empty: {final_pairs_path}")

    panel_count = 0
    for row in rows:
        clsinit = str(row["clsinit"])
        clsalt = str(row["clsalt"])
        source_path = Path(str(row["source"]))
        condition_path = Path(str(row["condition"]))
        paired_path = Path(str(row["paired_image"]))
        source_stem = source_path.stem

        out_path = (
            paths_cfg.work_root
            / "paper_panels"
            / f"{clsinit}_to_{clsalt}"
            / f"{source_stem}.png"
        ).resolve()
        if out_path.exists() and not args.overwrite:
            continue

        if not source_path.exists() or not condition_path.exists() or not paired_path.exists():
            raise FileNotFoundError(
                f"Panel input missing: source={source_path.exists()}, "
                f"condition={condition_path.exists()}, paired={paired_path.exists()}"
            )

        source_rgb = read_image_rgb(source_path)
        condition_rgb = read_image_rgb(condition_path)
        paired_rgb = read_image_rgb(paired_path)

        caption = (
            f"{clsinit} -> {clsalt} | alpha={float(row['alpha_selected']):g} | "
            f"prompt: {str(row['prompt'])}"
        )
        panel = compose_paper_panel(
            source_rgb=source_rgb,
            condition_rgb=condition_rgb,
            generated_rgb=paired_rgb,
            caption=caption,
            width=inf_cfg.width,
            height=inf_cfg.height,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(out_path)
        panel_count += 1

    print(f"[panels] final_pairs_rows={len(rows)}")
    print(f"[panels] generated_panels={panel_count}")
    print(f"[panels] output_dir={paths_cfg.work_root / 'paper_panels'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local data-construction pipeline for rough-condition generation:\n"
            "prepare -> generate -> finalize -> build-train-data -> panels"
        )
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default=str(DEFAULT_PATHS_CONFIG),
        help="Path to paths.json",
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default=str(DEFAULT_INFERENCE_CONFIG),
        help="Path to inference.json",
    )
    parser.add_argument(
        "--class-map-config",
        type=str,
        default=str(DEFAULT_CLASS_MAP_CONFIG),
        help="Path to class_map.json",
    )
    parser.add_argument(
        "--alpha-list-config",
        type=str,
        default=str(DEFAULT_ALPHA_LIST_CONFIG),
        help="Path to alpha_list.json",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Stage 1: build canny and source manifest")
    prepare_parser.add_argument(
        "--source-manifest",
        type=str,
        default=None,
        help="Output source manifest path (default: manifests/source_manifest.jsonl)",
    )
    prepare_parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Optional comma-separated class filter, e.g. Tiger,Horse",
    )
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing canny files",
    )
    prepare_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any image fails",
    )
    prepare_parser.add_argument(
        "--error-log",
        type=str,
        default=None,
        help="Path for prepare error log",
    )

    generate_parser = subparsers.add_parser("generate", help="Stage 2: generate candidate images and previews")
    generate_parser.add_argument(
        "--source-manifest",
        type=str,
        default=None,
        help="Input source manifest path",
    )
    generate_parser.add_argument(
        "--candidates-manifest",
        type=str,
        default=None,
        help="Output candidates manifest path",
    )
    generate_parser.add_argument(
        "--selection-template",
        type=str,
        default=None,
        help="Output selection template csv path",
    )
    generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated candidates",
    )
    generate_parser.add_argument(
        "--overwrite-previews",
        action="store_true",
        help="Overwrite existing preview sheets",
    )
    generate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any generation error occurs",
    )
    generate_parser.add_argument(
        "--error-log",
        type=str,
        default=None,
        help="Path for generate error log",
    )

    finalize_parser = subparsers.add_parser("finalize", help="Stage 3: finalize selected pairs from selection.csv")
    finalize_parser.add_argument(
        "--selection-csv",
        type=str,
        default=None,
        help="Input selection.csv path",
    )
    finalize_parser.add_argument(
        "--candidates-manifest",
        type=str,
        default=None,
        help="Input candidates manifest for validation",
    )
    finalize_parser.add_argument(
        "--final-pairs",
        type=str,
        default=None,
        help="Output final_pairs.jsonl path",
    )
    finalize_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any finalize validation error occurs",
    )
    finalize_parser.add_argument(
        "--error-log",
        type=str,
        default=None,
        help="Path for finalize error log",
    )
    finalize_parser.add_argument(
        "--require-plausible",
        dest="require_plausible",
        action="store_true",
        default=True,
        help="Require plausible=1 for keep=1 rows (default: enabled)",
    )
    finalize_parser.add_argument(
        "--no-require-plausible",
        dest="require_plausible",
        action="store_false",
        help="Disable plausible hard-check (compatibility mode)",
    )

    build_train_parser = subparsers.add_parser(
        "build-train-data",
        help="Build training panels/manifest from final_pairs (optional proxy masks)",
    )
    build_train_parser.add_argument(
        "--final-pairs",
        type=str,
        default=None,
        help="Input final_pairs.jsonl path",
    )
    build_train_parser.add_argument(
        "--output-manifest",
        type=str,
        default=None,
        help="Output train manifest jsonl path",
    )
    build_train_parser.add_argument(
        "--train-data-root",
        type=str,
        default=None,
        help="Output root directory for control_canny/image_canny/masks",
    )
    build_train_parser.add_argument(
        "--mask-mode",
        type=str,
        default="none",
        choices=["none", "diff_proxy"],
        help="Mask generation mode; diff_proxy is heuristic and not paper ground-truth",
    )
    build_train_parser.add_argument(
        "--diff-threshold",
        type=int,
        default=26,
        help="Pixel diff threshold for diff_proxy conflict mask",
    )
    build_train_parser.add_argument(
        "--diff-blur-kernel",
        type=int,
        default=5,
        help="Gaussian blur kernel for diff_proxy (odd preferred; 0/1 disables)",
    )
    build_train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output panels/masks",
    )
    build_train_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any build-train-data error occurs",
    )
    build_train_parser.add_argument(
        "--error-log",
        type=str,
        default=None,
        help="Path for build-train-data error log",
    )
    build_train_parser.add_argument(
        "--require-plausible",
        dest="require_plausible",
        action="store_true",
        default=True,
        help="Require plausible=1 in final_pairs (default: enabled)",
    )
    build_train_parser.add_argument(
        "--no-require-plausible",
        dest="require_plausible",
        action="store_false",
        help="Disable plausible check for old final_pairs compatibility",
    )

    panels_parser = subparsers.add_parser("panels", help="Optional: export paper-style panels")
    panels_parser.add_argument(
        "--final-pairs",
        type=str,
        default=None,
        help="Input final_pairs.jsonl path",
    )
    panels_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing paper panels",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = infer_repo_root(THIS_FILE.parent)
    paths_config_path = resolve_cli_path(args.paths_config, DEFAULT_PATHS_CONFIG, repo_root)
    inference_config_path = resolve_cli_path(args.inference_config, DEFAULT_INFERENCE_CONFIG, repo_root)
    class_map_path = resolve_cli_path(args.class_map_config, DEFAULT_CLASS_MAP_CONFIG, repo_root)
    alpha_list_path = resolve_cli_path(args.alpha_list_config, DEFAULT_ALPHA_LIST_CONFIG, repo_root)

    paths_cfg = load_paths_config(paths_config_path, repo_root=repo_root)
    inf_cfg = load_inference_config(inference_config_path)

    if args.command == "prepare":
        cmd_prepare(args=args, paths_cfg=paths_cfg, inf_cfg=inf_cfg)
    elif args.command == "generate":
        class_map = load_class_map(class_map_path)
        alphas = load_alpha_list(alpha_list_path)
        cmd_generate(
            args=args,
            paths_cfg=paths_cfg,
            inf_cfg=inf_cfg,
            class_map=class_map,
            alphas=alphas,
        )
    elif args.command == "finalize":
        cmd_finalize(args=args, paths_cfg=paths_cfg)
    elif args.command == "build-train-data":
        cmd_build_train_data(args=args, paths_cfg=paths_cfg, inf_cfg=inf_cfg)
    elif args.command == "panels":
        cmd_panels(args=args, paths_cfg=paths_cfg, inf_cfg=inf_cfg)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
