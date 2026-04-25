#!/usr/bin/env python
"""Read-only audit for SmartControl training implementation.

This script does not modify training logic. It prints:
1) parameter/trainable statistics
2) optimizer-covered parameter statistics
3) module-level parameter breakdown
4) keyword audit for loss/data assumptions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def _numel(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in params)


def _collect_optimizer_param_ids(optimizer_obj) -> Set[int]:
    """Collect parameter ids from torch optimizer or PL optimizer wrappers."""
    ids: Set[int] = set()

    def _collect_from_optimizer(opt) -> None:
        if not hasattr(opt, "param_groups"):
            return
        for group in opt.param_groups:
            for p in group.get("params", []):
                ids.add(id(p))

    if isinstance(optimizer_obj, torch.optim.Optimizer):
        _collect_from_optimizer(optimizer_obj)
        return ids

    if isinstance(optimizer_obj, tuple):
        # Some Lightning code returns (optimizers, schedulers)
        optimizer_obj = optimizer_obj[0]

    if isinstance(optimizer_obj, list):
        for item in optimizer_obj:
            if isinstance(item, dict) and "optimizer" in item:
                _collect_from_optimizer(item["optimizer"])
            else:
                _collect_from_optimizer(item)
        return ids

    _collect_from_optimizer(optimizer_obj)
    return ids


def _module_bucket(name: str) -> str:
    if name.startswith("model."):
        return "model(diffusion_wrapper)"
    if name.startswith("control_model."):
        return "control_model"
    if name.startswith("c_pre_list."):
        return "c_pre_list(CSP)"
    if name.startswith("first_stage_model."):
        return "first_stage_model(VAE)"
    if name.startswith("cond_stage_model."):
        return "cond_stage_model(CLIP)"
    return "others"


def _print_param_audit(model: torch.nn.Module, topn: int = 200) -> None:
    named_params: List[Tuple[str, torch.nn.Parameter]] = list(model.named_parameters())
    all_params = [p for _, p in named_params]
    trainable_named = [(n, p) for n, p in named_params if p.requires_grad]
    trainable_params = [p for _, p in trainable_named]

    print("==== PARAMETER AUDIT ====")
    print(f"total params:      {_numel(all_params):,}")
    print(f"trainable params:  {_numel(trainable_params):,}")
    if _numel(all_params) > 0:
        ratio = _numel(trainable_params) / _numel(all_params)
    else:
        ratio = 0.0
    print(f"trainable ratio:   {ratio:.4%}")

    bucket_total: Dict[str, int] = {}
    bucket_trainable: Dict[str, int] = {}
    for n, p in named_params:
        b = _module_bucket(n)
        bucket_total[b] = bucket_total.get(b, 0) + int(p.numel())
        if p.requires_grad:
            bucket_trainable[b] = bucket_trainable.get(b, 0) + int(p.numel())

    print("\n-- by module bucket --")
    for b in sorted(bucket_total.keys()):
        t = bucket_total[b]
        tr = bucket_trainable.get(b, 0)
        print(f"{b:28s} total={t:,} trainable={tr:,}")

    print(f"\n-- trainable parameter names (top {topn}) --")
    for n, _ in trainable_named[:topn]:
        print(n)


def _print_optimizer_coverage(model: torch.nn.Module) -> None:
    print("\n==== OPTIMIZER COVERAGE ====")
    optimizer_obj = model.configure_optimizers()
    opt_param_ids = _collect_optimizer_param_ids(optimizer_obj)

    named_params: List[Tuple[str, torch.nn.Parameter]] = list(model.named_parameters())
    in_opt = [(n, p) for n, p in named_params if id(p) in opt_param_ids]
    req_grad_not_in_opt = [
        (n, p) for n, p in named_params if p.requires_grad and id(p) not in opt_param_ids
    ]
    in_opt_not_req_grad = [
        (n, p) for n, p in named_params if (not p.requires_grad) and id(p) in opt_param_ids
    ]

    print(f"optimizer params (numel): {_numel([p for _, p in in_opt]):,}")
    print(f"requires_grad but NOT in optimizer (numel): {_numel([p for _, p in req_grad_not_in_opt]):,}")
    print(f"in optimizer but requires_grad=False (numel): {_numel([p for _, p in in_opt_not_req_grad]):,}")

    print("\n-- optimizer parameter names (top 200) --")
    for n, _ in in_opt[:200]:
        print(n)

    print("\n-- requires_grad but NOT in optimizer (top 200) --")
    for n, _ in req_grad_not_in_opt[:200]:
        print(n)


def _grep_keywords(path: Path, keywords: List[str]) -> None:
    print(f"\n==== KEYWORD AUDIT: {path} ====")
    if not path.exists():
        print("file not found")
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    for kw in keywords:
        found = False
        for i, line in enumerate(lines, start=1):
            if kw in line:
                print(f"[FOUND] {kw} at line {i}: {line.strip()}")
                found = True
        if not found:
            print(f"[MISS ] {kw}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SmartControl training consistency.")
    parser.add_argument("--config", type=Path, default=Path("models/cldm_v15.yaml"))
    parser.add_argument(
        "--clip-version",
        type=str,
        default="",
        help="Optional override for cond_stage_config.params.version.",
    )
    parser.add_argument(
        "--skip-cond-stage",
        action="store_true",
        default=False,
        help="Set cond_stage_config='__is_unconditional__' to avoid loading CLIP.",
    )
    parser.add_argument("--sd-locked", action="store_true", default=True)
    parser.add_argument("--no-sd-locked", action="store_false", dest="sd_locked")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-trainable-names", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(str(args.config))

    if args.skip_cond_stage:
        cfg.model.params["cond_stage_config"] = "__is_unconditional__"

    if args.clip_version:
        if "params" not in cfg.model.params.cond_stage_config:
            cfg.model.params.cond_stage_config["params"] = {}
        cfg.model.params.cond_stage_config.params["version"] = args.clip_version

    model = instantiate_from_config(cfg.model).cpu()
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked

    _print_param_audit(model, topn=args.max_trainable_names)
    _print_optimizer_coverage(model)

    _grep_keywords(
        Path("ldm/models/diffusion/ddpm.py"),
        ["L_c", "m_conflict", "m_bg", "lambda_c", "alpha_conflict", "alpha_bg", "loss_simple", "loss_vlb"],
    )
    _grep_keywords(
        Path("tutorial_dataset.py"),
        ["m_conflict", "m_bg", "source", "target", "prompt", "cv2.imread", "self.data.append"],
    )
    _grep_keywords(
        Path("cldm/cldm.py"),
        ["c_pre_list", "torch.sigmoid", "mode='c_ada'", "configure_optimizers", "sd_locked", "output_blocks"],
    )


if __name__ == "__main__":
    main()
