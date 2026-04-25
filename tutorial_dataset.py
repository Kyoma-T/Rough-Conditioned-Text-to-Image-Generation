import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self,
        json_path='data/data.json',
        auto_mask_mode: str = 'none',
        diff_threshold: int = 32,
        diff_blur_kernel: int = 5,
    ):
        self.data = []
        json_path = Path(json_path)
        with json_path.open('rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

        self.ref_num = 1
        self.json_root = json_path.parent.resolve()

        self.auto_mask_mode = str(auto_mask_mode).strip().lower()
        if self.auto_mask_mode not in {'none', 'diff_proxy'}:
            raise ValueError(f"Unsupported auto_mask_mode: {auto_mask_mode}")
        self.diff_threshold = int(diff_threshold)
        self.diff_blur_kernel = max(0, int(diff_blur_kernel))

        self.has_m_conflict = any('m_conflict' in item for item in self.data)
        self.has_m_bg = any('m_bg' in item for item in self.data)

        if self.has_m_conflict and not all('m_conflict' in item for item in self.data):
            raise ValueError("Inconsistent manifest: some rows miss 'm_conflict'.")
        if self.has_m_bg and not all('m_bg' in item for item in self.data):
            raise ValueError("Inconsistent manifest: some rows miss 'm_bg'.")
        if self.has_m_conflict != self.has_m_bg:
            raise ValueError("Manifest must provide both 'm_conflict' and 'm_bg' together.")

        # Fallback for manifests like train1.json without explicit masks:
        # derive heuristic proxy masks from (source image panel vs paired image panel).
        self.use_proxy_masks = (not self.has_m_conflict and self.auto_mask_mode == 'diff_proxy')

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        candidates = [
            p,  # relative to current working directory (legacy behavior)
            self.json_root / p,  # relative to manifest directory
            self.json_root.parent / p,  # relative to train root when manifest is in train/data
        ]
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        # Keep deterministic fallback for clear error messages.
        return (self.json_root / p).resolve()

    def _read_rgb(self, path_str: str) -> np.ndarray:
        path = self._resolve_path(path_str)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _split_left_half(self, img: np.ndarray) -> np.ndarray:
        width = img.shape[1]
        mid = width // 2
        return img[:, :mid, :]

    def _split_right_half(self, img: np.ndarray) -> np.ndarray:
        width = img.shape[1]
        mid = width // 2
        return img[:, mid:, :]

    def _read_optional_mask(
        self,
        path_str: Optional[str],
        target_h: int,
        target_w: int,
    ) -> Optional[np.ndarray]:
        if not path_str:
            return None
        path = self._resolve_path(str(path_str))
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")

        # Support both panel masks (2W) and single-side masks (W).
        if mask.shape[1] == target_w * 2:
            mask = mask[:, target_w:]
        if mask.shape[0] != target_h or mask.shape[1] != target_w:
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        return np.clip(mask, 0.0, 1.0)

    def _build_diff_proxy_masks(self, source_img_rgb: np.ndarray, paired_img_rgb: np.ndarray):
        src_gray = cv2.cvtColor(source_img_rgb, cv2.COLOR_RGB2GRAY)
        paired_gray = cv2.cvtColor(paired_img_rgb, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(src_gray, paired_gray)

        if self.diff_blur_kernel > 1:
            k = self.diff_blur_kernel if self.diff_blur_kernel % 2 == 1 else self.diff_blur_kernel + 1
            diff = cv2.GaussianBlur(diff, (k, k), 0)

        m_conflict = (diff >= self.diff_threshold).astype(np.float32)
        m_bg = np.where(m_conflict > 0.5, 0.0, 1.0).astype(np.float32)
        return m_conflict, m_bg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_rgb = self._read_rgb(source_filename)
        target_rgb = self._read_rgb(target_filename)

        # Pair-panel convention:
        # source uses the left condition panel, target uses the right image panel.
        source = self._split_left_half(source_rgb)
        source_img = self._split_left_half(target_rgb)
        paired_img = self._split_right_half(target_rgb)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        target = (paired_img.astype(np.float32) / 127.5) - 1.0

        output = dict(jpg=target, hint=source, txt=prompt)

        target_h, target_w = target.shape[:2]
        if self.has_m_conflict:
            m_conflict = self._read_optional_mask(item.get('m_conflict'), target_h, target_w)
            if m_conflict is None:
                raise ValueError("Manifest declares m_conflict but current sample has empty value.")
            output['m_conflict'] = m_conflict[..., None]

        if self.has_m_bg:
            m_bg = self._read_optional_mask(item.get('m_bg'), target_h, target_w)
            if m_bg is None:
                raise ValueError("Manifest declares m_bg but current sample has empty value.")
            output['m_bg'] = m_bg[..., None]

        if self.use_proxy_masks:
            m_conflict, m_bg = self._build_diff_proxy_masks(source_img_rgb=source_img, paired_img_rgb=paired_img)
            output['m_conflict'] = m_conflict[..., None]
            output['m_bg'] = m_bg[..., None]

        return output
