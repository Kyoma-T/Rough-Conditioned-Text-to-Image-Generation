from __future__ import annotations

import re
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


# ==================================================
# Local paths: edit these first
# ==================================================
DATASET_DIR = Path(r"SmartControl\SmartControl\train\data\get_openimage\download_cache")
OUTPUT_DIR = Path(r"SmartControl\SmartControl\train\data\get_openimage\openimages")
IMAGE_SUBDIR = "Horse"

EXPORT_IMAGE_DIR = OUTPUT_DIR / IMAGE_SUBDIR

# ==================================================
# FiftyOne dataset behavior
# ==================================================
DATASET_NAME = IMAGE_SUBDIR
FORCE_RECREATE_DATASET = True
PERSISTENT = True
LAUNCH_APP = False

# ==================================================
# Keep this section close to your preferred style
# ==================================================
DATASET_ZOO_NAME = "open-images-v7"
SPLIT = "train"
LABEL_TYPES = ["detections"]
CLASSES = [IMAGE_SUBDIR]
#"Horse", "Tiger", "Zebra", "Camel"
MAX_SAMPLES = 150
SHUFFLE = True
SEED = 42
INCLUDE_ID = True
ONLY_MATCHING = False

# ==================================================
# Cache sanity check (avoid silently using truncated CSV files)
# ==================================================
AUTO_DELETE_PARTIAL_CACHE = True
MIN_TRAIN_IMAGE_IDS_CSV_MB = 10
MIN_TRAIN_DETECTIONS_CSV_MB = 500


def _to_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _find_same_slug_dataset_name(name: str) -> str | None:
    target_slug = _to_slug(name)
    for existing in fo.list_datasets():
        if _to_slug(existing) == target_slug:
            return existing
    return None


def _resolve_openimages_id(sample) -> str:
    # FiftyOne Sample is not a dict; use has_field/get_field
    for key in (
        "open_images_id",
        "open_images_v7_id",
        "open_images_v6_id",
        "image_id",
    ):
        value = sample.get_field(key) if sample.has_field(key) else None
        if value:
            return str(value)
    return Path(sample.filepath).stem


def _validate_or_cleanup_train_cache(download_dir: Path) -> None:
    if not AUTO_DELETE_PARTIAL_CACHE:
        return
    if SPLIT != "train":
        return

    train_dir = download_dir / DATASET_ZOO_NAME / "train"
    checks: list[tuple[Path, int, str]] = [
        (
            train_dir / "metadata" / "image_ids.csv",
            MIN_TRAIN_IMAGE_IDS_CSV_MB,
            "image_ids.csv",
        ),
    ]
    if "detections" in LABEL_TYPES:
        checks.append(
            (
                train_dir / "labels" / "detections.csv",
                MIN_TRAIN_DETECTIONS_CSV_MB,
                "detections.csv",
            )
        )

    for path, min_mb, name in checks:
        if not path.exists():
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < min_mb:
            print(
                f"[CacheCheck] {name} looks incomplete "
                f"({size_mb:.2f} MB < {min_mb} MB). "
                "Deleting it so FiftyOne can re-download."
            )
            path.unlink(missing_ok=True)


def _copy_selected_images(dataset) -> list[dict[str, str]]:
    EXPORT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    used_names: set[str] = set()
    for sample in dataset:
        src = Path(sample.filepath).resolve()
        if not src.exists():
            continue

        open_id = _resolve_openimages_id(sample)
        suffix = src.suffix if src.suffix else ".jpg"
        base_name = f"{open_id}{suffix}"
        dst_name = base_name
        idx = 1
        while dst_name in used_names:
            dst_name = f"{open_id}_{idx}{suffix}"
            idx += 1
        used_names.add(dst_name)

        dst = (EXPORT_IMAGE_DIR / dst_name).resolve()
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)

        rows.append(
            {
                "open_images_id": open_id,
                "source_filepath": str(src),
                "export_filepath": str(dst),
            }
        )
    return rows


def main() -> None:
    download_dir = DATASET_DIR.resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    # Compatibility: some FiftyOne versions for OpenImages internally pass
    # dataset_dir again, which conflicts if we also pass dataset_dir here.
    # Set global zoo cache dir instead of passing dataset_dir argument.
    fo.config.dataset_zoo_dir = str(download_dir)
    _validate_or_cleanup_train_cache(download_dir)

    same_slug_name = _find_same_slug_dataset_name(DATASET_NAME)
    if FORCE_RECREATE_DATASET and same_slug_name is not None:
        print(
            f"Deleting existing dataset '{same_slug_name}' "
            f"(slug conflict with '{DATASET_NAME}') before reload ..."
        )
        fo.delete_dataset(same_slug_name)

    dataset = foz.load_zoo_dataset(
        DATASET_ZOO_NAME,
        split=SPLIT,
        label_types=LABEL_TYPES,
        classes=CLASSES,
        max_samples=MAX_SAMPLES,
        shuffle=SHUFFLE,
        seed=SEED,
        include_id=INCLUDE_ID,
        only_matching=ONLY_MATCHING,
        dataset_name=DATASET_NAME,
        persistent=PERSISTENT,
    )

    print("=" * 80)
    print("Dataset loaded successfully")
    print(f"Dataset name: {dataset.name}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Download dir: {download_dir}")
    print("=" * 80)

    if len(dataset) == 0:
        raise RuntimeError(
            "Loaded dataset has 0 samples. "
            "Check CLASSES/SPLIT, or set FORCE_RECREATE_DATASET=True, "
            "or change DATASET_NAME."
        )

    rows = _copy_selected_images(dataset)
    print(f"Images copied to local folder: {EXPORT_IMAGE_DIR.resolve()}")
    print(f"Copied image count: {len(rows)}")

    print("\n[Preview exported filepaths]")
    for i, row in enumerate(rows[:5], 1):
        print(f"{i}. {row['export_filepath']} | open_images_id={row['open_images_id']}")

    if LAUNCH_APP:
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    main()
