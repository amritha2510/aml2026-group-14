from typing import Any

import numpy as np
from PIL import Image


def get_image_aug_config(config: dict, model_key: str | None = None) -> dict:
    global_cfg = config.get("data_augmentation", {})
    model_cfg = {}

    if model_key is not None:
        model_cfg = config.get(model_key, {}).get("data_augmentation", {})

    cfg = {**global_cfg, **model_cfg}

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "probability": float(cfg.get("probability", 0.0)),
        "rotation_degrees": float(cfg.get("rotation_degrees", 0)),
        "translate_pixels": int(cfg.get("translate_pixels", 0)),
        "noise_std": float(cfg.get("noise_std", 0)),
        "brightness_delta": float(cfg.get("brightness_delta", 0)),
    }


def load_image_as_rgb_array(path: str, normalize: bool = True) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)

    if normalize:
        arr = arr / 255.0

    return arr


def augment_rgb_array(
    arr: np.ndarray,
    aug_cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB array with shape HxWx3, got {arr.shape}")

    arr = np.clip(arr, 0.0, 1.0)

    pil_img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")

    if aug_cfg["rotation_degrees"] > 0:
        angle = rng.uniform(
            -aug_cfg["rotation_degrees"],
            aug_cfg["rotation_degrees"],
        )

        pil_img = pil_img.rotate(
            angle,
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
        )

    if aug_cfg["translate_pixels"] > 0:
        dx = int(
            rng.integers(
                -aug_cfg["translate_pixels"],
                aug_cfg["translate_pixels"] + 1,
            )
        )
        dy = int(
            rng.integers(
                -aug_cfg["translate_pixels"],
                aug_cfg["translate_pixels"] + 1,
            )
        )

        pil_img = pil_img.transform(
            pil_img.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
        )

    arr = np.array(pil_img, dtype=np.float32) / 255.0

    if aug_cfg["brightness_delta"] > 0:
        factor = rng.uniform(
            1.0 - aug_cfg["brightness_delta"],
            1.0 + aug_cfg["brightness_delta"],
        )
        arr = arr * factor

    if aug_cfg["noise_std"] > 0:
        arr = arr + rng.normal(
            0,
            aug_cfg["noise_std"],
            size=arr.shape,
        )

    return np.clip(arr, 0.0, 1.0)


def make_deterministic_image_seed(
    img_idx: int,
    random_state: int,
) -> int:
    return int((img_idx + random_state * 1_000_003) % (2**32))


def augment_flattened_rgb_training_data(
    X: np.ndarray,
    y: np.ndarray,
    input_shape: list[int],
    aug_cfg: dict,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:

    if not aug_cfg["enabled"] or aug_cfg["probability"] <= 0:
        return X, y

    probability = float(aug_cfg["probability"])

    if probability > 1:
        raise ValueError(
            f"data_augmentation.probability must be between 0 and 1, got {probability}"
        )

    height, width, channels = input_shape

    X_out = []
    y_out = []

    n_augmented = 0

    for img_idx, (row, label) in enumerate(zip(X, y)):
        image_seed = make_deterministic_image_seed(
            img_idx=img_idx,
            random_state=random_state,
        )

        rng = np.random.default_rng(image_seed)

        arr = row.reshape(height, width, channels)

        should_augment = rng.random() < probability

        if should_augment:
            arr = augment_rgb_array(arr, aug_cfg, rng)
            n_augmented += 1

        X_out.append(arr.flatten())
        y_out.append(label)

    print(
        f"[INFO] Augmentation applied to {n_augmented}/{len(y)} "
        f"training images with probability={probability}"
    )

    return np.vstack(X_out), np.array(y_out, dtype=y.dtype)