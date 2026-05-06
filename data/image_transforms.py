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
        "copies_per_image": int(cfg.get("copies_per_image", 0)),
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

    pil_img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")

    if aug_cfg["rotation_degrees"] > 0:
        angle = rng.uniform(
            -aug_cfg["rotation_degrees"],
            aug_cfg["rotation_degrees"],
        )
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    if aug_cfg["translate_pixels"] > 0:
        dx = int(rng.integers(-aug_cfg["translate_pixels"], aug_cfg["translate_pixels"] + 1))
        dy = int(rng.integers(-aug_cfg["translate_pixels"], aug_cfg["translate_pixels"] + 1))

        shifted = Image.new("RGB", pil_img.size, color=(0, 0, 0))
        shifted.paste(pil_img, (dx, dy))
        pil_img = shifted

    arr = np.array(pil_img, dtype=np.float32) / 255.0

    if aug_cfg["brightness_delta"] > 0:
        factor = rng.uniform(
            1.0 - aug_cfg["brightness_delta"],
            1.0 + aug_cfg["brightness_delta"],
        )
        arr = arr * factor

    if aug_cfg["noise_std"] > 0:
        arr = arr + rng.normal(0, aug_cfg["noise_std"], size=arr.shape)

    return np.clip(arr, 0.0, 1.0)


def augment_flattened_rgb_training_data(
    X: np.ndarray,
    y: np.ndarray,
    input_shape: list[int],
    aug_cfg: dict,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not aug_cfg["enabled"] or aug_cfg["copies_per_image"] <= 0:
        return X, y

    rng = np.random.default_rng(random_state)
    height, width, channels = input_shape

    X_aug = [X]
    y_aug = [y]

    for _ in range(aug_cfg["copies_per_image"]):
        print("Aug started")
        augmented = []

        for row in X:
            arr = row.reshape(height, width, channels)
            arr_aug = augment_rgb_array(arr, aug_cfg, rng)
            augmented.append(arr_aug.flatten())

        X_aug.append(np.vstack(augmented))
        y_aug.append(y.copy())
        print("Aug image fin")

    return np.vstack(X_aug), np.concatenate(y_aug)