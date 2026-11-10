import numpy as np
from scipy import ndimage as ndi


def _rng(seed):
    return np.random.default_rng(seed)


def _diameter(mask):
    return 2.0 * np.sqrt(mask.sum() / np.pi)


def _signed_distance(mask):
    outside = ndi.distance_transform_edt(~mask)
    inside = ndi.distance_transform_edt(mask)
    return outside - inside


def shift_speckle(image, strength, seed=0):
    noise = _rng(seed).normal(0.0, 0.4 * strength, image.shape)
    return np.clip(image + image * noise, 0.0, 1.0)


def shift_blur(image, strength, seed=0):
    return ndi.gaussian_filter(image, sigma=4.0 * strength)


def shift_nonlinear_contrast(image, strength, seed=0):
    gamma = 1.0 + 1.5 * strength
    return np.clip(image, 0.0, 1.0) ** gamma


IMAGE_SHIFTS = {
    "speckle": shift_speckle,
    "blur": shift_blur,
    "nonlinear_contrast": shift_nonlinear_contrast,
}


def shift_dilation(mask, strength, seed=0):
    radius = 12.0 * strength
    return _signed_distance(mask) <= radius


def shift_erosion(mask, strength, seed=0):
    radius = 12.0 * strength
    return _signed_distance(mask) <= -radius


def shift_smoothing(mask, strength, seed=0):
    sigma = 10.0 * strength
    if sigma == 0:
        return mask.copy()
    return ndi.gaussian_filter(mask.astype(float), sigma=sigma) > 0.5


def shift_edge_offset(mask, strength, seed=0):
    offset = 0.15 * strength * _diameter(mask)
    return _signed_distance(mask) <= offset


def shift_placebo_jitter(mask, strength, seed=0):
    field = ndi.gaussian_filter(_rng(seed).normal(size=mask.shape), sigma=12.0)
    field = field / (field.std() + 1e-8)
    amplitude = 0.08 * strength * _diameter(mask)
    return _signed_distance(mask) <= amplitude * field


MASK_SHIFTS = {
    "dilation": shift_dilation,
    "erosion": shift_erosion,
    "smoothing": shift_smoothing,
    "edge_offset": shift_edge_offset,
    "placebo_jitter": shift_placebo_jitter,
}


def apply_shift(image, mask, image_shift=None, mask_shift=None, strength=0.5, seed=0):
    mask = mask.astype(bool)
    if image_shift is not None:
        image = IMAGE_SHIFTS[image_shift](image, strength, seed)
    if mask_shift is not None and mask.any():
        mask = MASK_SHIFTS[mask_shift](mask, strength, seed)
    return image, mask


def shift_dataset(images, masks, image_shift=None, mask_shift=None, strength=0.5, seed=0):
    out_images, out_masks = [], []
    for i, (image, mask) in enumerate(zip(images, masks)):
        shifted = apply_shift(image, mask, image_shift, mask_shift, strength, seed + i)
        out_images.append(shifted[0])
        out_masks.append(shifted[1])
    return out_images, out_masks


def predictions_rejected(predictions, max_bad_fraction=0.20, full_fraction=0.90):
    bad = 0
    for prediction in predictions:
        area = prediction.astype(bool).mean()
        if area == 0.0 or area >= full_fraction:
            bad += 1
    return bad / len(predictions) > max_bad_fraction


def known_mix(drop_mask_only, drop_image_only):
    total = drop_mask_only + drop_image_only
    if total == 0:
        return 0.0
    return drop_mask_only / total


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return 2.0 * (a & b).sum() / total
