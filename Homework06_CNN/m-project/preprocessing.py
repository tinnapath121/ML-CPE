import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image to img_size x img_size RGB. None if unusable."""

    if image is None or image.size == 0:
        return None

    # cv2 reads BGR, convert to RGB so images display correctly
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image (INTER_AREA is the right filter for shrinking)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):

    return np.ascontiguousarray(images, dtype=np.uint8)


def compute_norm_stats(X_train, chunk=512):
    """Per-channel mean and std of the TRAINING set (0-255 scale).

    Standardization is z = (x - mean) / std. The statistics come from the
    training set only, so no information from validation/test data leaks in.
    Computed in chunks to avoid making a float64 copy of the whole array.
    """

    channels = X_train.shape[-1]
    total = np.zeros(channels)
    total_sq = np.zeros(channels)
    count = 0

    for i in range(0, len(X_train), chunk):
        batch = X_train[i:i + chunk].astype(np.float64)
        total += batch.sum(axis=(0, 1, 2))
        total_sq += (batch ** 2).sum(axis=(0, 1, 2))
        count += batch.shape[0] * batch.shape[1] * batch.shape[2]

    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))

    return {"mean": mean.tolist(), "std": std.tolist()}
