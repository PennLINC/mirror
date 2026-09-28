"""Small, dependency-light helpers shared by image-based reports and figures."""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def first_glob(patterns: Iterable[str | Path]) -> Path | None:
    """Return the lexicographically first unique match across glob patterns."""

    matches = {Path(match) for pattern in patterns for match in glob(str(pattern))}
    return min(matches) if matches else None


def load_canonical(path: Path) -> nib.spatialimages.SpatialImage:
    """Load a NIfTI image and return its closest canonical orientation."""

    return nib.as_closest_canonical(nib.load(str(path)))


def resample_image(
    image: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    order: int,
) -> nib.spatialimages.SpatialImage:
    """Resample *image* to *reference* unless their grids already match."""

    if image.shape[:3] == reference.shape[:3] and np.allclose(
        image.affine,
        reference.affine,
        atol=1e-4,
    ):
        return image
    return resample_from_to(image, reference, order=order)
