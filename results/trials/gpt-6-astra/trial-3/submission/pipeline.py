#!/usr/bin/env python3
"""Render RGGB RAW PNGs with the adjacent, scene-independent camera profile.

Usage: python3 pipeline.py IN.raw.png IN.json OUT.png
Only profile.npz and the two input files are read at runtime.
"""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
from scipy.interpolate import BSpline
from scipy.ndimage import convolve, gaussian_filter
from scipy.sparse import csr_matrix

PROFILE_FILE = "profile.npz"
COLOR_CHUNK_SIZE = 49152


def normalize_raw(raw, metadata):
    """Subtract black, normalize white, and apply mosaic white balance."""
    black = float(metadata["black_level"])
    white = float(metadata["white_level"])
    gains = np.asarray(metadata["as_shot_wb"], dtype=np.float32)
    normalized = np.maximum((raw - black) / (white - black), 0)
    normalized[0::2, 0::2] *= gains[0]
    normalized[0::2, 1::2] *= gains[1]
    normalized[1::2, 0::2] *= gains[1]
    normalized[1::2, 1::2] *= gains[2]
    return normalized


def demosaic(raw):
    """Malvar interpolation in the white-balanced sensor space."""
    green_kernel = np.array([
        [0, 0, -1, 0, 0], [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]], dtype=float) / 8
    horizontal_kernel = np.array([
        [0, 0, .5, 0, 0], [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0],
        [0, 0, .5, 0, 0]], dtype=float) / 8
    diagonal_kernel = np.array([
        [0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0]], dtype=float) / 8
    rgb = np.empty((*raw.shape, 3), dtype=raw.dtype)
    rgb[..., 1] = convolve(raw, green_kernel, mode="mirror")
    rgb[0::2, 1::2, 1] = raw[0::2, 1::2]
    rgb[1::2, 0::2, 1] = raw[1::2, 0::2]
    horizontal = convolve(raw, horizontal_kernel, mode="mirror")
    vertical = convolve(raw, horizontal_kernel.T, mode="mirror")
    diagonal = convolve(raw, diagonal_kernel, mode="mirror")
    rgb[0::2, 0::2, 0] = raw[0::2, 0::2]
    rgb[0::2, 1::2, 0] = horizontal[0::2, 1::2]
    rgb[1::2, 0::2, 0] = vertical[1::2, 0::2]
    rgb[1::2, 1::2, 0] = diagonal[1::2, 1::2]
    rgb[1::2, 1::2, 2] = raw[1::2, 1::2]
    rgb[0::2, 1::2, 2] = vertical[0::2, 1::2]
    rgb[1::2, 0::2, 2] = horizontal[1::2, 0::2]
    rgb[0::2, 0::2, 2] = diagonal[0::2, 0::2]
    return rgb


def color_transform(rgb, profile):
    """Evaluate the smooth, scene-independent color calibration in chunks."""
    knots = profile["knots"]
    matrix = profile["mat"]
    coefficients = profile["coef"]
    n = len(knots) - 4
    flat = rgb.reshape(-1, 3)
    result = np.empty_like(flat)
    for start in range(0, len(flat), COLOR_CHUNK_SIZE):
        stop = min(start + COLOR_CHUNK_SIZE, len(flat))
        # Keep spline arithmetic in double precision while bounding its memory.
        z = flat[start:stop] @ matrix
        u = np.clip(np.sign(z) * np.abs(z) ** .4, knots[0], knots[-1])
        bases = [BSpline.design_matrix(u[:, c], knots, 3).tocsr()
                 for c in range(3)]
        values = [b.data.reshape(-1, 4) for b in bases]
        indices = [b.indices.reshape(-1, 4) for b in bases]
        weights = (values[0][:, :, None, None]
                   * values[1][:, None, :, None]
                   * values[2][:, None, None, :])
        columns = (indices[0][:, :, None, None] * n * n
                   + indices[1][:, None, :, None] * n
                   + indices[2][:, None, None, :])
        design = csr_matrix((weights.reshape(-1), columns.reshape(-1),
                             np.arange(stop - start + 1) * 64),
                            shape=(stop - start, n ** 3))
        result[start:stop] = design @ coefficients
    return result.reshape(rgb.shape)


def render(raw_path, metadata_path, output_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with Image.open(raw_path) as image:
        raw = np.asarray(image, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("The input must be a single-channel Bayer mosaic.")
    if metadata.get("cfa", "RGGB").upper() != "RGGB":
        raise ValueError("This camera profile expects an RGGB mosaic.")
    raw = normalize_raw(raw, metadata)
    with np.load(Path(__file__).with_name(PROFILE_FILE)) as profile:
        linear = demosaic(raw)
        del raw
        if int(profile.get("clipmode", 0)) == 2:
            np.clip(linear, 0, 1, out=linear)
        rgb = color_transform(linear, profile)
        del linear
        parameters = profile["sp"].astype(np.float32)
        sigma, amount, threshold, red_weight, green_weight = parameters[:5]
        power = parameters[5:8]
        small_amount, small_sigma, red_detail, blue_detail = parameters[8:]
        detail_direction = np.array([red_detail, 1., blue_detail], dtype=np.float32)

    work = rgb
    work += .5 / 255
    np.maximum(work, 0, out=work)
    np.power(work, power, out=work)
    luminance = (work[..., 0] * red_weight + work[..., 1] * green_weight
                 + work[..., 2] * (1 - red_weight - green_weight))
    detail = luminance - gaussian_filter(luminance, sigma)
    strong_detail = np.maximum(np.abs(detail) - threshold, 0)
    strong_detail *= np.sign(detail)
    small_detail = luminance - gaussian_filter(luminance, small_sigma)
    delta = small_amount * small_detail + amount * strong_detail
    np.add(work, delta[..., None] * detail_direction, out=work)
    del luminance, detail, strong_detail, small_detail, delta
    np.maximum(work, 0, out=work)
    np.power(work, 1 / power, out=work)
    work -= .5 / 255
    work *= 255
    np.rint(work, out=work)
    np.clip(work, 0, 255, out=work)
    output = work.astype(np.uint8)
    Image.fromarray(output).save(output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: pipeline.py IN.raw.png IN.json OUT.png")
    render(*sys.argv[1:])
