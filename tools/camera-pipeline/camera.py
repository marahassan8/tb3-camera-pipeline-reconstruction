"""Hidden camera pipeline (reference, float64).

RAW input: a 16-bit RGGB Bayer mosaic plus metadata (black level, white level,
as-shot white balance). Every stage is deterministic and closed-form. `P` holds
the camera's parameters; controls pass a modified copy.
"""

import json

import numpy as np
from PIL import Image
from scipy import ndimage

P = {
    "wb_tweak": (1.03, 1.0, 0.96),
    "demosaic": "malvar",
    "hl_start": 0.80,
    "ccm": ((1.62, -0.48, -0.14),
            (-0.20, 1.42, -0.22),
            (0.04, -0.46, 1.42)),
    "ltm_radius": 8,
    "ltm_eps": 0.04,
    "ltm_compress": 0.70,
    "ltm_anchor": float(np.log2(0.18)),
    "gain": 1.6,
    "white": 4.0,
    "sharp_sigma": 1.0,
    "sharp_amount": 0.6,
    "sharp_threshold": 0.01,
}

LUMA = np.array([0.2126, 0.7152, 0.0722])

# Malvar, He and Cutler (2004), 5x5 kernels scaled by 1/8.
_G_AT_RB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                     [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]) / 8
_HORIZ = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                   [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]]) / 8
_VERT = _HORIZ.T
_DIAG = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                  [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]) / 8


def load_raw(path_png, path_json):
    mosaic = np.asarray(Image.open(path_png), dtype=np.float64)
    meta = json.load(open(path_json))
    return mosaic, meta


def masks(h, w):
    y, x = np.mgrid[0:h, 0:w]
    r = (y % 2 == 0) & (x % 2 == 0)
    b = (y % 2 == 1) & (x % 2 == 1)
    g_rrow = (y % 2 == 0) & (x % 2 == 1)
    g_brow = (y % 2 == 1) & (x % 2 == 0)
    return r, g_rrow, g_brow, b


def demosaic_malvar(m):
    h, w = m.shape
    r, g_rrow, g_brow, b = masks(h, w)
    f = lambda k: ndimage.correlate(m, k, mode="mirror")
    g_rb, hz, vt, dg = f(_G_AT_RB), f(_HORIZ), f(_VERT), f(_DIAG)
    R = np.where(r, m, np.where(g_rrow, hz, np.where(g_brow, vt, dg)))
    G = np.where(r | b, g_rb, m)
    B = np.where(b, m, np.where(g_brow, hz, np.where(g_rrow, vt, dg)))
    return np.stack([R, G, B], -1)


def demosaic_bilinear(m):
    h, w = m.shape
    r, g_rrow, g_brow, b = masks(h, w)
    kg = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
    krb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    G = ndimage.correlate(m * (g_rrow | g_brow), kg, mode="mirror")
    R = ndimage.correlate(m * r, krb, mode="mirror")
    B = ndimage.correlate(m * b, krb, mode="mirror")
    return np.stack([R, G, B], -1)


def box(x, r):
    return ndimage.uniform_filter(x, size=2 * r + 1, mode="reflect")


def guided_self(I, r, eps):
    mean = box(I, r)
    var = box(I * I, r) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    return box(a, r) * I + box(b, r)


def srgb_oetf(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def process(mosaic, meta, p=P):
    x = (mosaic - meta["black_level"]) / (meta["white_level"] - meta["black_level"])
    x = np.maximum(x, 0)

    # White balance on the mosaic: as-shot gains times the camera's tweak.
    gains = np.array(meta["as_shot_wb"]) * np.array(p["wb_tweak"])
    r, g_rrow, g_brow, b = masks(*x.shape)
    x = x * np.where(r, gains[0], np.where(b, gains[2], gains[1]))

    rgb = demosaic_malvar(x) if p["demosaic"] == "malvar" else demosaic_bilinear(x)

    # Highlights: clip every channel where the first one saturates, then blend
    # towards neutral as the brightest channel approaches that level.
    if p["hl_start"] is not None:
        clip = gains.min()
        rgb = np.clip(rgb, 0, clip) / clip
        w = smoothstep(p["hl_start"], 1.0, rgb.max(-1))[..., None]
        rgb = rgb * (1 - w) + rgb.mean(-1, keepdims=True) * w
    else:
        rgb = np.clip(rgb, 0, None) / gains.min()

    rgb = np.maximum(rgb @ np.array(p["ccm"]).T, 0)

    # Local tone mapping: compress the edge-aware base layer of log luminance
    # around a mid-grey anchor and keep the detail layer.
    if p["ltm_compress"] != 1.0:
        Y = rgb @ LUMA
        l = np.log2(Y + 1e-4)
        base = guided_self(l, p["ltm_radius"], p["ltm_eps"])
        a = p["ltm_anchor"]
        out = a + p["ltm_compress"] * (base - a) + (l - base)
        rgb = rgb * np.power(2.0, out - l)[..., None]

    # Global tone curve (extended Reinhard), then the sRGB transfer function.
    v = p["gain"] * rgb
    v = v * (1 + v / p["white"] ** 2) / (1 + v)
    v = srgb_oetf(np.clip(v, 0, 1))

    # Unsharp mask on luma with a soft threshold.
    if p["sharp_amount"]:
        y = v @ LUMA
        d = y - ndimage.gaussian_filter(y, p["sharp_sigma"], mode="reflect", truncate=4.0)
        d = np.sign(d) * np.maximum(np.abs(d) - p["sharp_threshold"], 0)
        v = v + p["sharp_amount"] * d[..., None]

    return np.round(np.clip(v, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    import sys
    m, meta = load_raw(sys.argv[1], sys.argv[2])
    Image.fromarray(process(m, meta)).save(sys.argv[3])
