"""Independent implementation of the same pipeline, for calibration.

float32 throughout, hand-written filters, edge-replicate borders instead of
mirror/reflect. A correct reconstruction differs from the reference in about
these ways, so the grading tolerance must absorb them.
"""

import numpy as np

import camera

f32 = np.float32


def pad(x, r):
    return np.pad(x, ((r, r), (r, r)) + ((0, 0),) * (x.ndim - 2), mode="edge")


def corr(x, k):
    r = k.shape[0] // 2
    xp = pad(x, r)
    h, w = x.shape
    out = np.zeros_like(x)
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if k[i, j]:
                out += f32(k[i, j]) * xp[i:i + h, j:j + w]
    return out


def demosaic(m):
    r, g_rrow, g_brow, b = camera.masks(*m.shape)
    g_rb, hz, vt, dg = (corr(m, k) for k in (camera._G_AT_RB, camera._HORIZ, camera._VERT, camera._DIAG))
    R = np.where(r, m, np.where(g_rrow, hz, np.where(g_brow, vt, dg)))
    G = np.where(r | b, g_rb, m)
    B = np.where(b, m, np.where(g_brow, hz, np.where(g_rrow, vt, dg)))
    return np.stack([R, G, B], -1)


def box(x, r):
    xp = pad(x, r).astype(np.float64)
    c = np.cumsum(np.cumsum(np.pad(xp, ((1, 0), (1, 0))), 0), 1)
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return (s / (n * n)).astype(f32)


def gauss(x, sigma):
    r = int(4 * sigma + 0.5)
    t = np.arange(-r, r + 1, dtype=f32)
    k = np.exp(-t * t / (2 * sigma * sigma))
    k /= k.sum()
    h, w = x.shape
    xp = pad(x, r)
    y = sum(k[i] * xp[:, i:i + w] for i in range(2 * r + 1))
    return sum(k[i] * y[i:i + h, :] for i in range(2 * r + 1))


def process(mosaic, meta, p=camera.P):
    x = (mosaic.astype(f32) - meta["black_level"]) / f32(meta["white_level"] - meta["black_level"])
    x = np.maximum(x, 0)
    gains = (np.array(meta["as_shot_wb"]) * np.array(p["wb_tweak"])).astype(f32)
    r, g1, g2, b = camera.masks(*x.shape)
    x = x * np.where(r, gains[0], np.where(b, gains[2], gains[1])).astype(f32)
    rgb = demosaic(x)
    clip = gains.min()
    rgb = np.clip(rgb, 0, clip) / clip
    t = np.clip((rgb.max(-1) - p["hl_start"]) / (1 - p["hl_start"]), 0, 1)
    w = (t * t * (3 - 2 * t))[..., None]
    rgb = rgb + w * (rgb.mean(-1, keepdims=True) - rgb)
    rgb = np.maximum(np.einsum("hwc,kc->hwk", rgb, np.array(p["ccm"], f32)), 0)
    Y = rgb @ camera.LUMA.astype(f32)
    l = np.log2(Y + f32(1e-4))
    rr, eps = p["ltm_radius"], p["ltm_eps"]
    mean = box(l, rr)
    var = box(l * l, rr) - mean * mean
    a = var / (var + eps)
    bb = mean - a * mean
    base = box(a, rr) * l + box(bb, rr)
    an = p["ltm_anchor"]
    rgb = rgb * np.exp2((p["ltm_compress"] - 1) * (base - an))[..., None]
    v = f32(p["gain"]) * rgb
    v = v * (1 + v / f32(p["white"] ** 2)) / (1 + v)
    v = np.clip(v, 0, 1)
    v = np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** f32(1 / 2.4) - 0.055)
    y = v @ camera.LUMA.astype(f32)
    d = y - gauss(y, p["sharp_sigma"])
    d = np.sign(d) * np.maximum(np.abs(d) - f32(p["sharp_threshold"]), 0)
    v = v + f32(p["sharp_amount"]) * d[..., None]
    return np.round(np.clip(v, 0, 1) * 255).astype(np.uint8)
