#!/usr/bin/env python3
"""Camera RAW -> sRGB pipeline reproduction.

Usage: python3 pipeline.py IN.raw.png IN.json OUT.png

Stages (reverse-engineered from the training pairs; fitted constants in params.npz):
  1. black/white level normalisation, as-shot white balance, clip to [0, 1]
  2. Malvar-He-Cutler demosaic, clip to [0, 1]
  3. 3x3 colour matrix (camera RGB -> linear sRGB, includes a slight warm cast)
  4. highlight desaturation towards Rec.709 luminance, ramped on the
     (channel-weighted) max of camera RGB
  5. local tone mapping: per-pixel gain from log2 luminance,
     g = k * gaussian(logY, sigma) + beta * logY
  6. shared 1D tone curve (LUT over log2(value * gain)), round to 8 bit
"""
import sys, os, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
LW709 = np.array([0.2126, 0.7152, 0.0722])


def load_params():
    p = np.load(os.path.join(HERE, 'params.npz'))
    return {k: p[k] for k in p.files}


# Malvar-He-Cutler demosaic kernels
_K_G_RB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], float) / 8
_K_ROW = np.array([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], float) / 8
_K_COL = _K_ROW.T
_K_DIAG = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], float) / 8


def mhc_rggb(cfa):
    """Malvar-He-Cutler demosaic of an RGGB mosaic (memory-friendly)."""
    H, W = cfa.shape
    out = np.empty((H, W, 3), dtype=cfa.dtype)
    c = lambda k: ndi.correlate(cfa, k.astype(cfa.dtype), mode='reflect')
    # native samples
    out[..., 0] = 0; out[..., 1] = 0; out[..., 2] = 0
    out[0::2, 0::2, 0] = cfa[0::2, 0::2]
    out[1::2, 1::2, 2] = cfa[1::2, 1::2]
    out[0::2, 1::2, 1] = cfa[0::2, 1::2]
    out[1::2, 0::2, 1] = cfa[1::2, 0::2]
    t = c(_K_G_RB)
    out[0::2, 0::2, 1] = t[0::2, 0::2]; out[1::2, 1::2, 1] = t[1::2, 1::2]
    t = c(_K_ROW)   # G in R row -> R ; G in B row -> B
    out[0::2, 1::2, 0] = t[0::2, 1::2]; out[1::2, 0::2, 2] = t[1::2, 0::2]
    t = c(_K_COL)   # G in B row -> R ; G in R row -> B
    out[1::2, 0::2, 0] = t[1::2, 0::2]; out[0::2, 1::2, 2] = t[0::2, 1::2]
    t = c(_K_DIAG)  # B site -> R ; R site -> B
    out[1::2, 1::2, 0] = t[1::2, 1::2]; out[0::2, 0::2, 2] = t[0::2, 0::2]
    del t
    return out


def to_rggb(a, cfa):
    """Shift/flip mosaic so that it becomes RGGB; return (array, undo_fn)."""
    cfa = cfa.upper()
    if cfa == 'RGGB':
        return a, lambda o: o
    if cfa == 'GRBG':
        return a[:, 1:], lambda o: np.pad(o, ((0, 0), (1, 0), (0, 0)), mode='edge')
    if cfa == 'GBRG':
        return a[1:, :], lambda o: np.pad(o, ((1, 0), (0, 0), (0, 0)), mode='edge')
    if cfa == 'BGGR':
        return a[1:, 1:], lambda o: np.pad(o, ((1, 0), (1, 0), (0, 0)), mode='edge')
    return a, lambda o: o


def process(raw, meta, P):
    bl = float(meta.get('black_level', 0)); wl = float(meta.get('white_level', 65535))
    wb = meta.get('as_shot_wb', [1.0, 1.0, 1.0])
    cfa = meta.get('cfa', 'RGGB')
    x = ((raw.astype(np.float32) - np.float32(bl)) / np.float32(wl - bl)).astype(np.float32)
    x, undo = to_rggb(x, cfa)
    g = np.empty_like(x)
    g[0::2, 0::2] = wb[0]; g[1::2, 1::2] = wb[2 if len(wb) == 3 else 3]
    g[0::2, 1::2] = wb[1]; g[1::2, 0::2] = wb[1]
    x = np.clip(x * g, 0.0, 1.0).astype(np.float32)
    del g
    d = np.clip(mhc_rggb(x), 0.0, 1.0)
    del x
    M = P['M'].astype(np.float32)
    S0 = d @ M.T
    m = np.max(d * np.array([1.0, float(P['wG']), float(P['wB'])], dtype=np.float32), -1)
    t = np.clip((m - float(P['ta'])) / (float(P['tb']) - float(P['ta'])), 0, 1)[..., None]
    del d, m
    lw = LW709.astype(np.float32)
    S = S0 + t * ((S0 @ lw)[..., None] - S0)
    del S0, t
    Y = np.maximum(S, 0.0) @ lw
    lY = np.log2(np.maximum(Y, float(P.get('yeps', 1e-4))))
    lg = float(P['k']) * ndi.gaussian_filter(lY, float(P['sigma'])) + float(P.get('beta', 0.0)) * lY
    if float(P.get('k2', 0.0)) != 0.0:
        lg = lg + float(P['k2']) * ndi.gaussian_filter(lY, float(P['sigma2']))
    lv = np.where(S > 0, np.log2(np.maximum(S, np.float32(1e-12))), np.float32(-30.0)) + lg[..., None].astype(np.float32)
    del S
    lv = np.maximum(lv, np.float32(-30.0))
    out = np.interp(lv.ravel(), P['KN'], P['LUT']).reshape(lv.shape)
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return undo(out)


def main():
    if len(sys.argv) != 4:
        print('usage: pipeline.py IN.raw.png IN.json OUT.png', file=sys.stderr)
        sys.exit(2)
    raw = np.array(Image.open(sys.argv[1]))
    if raw.ndim == 3:
        raw = raw[..., 0]
    with open(sys.argv[2]) as f:
        meta = json.load(f)
    P = load_params()
    out = process(raw, meta, P)
    Image.fromarray(out, 'RGB').save(sys.argv[3])


if __name__ == '__main__':
    main()
