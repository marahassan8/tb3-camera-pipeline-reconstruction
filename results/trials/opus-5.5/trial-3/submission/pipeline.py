"""Reproduce the camera's RAW -> 8-bit sRGB rendering.

Usage: python3 pipeline.py IN.raw.png IN.json OUT.png

Reverse-engineered processing chain (parameters in params.json):
  1. normalise: (raw - black) / (white - black), clip [0, 1], multiply by as-shot WB gains
  2. Malvar-He-Cutler (MHC) demosaic, clip [0, 1]
  3. 3x3 colour matrix (camera WB RGB -> linear RGB, exposure folded in), clip >= 0
  4. highlight desaturation: blend towards a neutral (mean of the camera channels scaled
     to the luminance of white) with smoothstep(t0, t1, max channel of the demosaiced pixel)
  5. tone mapping in log space: luminance compressed with exponent c, chroma ratios
     scaled by exponent s, then per-channel Hable filmic curve and sRGB transfer
     followed by a near-identity 3x3 mix of the encoded channels
  6. unsharp mask on Rec.709 luma (Gaussian sigma, amount k), added equally to R, G, B
  7. round and clip to 8 bit
"""
import sys, json, os
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'params.json')))
NAMES_P = ['m00','m01','m02','m10','m11','m12','m20','m21','m22','le','e','s','c','lW','t0','t1','k','sig']
P = dict(zip(NAMES_P, CFG['p'][:len(NAMES_P)]))
CORR = np.array(CFG['p'][len(NAMES_P):], dtype=np.float64)
_s = CFG.get('S', [0, 0, 0, 0, 0, 0])
SMAT = np.array([[1 - _s[0] - _s[1], _s[0], _s[1]], [_s[2], 1 - _s[2] - _s[3], _s[3]], [_s[4], _s[5], 1 - _s[4] - _s[5]]])
CORR_KN = np.arange(-7.0, 2.01, 0.5)
W709 = np.array([0.2126, 0.7152, 0.0722])
W601 = np.array([0.299, 0.587, 0.114])
LUMA = W709

def mhc(y):
    """Malvar-He-Cutler demosaic of an RGGB mosaic (scipy convolve, mirror borders)."""
    GR_GB = np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]], float) / 8
    Rg_RB = np.array([[0,0,0.5,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],[0,-1,0,-1,0],[0,0,0.5,0,0]], float) / 8
    Rg_BR = Rg_RB.T.copy()
    Rb_BB = np.array([[0,0,-1.5,0,0],[0,2,0,2,0],[-1.5,0,6,0,-1.5],[0,2,0,2,0],[0,0,-1.5,0,0]], float) / 8
    H, W = y.shape
    out = np.empty((H, W, 3), np.float64)
    cg = convolve(y, GR_GB, mode='mirror')
    c1 = convolve(y, Rg_RB, mode='mirror')
    c2 = convolve(y, Rg_BR, mode='mirror')
    c3 = convolve(y, Rb_BB, mode='mirror')
    R, G, B = out[..., 0], out[..., 1], out[..., 2]
    # R sites (even row, even col)
    R[0::2, 0::2] = y[0::2, 0::2]; G[0::2, 0::2] = cg[0::2, 0::2]; B[0::2, 0::2] = c3[0::2, 0::2]
    # G sites in R rows (even row, odd col)
    R[0::2, 1::2] = c1[0::2, 1::2]; G[0::2, 1::2] = y[0::2, 1::2]; B[0::2, 1::2] = c2[0::2, 1::2]
    # G sites in B rows (odd row, even col)
    R[1::2, 0::2] = c2[1::2, 0::2]; G[1::2, 0::2] = y[1::2, 0::2]; B[1::2, 0::2] = c1[1::2, 0::2]
    # B sites (odd row, odd col)
    R[1::2, 1::2] = c3[1::2, 1::2]; G[1::2, 1::2] = cg[1::2, 1::2]; B[1::2, 1::2] = y[1::2, 1::2]
    return out

def srgb(v):
    v = np.clip(v, 0, None)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)

def hable(z):
    return (z * (0.15 * z + 0.05) + 0.004) / (z * (0.15 * z + 0.5) + 0.06) - 0.02 / 0.3

def srgb_inplace(v):
    np.maximum(v, 0, out=v)
    lo = v <= 0.0031308
    lov = v[lo] * 12.92
    np.power(v, 1 / 2.4, out=v)
    v *= 1.055
    v -= 0.055
    v[lo] = lov
    return v

def glob(d):
    H, W, _ = d.shape
    d2 = d.reshape(-1, 3)
    M = np.array([[P['m00'], P['m01'], P['m02']], [P['m10'], P['m11'], P['m12']], [P['m20'], P['m21'], P['m22']]])
    lin = d2 @ M.T
    np.maximum(lin, 0, out=lin)
    t = np.clip((d2.max(-1) - P['t0']) / max(P['t1'] - P['t0'], 1e-3), 0, 1)
    h = t * t * (3 - 2 * t)
    if CFG.get('hl_target') == 'meand':
        # neutral target: mean of the camera channels, scaled to the luminance of white
        T = d2.mean(-1) * (M.sum(1) @ W709)
        lin *= (1 - h)[:, None]
        lin += (h * T)[:, None]
        L = np.log(np.maximum(lin @ W709, 0) + np.exp(P['le']))
    else:
        Y = lin @ W709
        lin *= (1 - h)[:, None]
        lin += (h * Y)[:, None]
        L = np.log(np.maximum(Y, 0) + np.exp(P['le']))
    # lx = s*(ln(lin) - L) + c*L + e
    np.maximum(lin, 1e-12, out=lin)
    np.log(lin, out=lin)
    lin *= P['s']
    lin += ((P['c'] - P['s']) * L + P['e'])[:, None]
    lx = lin if not CORR.size else lin.copy()
    z = np.exp(lin, out=lin)
    # Hable filmic curve
    num = z * 0.15; num += 0.05; num *= z; num += 0.004
    den = z * 0.15; den += 0.5; den *= z; den += 0.06
    num /= den
    num -= 0.02 / 0.3
    num /= hable(np.exp(P['lW']))
    O = srgb_inplace(num)
    O *= 255
    if CORR.size:
        O += np.interp(lx, CORR_KN, np.concatenate([[0.0], CORR, [0.0]]))
    O = O @ SMAT.T
    return O.reshape(H, W, 3)

def _gains_map(H, W, vals_rggb):
    # vals_rggb: values for (R, G1, G2, B) positions of an RGGB mosaic
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.empty((H, W), np.float64)
    out[(yy % 2 == 0) & (xx % 2 == 0)] = vals_rggb[0]
    out[(yy % 2 == 0) & (xx % 2 == 1)] = vals_rggb[1]
    out[(yy % 2 == 1) & (xx % 2 == 0)] = vals_rggb[2]
    out[(yy % 2 == 1) & (xx % 2 == 1)] = vals_rggb[3]
    return out

def process(raw, meta):
    raw = raw.astype(np.float64)
    if raw.ndim == 3:
        raw = raw[..., 0]
    cfa = str(meta.get('cfa', 'RGGB')).upper()
    # bring the mosaic to RGGB phase by padding one row/column (mirror keeps colours consistent)
    dy = 1 if cfa in ('GBRG', 'BGGR') else 0
    dx = 1 if cfa in ('GRBG', 'BGGR') else 0
    H0, W0 = raw.shape
    if dy or dx:
        raw = np.pad(raw, ((dy, 0), (dx, 0)), mode='reflect')
        if dy:
            raw[0] = raw[2]
        if dx:
            raw[:, 0] = raw[:, 2]
    H, W = raw.shape
    bl = meta.get('black_level', 0)
    if np.ndim(bl) == 0:
        blm = float(bl)
    else:
        bl = list(np.ravel(bl))
        bl = (bl * 4)[:4]
        blm = _gains_map(H, W, bl)
    wl = float(np.max(meta.get('white_level', 65535)))
    g = list(meta.get('as_shot_wb', [1.0, 1.0, 1.0]))
    if len(g) == 4:
        g = [g[0], g[1], g[3]]
    gain = _gains_map(H, W, [g[0], g[1], g[1], g[2]])
    x = (raw - blm) / (wl - blm)
    if CFG.get('clip_before_wb'):
        y = np.clip(x, 0, 1) * gain
    else:
        y = np.clip(x * gain, 0, 1)
    d = np.clip(mhc(y), 0, 1)
    O = glob(d)
    Yl = O @ LUMA
    out = O + P['k'] * (Yl - gaussian_filter(Yl, P['sig']))[..., None]
    out = out[dy:dy + H0, dx:dx + W0]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

if __name__ == '__main__':
    raw = np.array(Image.open(sys.argv[1]))
    meta = json.load(open(sys.argv[2]))
    Image.fromarray(process(raw, meta)).save(sys.argv[3])
