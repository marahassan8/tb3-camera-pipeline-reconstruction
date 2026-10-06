#!/usr/bin/env python3
"""Reproduce the camera's own processing: RAW (16-bit Bayer PNG) + metadata JSON -> 8-bit sRGB PNG.

usage: python pipeline.py IN.raw.png IN.json OUT.png

Pipeline (reverse-engineered from the training pairs):
  1. black-level subtraction, normalisation to [0, 1]   (clipped at 0)
  2. white balance applied on the mosaic (NOT clipped)
  3. Malvar-He-Cutler demosaicing; result clipped to [0, 1]
  4. 3x3 colour matrix; result clipped at 0
  5. luminance-preserving saturation boost  v' = Y * (v / Y) ** s   (Y = Rec.709 luminance)
  6. highlight desaturation towards luminance as the max white-balanced channel approaches clipping
  7. tone curve: piecewise-linear in ln(v) (power-law toe, log-linear shoulder)
  8. luminance unsharp mask (Gaussian, one or two scales) in the output domain
  9. rounding to 8 bit
"""
import sys, json
import numpy as np
from PIL import Image
from scipy import ndimage

PARAMS = {
 "M": [
  [
   4.5015007370786,
   -1.3231702507665992,
   -0.33698810545754354
  ],
  [
   -0.548002857161126,
   3.799949676185418,
   -0.5612699148361744
  ],
  [
   0.11131564761102056,
   -1.2332481099963244,
   3.650077789461218
  ]
 ],
 "s": 1.420164768322656,
 "w_sat": [
  0.2126,
  0.7152,
  0.0722
 ],
 "highlight": [
  0.7806864404548569,
  0.9813503189954296
 ],
 "knots": [
  -6.2298917783039265,
  -5.7298917783039265,
  -5.2298917783039265,
  -4.7298917783039265,
  -4.2298917783039265,
  -3.7298917783039265,
  -3.2298917783039265,
  -2.7298917783039265,
  -2.2298917783039265,
  -1.7298917783039263,
  -1.2298917783039263,
  -0.7298917783039263,
  -0.22989177830392626,
  0.27010822169607374,
  0.7701082216960737,
  1.1201082216960736,
  1.4701082216960737
 ],
 "values": [
  17.28679726208701,
  21.99210395746908,
  27.932037770162655,
  34.76242640506937,
  42.42683126989367,
  51.12342721111821,
  61.19993291280899,
  72.55332007157898,
  85.01037616909602,
  99.19914551598956,
  114.45564242576542,
  130.8270039310429,
  148.18589173439005,
  165.72245030343777,
  183.1147995250688,
  194.44767810821673,
  206.46301313184267
 ],
 "toe_exp": 0.640833849204844,
 "w_usm": [
  0.2126,
  0.7152,
  0.0722
 ],
 "usm": [
  [
   0.5509879125453184,
   1.0093180096298027
  ]
 ]
}

def cfa_masks(cfa, H, W):
    cfa = cfa.upper()
    assert len(cfa) == 4 and sorted(cfa) == ["B", "G", "G", "R"], cfa
    yy, xx = np.mgrid[0:H, 0:W]
    pat = np.array(list(cfa)).reshape(2, 2)
    site = pat[yy % 2, xx % 2]
    rows_with_R = [r for r in range(2) if "R" in pat[r]]
    isR = site == "R"; isB = site == "B"; isG = site == "G"
    inRrow = np.isin(yy % 2, rows_with_R)
    return isR, isG & inRrow, isG & ~inRrow, isB

def mhc_demosaic(m, cfa):
    """Malvar-He-Cutler (2004) demosaicing of a Bayer mosaic m (H x W, float)."""
    H, W = m.shape
    GR = np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]], float) / 8
    RgRB = np.array([[0,0,0.5,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],[0,-1,0,-1,0],[0,0,0.5,0,0]], float) / 8
    RgBR = RgRB.T.copy()
    Rb = np.array([[0,0,-1.5,0,0],[0,2,0,2,0],[-1.5,0,6,0,-1.5],[0,2,0,2,0],[0,0,-1.5,0,0]], float) / 8
    conv = lambda k: ndimage.convolve(m, k, mode='mirror')
    g_at_rb = conv(GR); c_h = conv(RgRB); c_v = conv(RgBR); c_x = conv(Rb)
    isR, isG1, isG2, isB = cfa_masks(cfa, H, W)
    R = np.empty_like(m); G = np.empty_like(m); B = np.empty_like(m)
    G[isG1 | isG2] = m[isG1 | isG2]; G[isR | isB] = g_at_rb[isR | isB]
    R[isR] = m[isR]; R[isG1] = c_h[isG1]; R[isG2] = c_v[isG2]; R[isB] = c_x[isB]
    B[isB] = m[isB]; B[isG1] = c_v[isG1]; B[isG2] = c_h[isG2]; B[isR] = c_x[isR]
    return np.stack([R, G, B], -1)

def tone_curve(knots, values, toe_exp, vd):
    knots = np.asarray(knots, float); values = np.asarray(values, float)
    lv = np.log(np.where(vd > 0, vd, 1e-12))
    t = np.interp(lv, knots, values)
    t = np.where(lv < knots[0], values[0] * np.exp(toe_exp * (lv - knots[0])), t)
    slope = (values[-1] - values[-2]) / (knots[-1] - knots[-2])
    t = np.where(lv > knots[-1], values[-1] + slope * (lv - knots[-1]), t)
    return np.where(vd > 0, t, -1e9)

def process(raw, meta, P):
    black = float(meta["black_level"]); white = float(meta["white_level"])
    wb = [float(g) for g in meta["as_shot_wb"]]
    cfa = meta.get("cfa", "RGGB")
    H, W = raw.shape
    x = np.clip((raw - black) / (white - black), 0.0, 1.0)
    isR, isG1, isG2, isB = cfa_masks(cfa, H, W)
    gain = np.empty((H, W)); gain[isR] = wb[0]; gain[isG1 | isG2] = wb[1]; gain[isB] = wb[2]
    lin = np.clip(mhc_demosaic(x * gain, cfa), 0.0, 1.0)
    M = np.array(P["M"], float)
    v = np.maximum(lin @ M.T, 0.0)
    w_sat = np.array(P["w_sat"], float)
    Y = v @ w_sat
    Ypos = Y > 1e-9
    Yp = np.where(Ypos, Y, 1e-9)[..., None]
    r = v / Yp
    vs = Yp * np.sign(r) * np.power(np.abs(r), P["s"])
    vs = np.where(Ypos[..., None], vs, 0.0)
    # highlight desaturation toward luminance, keyed on the maximum white-balanced channel
    a, b = P["highlight"]            # smoothstep ramp: full colour below a, fully neutral above b
    tt = np.clip((lin.max(-1) - a) / max(b - a, 1e-4), 0.0, 1.0)
    k = (1.0 - (3.0 * tt * tt - 2.0 * tt * tt * tt))[..., None]
    vd = Y[..., None] + k * (vs - Y[..., None])
    t = tone_curve(P["knots"], P["values"], P["toe_exp"], vd)
    t = np.clip(t, 0.0, 255.0)
    w_usm = np.array(P["w_usm"], float)
    Yo = t @ w_usm
    corr = np.zeros_like(Yo)
    for amount, sigma in P["usm"]:
        corr += amount * (Yo - ndimage.gaussian_filter(Yo, sigma, mode='nearest'))
    t = np.clip(t + corr[..., None], 0.0, 255.0)
    return np.round(t).astype(np.uint8)

def main():
    raw_path, json_path, out_path = sys.argv[1:4]
    raw = np.array(Image.open(raw_path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = raw.astype(np.float64)
    meta = json.load(open(json_path))
    out = process(raw, meta, PARAMS)
    Image.fromarray(out, mode="RGB").save(out_path)

if __name__ == "__main__":
    main()
