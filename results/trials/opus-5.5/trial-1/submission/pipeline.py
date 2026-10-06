#!/usr/bin/env python3
"""RAW (RGGB Bayer PNG + JSON metadata) -> camera sRGB rendering."""
import sys, os, json
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, correlate

HERE = os.path.dirname(os.path.abspath(__file__))


def malvar(m):
    """Malvar-He-Cutler demosaic of an RGGB mosaic (R at [0,0])."""
    h, w = m.shape
    R = np.zeros((h, w), bool); R[0::2, 0::2] = True
    B = np.zeros((h, w), bool); B[1::2, 1::2] = True
    G = ~(R | B)
    GR = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]) / 8.
    Rg_rb = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]]) / 8.
    Rg_br = Rg_rb.T
    Rb = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]) / 8.
    c = lambda k: convolve(m, k, mode='mirror')
    g = np.where(G, m, c(GR))
    a, b, d = c(Rg_rb), c(Rg_br), c(Rb)
    rowR = np.zeros((h, w), bool); rowR[0::2] = True
    r = np.where(R, m, np.where(G & rowR, a, np.where(G, b, d)))
    bb = np.where(B, m, np.where(G & ~rowR, a, np.where(G, b, d)))
    return np.stack([r, g, bb], -1)


def lut(t, kn, v):
    pos = np.clip(np.nan_to_num((t - kn[0]) / (kn[1] - kn[0])), 0, len(kn) - 1 - 1e-3)
    i = np.floor(pos).astype(np.int64); f = pos - i
    return v[i] * (1 - f) + v[i + 1] * f


def process(raw, meta, P):
    bl, wl = float(meta['black_level']), float(meta['white_level'])
    wb = meta['as_shot_wb']
    r = (raw.astype(np.float64) - bl) / (wl - bl)
    h, w = r.shape
    g = np.full((h, w), float(wb[1])); g[0::2, 0::2] = wb[0]; g[1::2, 1::2] = wb[2]
    m = np.minimum(r * g, 1.0)
    x = malvar(m)
    z = x @ P['M'].T
    L = z @ P['w']
    bk = P['bk']
    Lb = np.maximum(correlate(L, bk, mode='mirror'), 1e-4)
    A = lambda v: np.arcsinh(v / 0.003)
    tl = A(Lb)
    gain = np.stack([np.exp(np.clip(lut(tl, P['gkn'], P['gv'][c]), -5, 5)) for c in range(len(P['gv']))], -1)
    t = A(z * gain)
    o = np.stack([lut(t[..., c], P['Ekn'], P['Ev'][c]) for c in range(3)], -1)
    o = o @ P['Amat'].T + P['b']
    sk = P['sk']
    o = np.stack([correlate(o[..., c], sk, mode='nearest') for c in range(3)], -1)
    return o, x


def conv3(f, W, b):
    """3x3 'same' cross-correlation (edge-replicated), f: (Cin,H,W), W: (Cout,Cin,3,3)."""
    cin, h, w = f.shape
    fp = np.pad(f, ((0, 0), (1, 1), (1, 1)), mode='edge')
    Wm = W.reshape(W.shape[0], cin, 9).astype(np.float32)
    out = np.empty((W.shape[0], h, w), np.float32)
    step = max(8, int(2e6 // max(1, cin * 9 * w)))
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        cols = np.stack([fp[:, y0 + dy:y1 + dy, dx:dx + w] for dy in range(3) for dx in range(3)], 1)
        out[:, y0:y1] = np.tensordot(Wm, cols, axes=([1, 2], [0, 1])) + b[:, None, None]
    return out


def refine(o, x, C):
    feats = [o / 128.0 - 1, np.arcsinh(x / 0.01) / 5.0]
    idx = sorted(int(k.split('.')[0]) for k in C if k.endswith('.weight'))
    if C['%d.weight' % idx[0]].shape[1] == 9:
        feats.append(np.arcsinh(x / 0.0005) / 8.0)
    h = np.concatenate(feats, -1).transpose(2, 0, 1).astype(np.float32)
    for j, i in enumerate(idx):
        h = conv3(h, C['%d.weight' % i], C['%d.bias' % i])
        if j < len(idx) - 1:
            h = np.maximum(h, 0)
    return o + h.transpose(1, 2, 0)


def finish(o, P, x=None, C=None):
    if 'rlut' in P:
        o = o + apply_lut3(o, P['rlut'])
    if C is not None and o.shape[0] * o.shape[1] <= 3.5e6:
        o = refine(o, x, C)
    return np.clip(np.round(o), 0, 255).astype(np.uint8)


def apply_lut3(o, lut3):
    G = lut3.shape[0]
    t = np.clip(o, 0, 255) / 255.0 * (G - 1)
    i0 = np.clip(np.floor(t).astype(np.int64), 0, G - 2); f = t - i0
    out = 0
    for dr in (0, 1):
        wr = f[..., 0] if dr else 1 - f[..., 0]
        for dg in (0, 1):
            wg = f[..., 1] if dg else 1 - f[..., 1]
            for db in (0, 1):
                wbb = f[..., 2] if db else 1 - f[..., 2]
                out = out + (wr * wg * wbb)[..., None] * lut3[i0[..., 0] + dr, i0[..., 1] + dg, i0[..., 2] + db]
    return out


def main():
    raw_p, json_p, out_p = sys.argv[1:4]
    raw = np.array(Image.open(raw_p))
    with open(json_p) as f:
        meta = json.load(f)
    P = dict(np.load(os.path.join(HERE, 'params.npz')))
    cp = os.path.join(HERE, 'cnn.npz')
    C = dict(np.load(cp)) if os.path.exists(cp) else None
    o, x = process(raw, meta, P)
    Image.fromarray(finish(o, P, x, C)).save(out_p)


if __name__ == '__main__':
    main()
