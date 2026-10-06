"""Check A, part 1: generic colour fitting on the 4 training pairs.

The input is given every advantage: the RAW is white-balanced with the
as-shot gains and demosaiced with the camera's own method (Malvar). Only the
mapping to the finished image is fitted.

  lut3d  33^3 lookup table, trilinear, cube-root input encoding
  poly   per-pixel degree-3 polynomial colour regression on the encoded input
"""

import glob
import itertools

import numpy as np
from PIL import Image

import camera
import score

N = 33


def linear_input(m, meta):
    x = np.maximum((m - meta["black_level"]) / (meta["white_level"] - meta["black_level"]), 0)
    g = np.array(meta["as_shot_wb"])
    r, _, _, b = camera.masks(*x.shape)
    x = x * np.where(r, g[0], np.where(b, g[2], 1.0))
    return camera.demosaic_malvar(x)


def encode(rgb):
    return np.clip(np.cbrt(np.clip(rgb, 0, None) / 2.5), 0, 1)


def load(split):
    out = []
    for f in sorted(glob.glob(f"data/{split}/*.raw.png")):
        base = f[:-8]
        m, meta = camera.load_raw(f, base + ".json")
        out.append((base.split("/")[-1], encode(linear_input(m, meta)),
                    np.asarray(Image.open(base + ".out.png")).astype(np.float64)))
    return out


def trilinear_weights(x):
    g = x * (N - 1)
    i0 = np.minimum(np.floor(g).astype(int), N - 2)
    f = g - i0
    idx, w = [], []
    for c in itertools.product((0, 1), repeat=3):
        c = np.array(c)
        idx.append(((i0 + c) * np.array([N * N, N, 1])).sum(-1))
        w.append(np.prod(np.where(c, f, 1 - f), -1))
    return np.stack(idx, -1), np.stack(w, -1)


def fit_lut(train):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    X = np.concatenate([x.reshape(-1, 3) for _, x, _ in train])
    Y = np.concatenate([y.reshape(-1, 3) for _, _, y in train])
    idx, w = trilinear_weights(X)
    rows = np.repeat(np.arange(len(X)), 8)
    A = coo_matrix((w.ravel(), (rows, idx.ravel())), shape=(len(X), N ** 3)).tocsr()
    # Small smoothness prior keeps empty cells defined (graph Laplacian).
    lap = []
    for axis_stride in (N * N, N, 1):
        for cell in range(N ** 3):
            coord = (cell // (N * N), (cell // N) % N, cell % N)
            k = {N * N: 0, N: 1, 1: 2}[axis_stride]
            if coord[k] < N - 1:
                lap.append((cell, cell + axis_stride))
    lap = np.array(lap)
    L = coo_matrix((np.r_[np.ones(len(lap)), -np.ones(len(lap))],
                    (np.r_[np.arange(len(lap)), np.arange(len(lap))], np.r_[lap[:, 0], lap[:, 1]])),
                   shape=(len(lap), N ** 3)).tocsr()
    from scipy.sparse import vstack
    M = vstack([A, 0.5 * L])
    lut = np.stack([lsqr(M, np.r_[Y[:, c], np.zeros(len(lap))], atol=1e-8, btol=1e-8, iter_lim=3000)[0]
                    for c in range(3)], -1)
    return lambda x: (lambda iw: (lut[iw[0]] * iw[1][..., None]).sum(-2))(trilinear_weights(x))


def poly_features(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    feats = [np.ones_like(r)]
    for d in range(1, 4):
        for c in itertools.combinations_with_replacement((r, g, b), d):
            feats.append(np.prod(c, 0))
    return np.stack(feats, -1)


def fit_poly(train):
    F = np.concatenate([poly_features(x).reshape(-1, 20) for _, x, _ in train])
    Y = np.concatenate([y.reshape(-1, 3) for _, _, y in train])
    W, *_ = np.linalg.lstsq(F, Y, rcond=None)
    return lambda x: poly_features(x) @ W


def main():
    train, hidden = load("train"), load("hidden")
    for name, fit in (("lut3d", fit_lut), ("poly3", fit_poly)):
        f = fit(train)
        for n, x, y in train + hidden:
            pred = np.clip(np.round(f(x)), 0, 255).astype(np.uint8)
            print(f"{name:6s} {n:15s} {score.fmt(score.compare(pred, y.astype(np.uint8)))}")


if __name__ == "__main__":
    main()
