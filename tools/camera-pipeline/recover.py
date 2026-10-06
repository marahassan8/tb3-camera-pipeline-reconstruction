"""Check B: blind recovery of the camera pipeline from the 4 training pairs.

Uses only data/train (RAW + finished image) and never reads camera.P. The
structure is the hypothesis an imaging expert would test (demosaic, white
balance tweak, highlight clip and blend, colour matrix, local tone mapping on
log luminance, global tone curve, unsharp mask), with generic forms where the
exact formula cannot be known:

  - global tone + sRGB encoding: one free monotone curve (48 knots)
  - demosaic method: chosen from {bilinear, malvar}
  - local tone mapping base filter: chosen from {gaussian, guided} and radius

Continuous parameters are fitted with L-BFGS through a differentiable torch
model. The winner is then rendered on the hidden RAWs and scored.
"""

import glob
import itertools
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import camera
import score

torch.set_num_threads(6)
LUMA = torch.tensor(camera.LUMA, dtype=torch.float64)
KNOTS = 48
XMAX = 4.0


def load(split):
    items = []
    for f in sorted(glob.glob(f"data/{split}/*.raw.png")):
        base = f[:-8]
        m, meta = camera.load_raw(f, base + ".json")
        y = np.asarray(Image.open(base + ".out.png"))
        items.append((base.split("/")[-1], m, meta, y))
    return items


def channel_parts(m, meta, method):
    """Demosaic is linear, so demosaic(mosaic * gains) = sum_c gain_c * D_c."""
    x = np.maximum((m - meta["black_level"]) / (meta["white_level"] - meta["black_level"]), 0)
    asg = np.array(meta["as_shot_wb"])
    r, g1, g2, b = camera.masks(*x.shape)
    dem = camera.demosaic_malvar if method == "malvar" else camera.demosaic_bilinear
    parts = [dem(x * asg[0] * r), dem(x * asg[1] * (g1 | g2)), dem(x * asg[2] * b)]
    return torch.tensor(np.stack(parts), dtype=torch.float64), torch.tensor(asg)


def pad(x, r):
    return F.pad(x[None, None], (r, r, r, r), mode="reflect")[0, 0]


def box(x, r):
    return F.avg_pool2d(pad(x, r)[None, None], 2 * r + 1, stride=1)[0, 0]


def gauss_blur(x, sigma, radius):
    t = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-t * t / (2 * sigma * sigma))
    k = k / k.sum()
    xp = pad(x, radius)[None, None]
    xp = F.conv2d(xp, k.view(1, 1, 1, -1))
    xp = F.conv2d(xp, k.view(1, 1, -1, 1))
    return xp[0, 0]


class Model(torch.nn.Module):
    def __init__(s, ltm_family, radius):
        super().__init__()
        s.family, s.radius = ltm_family, radius
        f = lambda v: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64))
        s.tweak = f([1.0, 1.0])            # R and B relative to G
        s.off = f([0.0] * 6)               # colour matrix off-diagonals, rows sum to 1
        s.hl = f(0.85)
        s.log_eps = f(np.log(0.05))
        s.compress = f(0.8)
        xs = XMAX * (np.arange(KNOTS) / (KNOTS - 1)) ** 2
        init = camera.srgb_oetf(np.clip(1.5 * xs / (1 + 1.5 * xs), 0, 1))
        inc = np.maximum(np.diff(init, prepend=0), 1e-4)
        s.curve = f(np.log(np.expm1(inc)).tolist())
        s.s_sigma, s.s_amount, s.s_thr = f(1.2), f(0.4), f(0.005)

    def ccm(s):
        o = s.off
        M = torch.stack([torch.stack([1 - o[0] - o[1], o[0], o[1]]),
                         torch.stack([o[2], 1 - o[2] - o[3], o[3]]),
                         torch.stack([o[4], o[5], 1 - o[4] - o[5]])])
        return M

    def forward(s, parts, asg):
        g = torch.stack([s.tweak[0], torch.tensor(1.0, dtype=torch.float64), s.tweak[1]])
        rgb = torch.einsum("c,chwk->hwk", g, parts)
        clip = (asg * g).min()
        rgb = torch.clamp(rgb, min=0).clamp(max=clip) / clip
        t = torch.clamp((rgb.max(-1).values - s.hl) / (1 - s.hl), 0, 1)
        w = (t * t * (3 - 2 * t))[..., None]
        rgb = rgb * (1 - w) + rgb.mean(-1, keepdim=True) * w
        rgb = torch.clamp(rgb @ s.ccm().T, min=0)
        l = torch.log2(rgb @ LUMA + 1e-4)
        if s.family == "guided":
            r, eps = s.radius, torch.exp(s.log_eps)
            mean = box(l, r)
            var = box(l * l, r) - mean * mean
            a = var / (var + eps)
            b = mean - a * mean
            base = box(a, r) * l + box(b, r)
        else:
            base = gauss_blur(l, s.radius / 2, 2 * s.radius)
        rgb = rgb * torch.exp2((s.compress - 1) * (base - np.log2(0.18)))[..., None]
        # free monotone tone curve in sqrt-spaced knots, includes the encoding
        ys = torch.cumsum(F.softplus(s.curve), 0)
        u = torch.sqrt(torch.clamp(rgb, 0, XMAX) / XMAX) * (KNOTS - 1)
        i0 = torch.clamp(u.floor().long(), 0, KNOTS - 2)
        fr = u - i0
        v = ys[i0] * (1 - fr) + ys[i0 + 1] * fr
        v = torch.clamp(v, 0, 1)
        y = v @ LUMA
        d = y - gauss_blur(y, s.s_sigma, 5)
        d = torch.sign(d) * torch.clamp(d.abs() - s.s_thr, min=0)
        v = v + s.s_amount * d[..., None]
        return torch.clamp(v, 0, 1) * 255


def fit(model, data, iters):
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    B = score.BORDER

    def closure(grad=True):
        opt.zero_grad()
        loss = 0
        for parts, asg, y in data:
            pred = model(parts, asg)
            loss = loss + ((pred - y)[B:-B, B:-B] ** 2).mean()
        loss = loss / len(data)
        if grad:
            loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(closure(grad=False))


def prepare(items, method):
    out = []
    for _, m, meta, y in items:
        parts, asg = channel_parts(m, meta, method)
        out.append((parts, asg, torch.tensor(y, dtype=torch.float64)))
    return out


def main():
    train, hidden = load("train"), load("hidden")
    results = []
    t0 = time.time()
    cache = {d: prepare(train, d) for d in ("bilinear", "malvar")}
    for dem, fam, rad in itertools.product(("bilinear", "malvar"), ("gaussian", "guided"), (4, 8, 16)):
        torch.manual_seed(0)
        m = Model(fam, rad)
        mse = fit(m, cache[dem], 60)
        results.append((mse, dem, fam, rad, m))
        print(f"screen {dem:8s} {fam:8s} r={rad:2d}  train MSE {mse:8.3f}  ({time.time() - t0:5.0f}s)", flush=True)
    results.sort(key=lambda r: r[0])
    _, dem, fam, rad, _ = results[0]
    best = None
    for r in sorted({max(2, rad - 2), rad - 1, rad, rad + 1, rad + 2}):
        m = Model(fam, r)
        mse = fit(m, cache[dem], 300)
        print(f"refine {dem} {fam} r={r:2d}  train MSE {mse:8.4f}  ({time.time() - t0:5.0f}s)", flush=True)
        if best is None or mse < best[0]:
            best = (mse, r, m)
    mse, r, m = best
    print(f"chosen: {dem} demosaic, {fam} LTM r={r}, train MSE {mse:.4f}")
    with torch.no_grad():
        print("tweak", m.tweak.numpy().round(4), "hl", round(float(m.hl), 4),
              "eps", round(float(torch.exp(m.log_eps)), 4), "compress", round(float(m.compress), 4),
              "sharp", [round(float(v), 4) for v in (m.s_sigma, m.s_amount, m.s_thr)])
        print("ccm", m.ccm().numpy().round(3).tolist())
        for name, mm, meta, y in train + hidden:
            parts, asg = channel_parts(mm, meta, dem)
            pred = np.round(m(parts, asg).numpy()).astype(np.uint8)
            Image.fromarray(pred).save(f"out/recovered_{name}.png")
            print(f"recovered {name:15s} {score.fmt(score.compare(pred, y))}", flush=True)
    torch.save(m.state_dict(), "out/recovered.pt")
    json.dump({"demosaic": dem, "family": fam, "radius": r}, open("out/recovered.json", "w"))


if __name__ == "__main__":
    main()
