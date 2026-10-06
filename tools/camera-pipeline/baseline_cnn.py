"""Check A, part 2: a small U-Net trained on the 4 training pairs (CPU).

Same advantaged input as baseline_lut.py (as-shot white balance, the camera's
own demosaic, cube-root encoding). Receptive field covers the local tone
mapping window. Trained on random 128x128 patches with L1 loss.
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import baseline_lut as bl
import score

torch.manual_seed(0)
np.random.seed(0)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        s.e1, s.e2, s.e3, s.e4 = block(3, c), block(c, 2 * c), block(2 * c, 2 * c), block(2 * c, 2 * c)
        s.d3, s.d2, s.d1 = block(4 * c, 2 * c), block(4 * c, 2 * c), block(3 * c, c)
        s.out = nn.Conv2d(c, 3, 1)

    def forward(s, x):
        e1 = s.e1(x)
        e2 = s.e2(F.avg_pool2d(e1, 2))
        e3 = s.e3(F.avg_pool2d(e2, 2))
        e4 = s.e4(F.avg_pool2d(e3, 2))
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        d3 = s.d3(torch.cat([up(e4, e3), e3], 1))
        d2 = s.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = s.d1(torch.cat([up(d2, e1), e1], 1))
        return torch.sigmoid(s.out(d1))


def t(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()[None]


def main(minutes=25.0):
    train, hidden = bl.load("train"), bl.load("hidden")
    X = [t(x) for _, x, _ in train]
    Y = [t(y / 255) for _, _, y in train]
    net = UNet()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched_end = time.time() + minutes * 60
    step, P = 0, 128
    while time.time() < sched_end:
        xs, ys = [], []
        for _ in range(16):
            k = np.random.randint(len(X))
            h, w = X[k].shape[-2:]
            i, j = np.random.randint(h - P), np.random.randint(w - P)
            xs.append(X[k][..., i:i + P, j:j + P])
            ys.append(Y[k][..., i:i + P, j:j + P])
        loss = F.l1_loss(net(torch.cat(xs)), torch.cat(ys))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step == 2000:
            for g in opt.param_groups:
                g["lr"] = 5e-4
        if step % 200 == 0:
            print(f"step {step} loss {loss.item() * 255:.2f} levels", flush=True)
    net.eval()
    with torch.no_grad():
        for n, x, y in train + hidden:
            pred = net(t(x))[0].numpy().transpose(1, 2, 0)
            pred = np.clip(np.round(pred * 255), 0, 255).astype(np.uint8)
            Image.fromarray(pred).save(f"out/cnn_{n}.png")
            print(f"cnn {n:15s} {score.fmt(score.compare(pred, y.astype(np.uint8)))}", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 25.0)
