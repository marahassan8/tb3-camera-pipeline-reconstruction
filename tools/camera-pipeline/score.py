"""Compare a rendered image with the truth.

A pixel agrees within k if every channel differs by at most k levels. A
16-pixel border is excluded, as is common in image-restoration benchmarks.
"""

import numpy as np

BORDER = 16


def compare(pred, truth):
    p = pred.astype(int)[BORDER:-BORDER, BORDER:-BORDER]
    t = truth.astype(int)[BORDER:-BORDER, BORDER:-BORDER]
    d = np.abs(p - t).max(-1)
    mse = np.mean((p - t) ** 2)
    return {
        "w2": float(np.mean(d <= 2)),
        "w8": float(np.mean(d <= 8)),
        "psnr": float(10 * np.log10(255 ** 2 / mse)) if mse else float("inf"),
    }


def fmt(s):
    return f"{s['w2']:8.3%} {s['w8']:8.3%} {s['psnr']:6.1f}dB"
