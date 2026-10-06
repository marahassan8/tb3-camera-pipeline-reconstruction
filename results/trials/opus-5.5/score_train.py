# Reference score on the training photos, same metric as the verifier (16-px border, max over channels).
import glob, json, os, subprocess, sys, time
import numpy as np
from PIL import Image
B = 16
for raw in sorted(glob.glob("/train/*.raw.png")):
    n = os.path.basename(raw)[:-8]
    out = f"/tmp/{n}.png"
    t0 = time.time()
    p = subprocess.run(["python3", "/app/camera/pipeline.py", raw, raw[:-8] + ".json", out], capture_output=True, text=True, cwd="/app/camera")
    dt = time.time() - t0
    if p.returncode != 0:
        print(f"train {n:14s} FAILED: {p.stderr[-300:]}"); continue
    a = np.asarray(Image.open(out).convert("RGB")).astype(int)
    b = np.asarray(Image.open(raw[:-8] + ".out.png").convert("RGB")).astype(int)
    d = np.abs(a - b).max(-1)[B:-B, B:-B]
    print(f"train {n:14s} {np.mean(d <= 2):8.3%} within 2  {np.mean(d <= 8):8.3%} within 8  {dt:5.1f} s")
