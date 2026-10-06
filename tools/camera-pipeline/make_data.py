"""Build the training pairs and hidden RAW photos.

Scenes are scikit-image sample photos (public-domain / CC0) turned back into
linear scene light, plus procedural charts. Each scene is "photographed": light
in camera RGB, illuminant cast, exposure, specular highlights, per-channel
sensor saturation, RGGB mosaic, shot and read noise, 14-bit levels. The camera
pipeline in camera.py then renders the finished image.
"""

import json
import os

import numpy as np
import skimage.data as sd
from PIL import Image
from skimage.transform import rescale

import camera

BLACK, WHITE = 512, 16383
ILLUMINANT = {  # as-shot white balance the camera records
    "daylight": [2.00, 1.0, 1.55],
    "cloudy": [2.20, 1.0, 1.40],
    "tungsten": [1.35, 1.0, 2.40],
    "fluorescent": [1.70, 1.0, 2.00],
}
INV_CCM = np.linalg.inv(np.array(camera.P["ccm"]))


def to_linear(img8):
    x = img8[..., :3].astype(np.float64) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def photo(name, down=1):
    x = getattr(sd, name)()
    if isinstance(x, tuple):
        x = x[0]
    lin = to_linear(x)
    if down > 1:
        lin = rescale(lin, 1 / down, channel_axis=-1, anti_aliasing=True)
    return np.maximum(lin @ INV_CCM.T, 0)  # scene light in camera RGB


def even(a):
    return a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]


# ColorChecker Classic, approximate sRGB values.
CHECKER = [(115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
           (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
           (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161),
           (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52)]


def chart(h=480, w=720):
    """Calibration-style shot: colour checker, grey ramp, a clipped white
    patch, a Siemens star and fine bars. Every mechanism appears here at
    moderate strength."""
    lin = np.full((h, w, 3), 0.18)
    ph, pw = 60, 70
    for i, c in enumerate(CHECKER):
        y, x = 20 + (i // 6) * (ph + 10), 20 + (i % 6) * (pw + 10)
        lin[y:y + ph, x:x + pw] = to_linear(np.array(c, np.uint8))
    lin[300:340, 20:500] = np.linspace(0.002, 1.0, 480)[None, :, None]
    lin[360:460, 20:120] = 3.0  # overexposed white patch
    yy, xx = np.mgrid[-50:50, -50:50]
    star = 0.5 + 0.45 * np.sign(np.sin(16 * np.arctan2(yy, xx)))
    lin[360:460, 140:240] = star[..., None] * 0.6
    bars = 0.05 + 0.55 * (np.sin(np.arange(220) * np.pi / 2.2) > 0)
    lin[360:460, 260:480] = bars[None, :, None]
    lin[20:460, 520:700] = np.linspace(0.01, 0.6, 440)[:, None, None] * np.array([1.0, 0.55, 0.25])
    return np.maximum(lin @ INV_CCM.T, 0)


def saturated_chart(h=480, w=720):
    """Hidden: colours far more saturated than anything in training,
    including narrow-band light given directly in camera RGB."""
    cam = np.full((h, w, 3), 0.05)
    wheel = to_linear(sd.colorwheel())
    wheel = np.maximum(wheel @ INV_CCM.T, 0)
    cam[40:40 + 370, 20:20 + 371] = wheel[:, :, :3]
    narrow = [(0.9, 0.04, 0.01), (0.02, 0.8, 0.03), (0.01, 0.06, 0.9), (0.7, 0.6, 0.01),
              (0.01, 0.5, 0.6), (0.6, 0.02, 0.5), (1.6, 0.1, 0.05), (0.05, 0.2, 1.8)]
    for i, c in enumerate(narrow):
        y, x = 30 + (i // 2) * 105, 420 + (i % 2) * 145
        cam[y:y + 90, x:x + 130] = c
    return cam


def zone_plate(n=300):
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    z = 0.5 + 0.45 * np.cos(np.pi * (xx ** 2 + yy ** 2) / (1.1 * n))
    return np.repeat(z[..., None], 3, -1) * 0.5 @ INV_CCM.T


def speculars(cam, spots, seed):
    rng = np.random.default_rng(seed)
    h, w, _ = cam.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(spots):
        cy, cx = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
        s = rng.uniform(3, 12)
        tint = rng.uniform(0.6, 1.0, 3)
        cam = cam + 8 * tint * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))[..., None]
    return cam


def shoot(cam, illuminant, exposure, seed):
    wb = np.array(ILLUMINANT[illuminant])
    x = np.minimum(even(cam) * exposure / wb, 1.0)  # illuminant cast, sensor saturation
    r, g1, g2, b = camera.masks(*x.shape[:2])
    m = np.where(r, x[..., 0], np.where(b, x[..., 2], x[..., 1]))
    rng = np.random.default_rng(seed)
    m = m + np.sqrt(1e-4 * m + 3e-4 ** 2) * rng.standard_normal(m.shape)
    raw = np.clip(np.round(BLACK + m * (WHITE - BLACK)), 0, WHITE).astype(np.uint16)
    meta = {"cfa": "RGGB", "black_level": BLACK, "white_level": WHITE, "as_shot_wb": list(wb)}
    return raw, meta


SHOTS = {
    "train": [
        ("t1_skin", lambda: photo("skin", 2), "daylight", 1.0, 0),
        ("t2_astronaut", lambda: speculars(photo("astronaut"), 3, 2), "cloudy", 1.4, 0),
        ("t3_coffee", lambda: photo("coffee"), "tungsten", 0.5, 0),
        ("t4_chart", chart, "fluorescent", 1.0, 0),
    ],
    "hidden": [
        ("h1_hubble", lambda: speculars(photo("hubble_deep_field", 2), 6, 11), "daylight", 0.35, 0),
        ("h2_saturated", saturated_chart, "tungsten", 1.0, 0),
        ("h3_motorcycle", lambda: speculars(photo("stereo_motorcycle"), 5, 13), "fluorescent", 1.8, 0),
        ("h4_texture", lambda: np.concatenate([photo("chelsea"), zone_plate(300)], 1), "cloudy", 1.0, 0),
    ],
}


def main(out="data"):
    for split, shots in SHOTS.items():
        d = os.path.join(out, split)
        os.makedirs(d, exist_ok=True)
        for i, (name, scene, ill, ev, _) in enumerate(shots):
            raw, meta = shoot(scene(), ill, ev, seed=100 + i + (50 if split == "hidden" else 0))
            Image.fromarray(raw).save(os.path.join(d, f"{name}.raw.png"))
            json.dump(meta, open(os.path.join(d, f"{name}.json"), "w"), indent=1)
            img = camera.process(raw.astype(np.float64), meta)
            Image.fromarray(img).save(os.path.join(d, f"{name}.out.png"))
            sat = (raw >= WHITE).mean()
            print(f"{split:6s} {name:15s} {raw.shape} {ill:11s} EV x{ev:<4} clipped {sat:6.2%} "
                  f"out mean {img.mean():5.1f} dark {np.mean(img.max(-1) < 30):5.1%}")


if __name__ == "__main__":
    main()
