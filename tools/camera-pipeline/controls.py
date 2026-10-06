"""Single-edit controls: the reference pipeline with one stage removed or one
parameter perturbed, scored on the training and hidden images."""

import copy
import glob

import numpy as np
from PIL import Image

import camera
import score

CCM = np.array(camera.P["ccm"])


def edit(**kw):
    p = copy.deepcopy(camera.P)
    p.update(kw)
    return p


CONTROLS = [
    ("reference", edit()),
    # stage missing or wrong
    ("bilinear demosaic", edit(demosaic="bilinear")),
    ("no highlight blend", edit(hl_start=None)),
    ("no local tone mapping", edit(ltm_compress=1.0)),
    ("no sharpening", edit(sharp_amount=0)),
    ("no WB tweak", edit(wb_tweak=(1, 1, 1))),
    ("identity colour matrix", edit(ccm=np.eye(3))),
    # fitting errors (small)
    ("CCM off-diagonals x1.05", edit(ccm=CCM + 0.05 * (CCM - np.diag(np.diag(CCM))) - 0.05 * np.diag((CCM - np.diag(np.diag(CCM))).sum(1)))),
    ("gain 1.6 -> 1.62", edit(gain=1.62)),
    ("white 4.0 -> 4.2", edit(white=4.2)),
    ("LTM radius 8 -> 7", edit(ltm_radius=7)),
    ("LTM eps 0.04 -> 0.05", edit(ltm_eps=0.05)),
    ("LTM compress 0.70 -> 0.72", edit(ltm_compress=0.72)),
    ("hl_start 0.80 -> 0.75", edit(hl_start=0.75)),
    ("sharp amount 0.6 -> 0.55", edit(sharp_amount=0.55)),
    ("sharp sigma 1.0 -> 1.1", edit(sharp_sigma=1.1)),
    ("sharp threshold 0.01 -> 0.015", edit(sharp_threshold=0.015)),
    ("WB tweak R 1.03 -> 1.02", edit(wb_tweak=(1.02, 1.0, 0.96))),
]


def images():
    for f in sorted(glob.glob("data/*/*.raw.png")):
        base = f[:-8]
        m, meta = camera.load_raw(f, base + ".json")
        yield base.split("/")[-1], m, meta, np.asarray(Image.open(base + ".out.png"))


def main():
    data = list(images())
    names = [n for n, *_ in data]
    print(f"{'control':32s}" + "".join(f"{n[:9]:>10s}" for n in names) + "   min hidden w2")
    for label, p in CONTROLS:
        s = [score.compare(camera.process(m, meta, p), t)["w2"] for _, m, meta, t in data]
        hidden = [v for n, v in zip(names, s) if n.startswith("h")]
        print(f"{label:32s}" + "".join(f"{v:10.2%}" for v in s) + f"   {min(hidden):8.2%}")


if __name__ == "__main__":
    main()
