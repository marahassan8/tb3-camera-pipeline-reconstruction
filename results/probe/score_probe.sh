#!/usr/bin/env bash
# Score the agent's pipeline.py on the hidden RAW photos (and, for reference, the training ones).
# Usage: score_probe.sh [workspace]   (default ~/cam-probe)
set -u
WS="${1:-$HOME/cam-probe}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$HERE/../prototype/data"
OUT="$HERE/probe_outputs"
mkdir -p "$OUT"
[ -f "$WS/pipeline.py" ] || { echo "no pipeline.py in $WS"; exit 1; }
for split in train hidden; do
  for raw in "$DATA/$split"/*.raw.png; do
    name=$(basename "$raw" .raw.png)
    start=$(date +%s)
    ( cd "$WS" && perl -e 'alarm shift; exec @ARGV' 120 uv run -q --python 3.13 --with numpy --with scipy --with pillow --with scikit-image \
        python pipeline.py "$raw" "${raw%.raw.png}.json" "$OUT/$name.png" ) > "$OUT/$name.log" 2>&1 \
      || echo "$name: pipeline failed (see $OUT/$name.log)"
    echo "$name $(( $(date +%s) - start ))s" >> "$OUT/timing.txt"
  done
done
"$HERE/../prototype/.venv/bin/python" - "$DATA" "$OUT" "${W2:-__W2__}" "${W8:-__W8__}" <<'EOF'
import sys, glob, os
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(sys.argv[1], ".."))
import score
data, out, w2, w8 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
ok = True
for split in ("train", "hidden"):
    for f in sorted(glob.glob(f"{data}/{split}/*.out.png")):
        n = os.path.basename(f)[:-8]
        p = f"{out}/{n}.png"
        if not os.path.exists(p):
            print(f"{split:6s} {n:15s} MISSING"); ok &= split != "hidden"; continue
        s = score.compare(np.asarray(Image.open(p).convert("RGB")), np.asarray(Image.open(f)))
        passed = s["w2"] * 100 >= w2 and s["w8"] * 100 >= w8
        if split == "hidden": ok &= passed
        print(f"{split:6s} {n:15s} {score.fmt(s)}  {'pass' if passed else 'FAIL'}")
print("PROBE:", "PASSED (Fable solved it)" if ok else "FAILED on at least one hidden photo")
EOF
