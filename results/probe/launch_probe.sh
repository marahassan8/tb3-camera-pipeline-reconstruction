#!/usr/bin/env bash
# One Fable 5.1 probe run (max effort, 128k output cap as in CI) in a clean workspace ~/cam-probe,
# then score its pipeline.py on the hidden photos.
#   ./launch_probe.sh            # fresh run (wipes ~/cam-probe)
#   ./launch_probe.sh score      # only re-score the existing ~/cam-probe/pipeline.py
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HOME/cam-probe"
grep -q "__W2__" "$HERE/PROMPT.md" && { echo "PROMPT.md still has the placeholder pass mark"; exit 1; }

if [ "${1:-}" != "score" ]; then
  rm -rf "$WS" && mkdir -p "$WS/train"
  cp "$HERE"/../prototype/data/train/* "$WS/train/"
  echo "workspace ready: $WS ($(ls "$WS/train" | wc -l | tr -d ' ') files). Started $(date '+%H:%M'); expect up to ~2 h."
  ( cd "$WS" && CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 caffeinate -i claude -p "$(cat "$HERE/PROMPT.md")" \
      --model claude-fable-5-1 --effort max --dangerously-skip-permissions --output-format json \
      > "$HERE/probe_fable.json" 2> "$HERE/probe_fable.err" ) || echo "claude exited with an error (see probe_fable.err)"
  echo "agent finished $(date '+%H:%M')"
fi
W2=$(sed -n 's/.*at least \([0-9.]*\)% of pixels.*/\1/p' "$HERE/PROMPT.md")
W8=$(sed -n 's/.*and at least \([0-9.]*\)% within 8.*/\1/p' "$HERE/PROMPT.md")
W2="$W2" W8="$W8" bash "$HERE/score_probe.sh" "$WS" | tee "$HERE/probe_score.txt"
