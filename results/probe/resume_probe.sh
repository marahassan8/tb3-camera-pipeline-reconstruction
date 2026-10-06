#!/usr/bin/env bash
# Resume the Fable probe session that ended while waiting on its own background job, then re-score.
# The message is neutral: it only explains that background jobs won't wake the agent in this mode.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SID=$(python3 -c "import json;print(json.load(open('$HERE/probe_fable.json'))['session_id'])")
MSG="Your background joint fit has finished (see work/jointfit_base.log). In this session nothing re-invokes you when background jobs or monitors complete, so keep working in the foreground until pipeline.py is complete and you are satisfied with it."
echo "resuming session $SID at $(date '+%H:%M')"
( cd "$HOME/cam-probe" && CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 caffeinate -i claude -p --resume "$SID" "$MSG" \
    --model claude-fable-5-1 --effort max --dangerously-skip-permissions --output-format json \
    > "$HERE/probe_fable_resume.json" 2> "$HERE/probe_fable_resume.err" ) || echo "claude exited with an error"
echo "agent finished $(date '+%H:%M')"
"$HERE/launch_probe.sh" score
