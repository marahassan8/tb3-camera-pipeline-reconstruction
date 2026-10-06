# Claude Fable 5.1 probe — files as run

| File | What |
|---|---|
| `PROMPT.md` | The prompt given to the agent (the task instruction, with workspace-relative paths) |
| `launch_probe.sh`, `resume_probe.sh`, `score_probe.sh` | The scripts as run during development. They expect the development layout (`../prototype/data`), whose contents are now in `tools/camera-pipeline/` and the task folder |
| `probe_fable.json`, `probe_fable_resume.json` | Claude Code's JSON output for session 1 and the resumed session 2 (cost, turns, duration, final message) |
| `probe_score.txt` | Scores of the submitted `pipeline.py` on the training and hidden photos |
| `fable_submitted_pipeline.py` | The agent's final `pipeline.py` |

The agent's working files and transcript are not included.
