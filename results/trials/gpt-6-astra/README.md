# Standard trials: Codex / GPT-6 Astra, xhigh effort

Three independent standard Harbor trials of `camera-pipeline-reconstruction` with Codex and GPT-6 Astra (`gpt-6-astra`) at `xhigh` reasoning effort, the current Terminal-Bench CI configuration. **All three completed and failed (reward 0).**

## Results

| Trial | Reward | Hidden photos, % within ±2 (Hubble / saturated / motorcycle / texture) | Worst | Training photos, % within ±2 | Agent time | Cost |
|---|---|---|---|---|---|---|
| [1](trial-1/) | **0** | 96.7 / **44.8** / 90.1 / 89.1 | 44.8% | 98.7 / 96.9 / 98.2 / 99.4 | 176 min, 265 commands | 232.5 of 750 combined ChatGPT credits (Harbor estimate: $42.13) |
| [2](trial-2/) | 0 | 96.6 / **44.8** / 90.0 / 88.8 | 44.8% | 98.7 / 96.9 / 98.3 / 99.4 | 170 min, 247 commands | 258.2 of 750 combined ChatGPT credits (Harbor estimate: $39.84) |
| [3](trial-3/) | 0 | 96.6 / **44.7** / 90.2 / 89.3 | 44.7% | 98.7 / 97.0 / 98.2 / 99.4 | 174 min, 281 commands | 259.4 of 750 combined ChatGPT credits (Harbor estimate: $44.76) |

The three run windows overlap and use one combined 750-credit total. The shared five-hour plan allowance is counted once for the group, not once per row.

Each run has the same retained artifact layout: the submitted `pipeline.py` and `profile.npz`, a run summary, CTRF output, verifier stdout, reward and the agent's final report. The implementations use different helper structure, spline chunking, array operations and sharpening parameters. The three submissions have distinct hashes and were graded independently by Harbor's separate verifier.

- **Pass mark:** on every hidden photo, at least 95% of pixels within 2 levels and 99.5% within 8.
- **Within ±8 on the hidden photos:** trial 1 99.662 / 60.323 / 95.837 / 97.594%; trial 2 99.662 / 60.327 / 95.838 / 97.600%; trial 3 99.664 / 60.309 / 95.832 / 97.589%.
- **Training photos** are in the order skin / astronaut / coffee / chart. They were scored by running each submitted pipeline on the training pairs. Minimum within ±2 was 96.95%, 96.85% and 97.00% for trials 1–3; minimum within ±8 was 99.66% for each.
- **Settings:** the Harbor run records and Codex session logs confirm `gpt-6-astra` with `model_reasoning_effort=xhigh`.

## What the agents did

GPT-6 Astra reverse-engineered much of the pipeline methodically:
- **Demosaic:** *"Local noise patterns confirm Malvar demosaicing with white balance applied first."* Correct.
- **Sharpening:** it found two-component luma sharpening with a soft threshold (*"suppress small details before boosting larger ones"*), close to the camera's unsharp mask.
- **Tone curve:** read from the neutral grey ramp.

It did **not** identify the edge-aware local tone map, or that the colour stage is a 3×3 matrix. Instead it fitted a flexible colour profile directly to the rendered outputs: a 2.2 MB `profile.npz` holding B-spline lookup tables. That fits the training photos better than any other agent's model (all four pass), but it can't extrapolate to colours the training photos don't contain. Hence 44.8% on the saturated chart.

The runs recognized the risk that fitted profiles might not extrapolate, and each final report states that accuracy on unseen photos remained unverified. See the final reports for [trial 1](trial-1/agent_final_report.md), [trial 2](trial-2/agent_final_report.md), and [trial 3](trial-3/agent_final_report.md).

**Analysis criteria** (from [trial-analysis.toml](../../../configs/trial-analysis.toml), assessed by hand):

| Criterion | Result |
|---|---|
| Difficulty crux | Failed for the intended reason: the local tone map was never modelled, and a fitted colour table stood in for the matrix and did not extrapolate. |
| Near miss | No. The saturated chart misses by 50 points within ±2 (60% within ±8); the Hubble photo passes. |
| Low timeout | **Borderline.** The agents stopped on their own after 170–176 of 180 minutes, with final reports and no timeout. |
| Reward hacking / refusals | None seen: the agent worked only in `/app`. |

## How the trials were run

- **All three** were standard Harbor trials, produced with:
  ```bash
  harbor run -p tasks/camera-pipeline-reconstruction --agent codex --model openai/gpt-6-astra \
    --ak reasoning_effort=xhigh -k 1 --env docker --yes --agent-setup-timeout-multiplier 4
  ```
  It authenticated with a ChatGPT Plus login (`CODEX_FORCE_AUTH_JSON`), following the brief's sample command. The script is [`run_codex_trial.sh`](../../../run_codex_trial.sh).
- The runs used separate task containers and separate Codex sessions. Harbor copied each `/app/camera` artifact into a fresh verifier container for scoring.
- **Other attempts, not counted:**
  - one start cancelled during setup;
  - one failed on an expired login (HTTP 401 before any work);
  - one reduced-time (1-hour) variant, stopped by us after 10 minutes.

  None produced a submission.
- **Agent transcripts are not included,** for any trial.
