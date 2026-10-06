# Standard trials: Claude Code / Opus 5.5, max effort

Three standard trials of `camera-pipeline-reconstruction` with Claude Opus 5.5 (`claude-opus-5-5`) at max reasoning effort, run on 2026-10-05/06. **All three failed (reward 0).** Each finished well inside the 3-hour limit with no intervention.

## Results

| Trial | Reward | Hidden photos, % within ±2 (Hubble / saturated / motorcycle / texture) | Worst | Training photos, % within ±2 | Task turn | Thinking tokens | Cost |
|---|---|---|---|---|---|---|---|
| [1](trial-1/) | 0 | 74.2 / **37.9** / 76.2 / 80.5 | 37.9% | 97.2 / 75.5 / 93.2 / 96.3 | 143 min, 109 turns | 67k | $7.52 |
| [2](trial-2/) | 0 | 83.7 / **69.0** / 77.2 / 71.6 | 69.0% | 93.1 / 82.5 / 82.5 / 92.6 | 136 min, 192 turns | 244k | $24.43 |
| [3](trial-3/) | 0 | 93.9 / **75.0** / 92.7 / 85.3 | 75.0% | 96.0 / 95.0 / 96.3 / 95.1 | 126 min, 270 turns | 237k | $32.06 |

- **Pass mark:** on every hidden photo, at least 95% of pixels within 2 levels and 99.5% within 8.
- **Within ±8 on the hidden photos:** trial 1 97.4 / 73.5 / 90.6 / 95.6%; trial 2 97.8 / 80.5 / 94.7 / 92.1%; trial 3 99.8 / 87.8 / 98.9 / 96.9%.
- **Training photos** are in the order skin / astronaut / coffee / chart. Each score matches the agent's own report, which confirms the submission was transferred exactly.
- **Costs** are Harbor's list-price estimates for the agent execution.

**Effort.** All three ran with Harbor's `reasoning_effort=max` agent setting. Trial 1 was the leanest run: 109 turns and 67k thinking tokens, against 192–270 turns and 237–244k for trials 2 and 3.

## What each agent did

All three reconstructed black level, as-shot white balance, Malvar–He–Cutler demosaicing and a 3×3 colour matrix. Trials 1 and 3 confirmed the demosaic from how sensor noise propagates; trial 2 from its fitted constants matching the textbook kernel. **None reproduced the edge-aware local tone map**, the stage the task is built around.

- **Trial 1** noticed the local effect: "fine detail passes through with about 1.37× the gain of broad areas". It modelled this as a gain driven by a blurred brightness map. It then patched the remaining error with a fitted 3D colour lookup table and a small residual CNN, trained with PyTorch on the four photos and run in NumPy. Its own hold-out tests were mixed, and the fitted corrections did not carry over to the hidden photos (37.9% on the saturated chart). [Report](trial-1/agent_final_report.md), [pipeline](trial-1/submission/pipeline.py).
- **Trial 2** also noticed it: "each pixel's brightening depends on the brightness of its immediate neighbours (about a 1-pixel blur radius), with no halos at edges". It modelled this as a Gaussian of log-luminance plus a shared 138-knot tone curve. It tried edge-aware blurs and median filters without finding one that fitted. It found the highlight blend to neutral (starting at 0.80) but has no sharpening stage. [Report](trial-2/agent_final_report.md), [pipeline](trial-2/submission/pipeline.py).
- **Trial 3** missed it, as Fable 5.1 did, modelling tone globally: a luminance exponent of 0.69, a Hable filmic curve, then sRGB. It found the smoothstep highlight blend (starting at 0.78; the camera's starts at 0.80) and the luma unsharp mask (σ 1.0, amount 0.55; the camera's is σ 1.0, amount 0.6). It reached 95% on every training photo and still failed every hidden one, worst on the saturated chart (75.0%). [Report](trial-3/agent_final_report.md), [pipeline](trial-3/submission/pipeline.py).

**Analysis criteria** (from [trial-analysis.toml](../../../configs/trial-analysis.toml), assessed by hand):

| Criterion | Result |
|---|---|
| Difficulty crux | All three failed for the intended reason: the local tone map was missed or approximated, and fitted stand-ins did not extrapolate. |
| Near miss | No. The saturated chart misses by 20–57 points within ±2, with 73–88% within ±8. Trial 3 came closest elsewhere (Hubble 93.9%, motorcycle 92.7%). |
| Low timeout | No. The runs finished with 37–54 minutes to spare. |
| Reward hacking / refusals | No sign of either: no web searches were recorded, and each agent worked in its own `/app`. |

## How the trials were run

All three are real, independent Harbor container trials.

- **Agent:** Harbor's `claude-code` agent with Claude Code 2.1.289, model `anthropic/claude-opus-5-5`, and `reasoning_effort=max`.
- **Environment:** a fresh Docker task container for every run, limited by `task.toml` to 2 CPUs and 4 GB. Python 3.13.14 with numpy 2.5.3, scipy 1.18.1, Pillow 12.3.0 and scikit-image 0.26.0 was on `PATH`. Only the training pairs were available under `/app/train`; the hidden inputs and solution were not present in the agent container.
- **Prompt and limit:** Harbor sent the task's [`instruction.md`](../../../tasks/camera-pipeline-reconstruction/instruction.md) with its canary stripped and enforced the 10,800-second agent timeout. No manual follow-up or resume messages were used.
- **Grading:** Harbor transferred `/app/camera` unchanged to a fresh separate-verifier container and ran `/tests/test.sh`. The retained submission hashes, CTRF results, verifier stdout and reward are under each trial directory. [score_train.py](score_train.py) applies the same metric to the training photos for reference.

The trials were launched with `run_claude_trials.sh`, whose standard-run command uses:

```bash
harbor run -p tasks/camera-pipeline-reconstruction \
  --agent claude-code --model anthropic/claude-opus-5-5 \
  --ak reasoning_effort=max --ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 \
  --env docker --yes -k 3 --n-concurrent 1
```

To re-score a submission: `docker build -t cam-verifier tasks/camera-pipeline-reconstruction/tests`, then `docker run --rm -v "$PWD/results/trials/opus-5.5/trial-3/submission":/app/camera cam-verifier bash -c '/tests/test.sh; cat /logs/verifier/reward.txt'`.
