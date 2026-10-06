# Camera Pipeline Reconstruction — a Terminal-Bench 3 task

**Author:** Mara Hassan (GitHub `marahassan8`)

**Task:** [`tasks/camera-pipeline-reconstruction/`](tasks/camera-pipeline-reconstruction/) (Harbor task format, separate-verifier mode)

**Built against:** `harbor-framework/terminal-bench` at commit `1dcda871`, Harbor `0.23.1.dev202609170426`

## Summary

A camera turns each RAW sensor capture into a finished photo through a chain of processing stages. The agent gets four RAW photos and the camera's finished images of them. It must write a program that reproduces the camera's processing on **photos it has never seen**: very saturated colours, large clipped highlights, deep shadows, dense fine texture. Pass mark: on every hidden photo, at least 95% of pixels within 2 levels and 99.5% within 8.

Fitting the mapping fails on the unseen photos:
- a 3D colour lookup table reaches 32% on the worst hidden photo;
- a neural network trained on the pairs reaches 0.7%.

Only identifying the actual stages extrapolates. A blind reconstruction from the four training pairs passes every hidden photo (worst 96.4%).

**Claude Code with Claude Fable 5.1 (the current CI default) failed, for the intended reason.**
- **What it got right:** it identified the demosaic method, white balance, colour matrix, highlight handling and sharpening, with near-exact parameters.
- **What it missed:** the edge-aware local tone-mapping stage. It modelled that stage as a global saturation boost and blamed the remaining errors on noise and demosaic ringing.
- **Result:** 66–91% on the hidden photos, failing all four.

**Claude Code with Claude Opus 5.5 at max effort failed all three runs, for the same reason** (§4.3). Each reconstructed the demosaic, white balance and colour matrix; none reproduced the edge-aware local tone map. On the worst hidden photo they reached 37.9, 69.0 and 75.0% within ±2. All three were standard Harbor container trials.

**Codex with GPT-6 Astra at xhigh (the current CI default for Codex) failed all three runs, also for the intended reason** (§4.4). All three were independent standard Harbor trials. They identified the demosaic and sharpening, then fitted flexible colour profiles instead of the colour matrix and local tone map. The profiles passed all four training photos but reached only 44.7–44.8% on the saturated hidden chart.

| Requirement | Status |
|---|---|
| Static checks (26 scripts in `scripts/checks/`) | ✅ 26/26 pass ([logs](results/checks/static/)) |
| Docker build (agent and verifier images) | ✅ |
| Oracle | ✅ reward 1.0; hidden photos 99.67 / 96.44 / 99.996 / 99.998 % within ±2 ([output](results/execution/oracle/verifier/test-stdout.txt)) |
| Nop | ✅ reward 0.0 ([output](results/execution/nop/verifier/test-stdout.txt)) |
| Anti-cheat (local exploit test) | ✅ a pipeline that tries to read the answers or write the reward scores 0 ([record](results/verifier-checks.md)) |
| Standard trials, Claude Code / Fable 5.1 (max) | ⚠️ **1 run, failed** all 4 hidden photos (66.1–90.9% within ±2 vs 95%). It ran as a probe through the Claude Code CLI, not through Harbor, and was resumed once (§4.2). The other 2 runs were **not run** (cost) |
| Standard trials, Claude Code / Opus 5.5 (max) | ⚠️ **3 Harbor runs, all failed** all 4 hidden photos (worst photo 37.9 / 69.0 / 75.0% within ±2 vs 95%), each finishing in 126–143 min with no intervention (§4.3) |
| Standard trials, Codex / GPT-6 Astra (xhigh) | ⚠️ **3 Harbor runs, all failed** 3 of 4 hidden photos (worst 44.8 / 44.8 / 44.7% within ±2 vs 95%), each finishing in 170–176 min (§4.4) |
| Cheat trials | ❌ **not run** (cost). Anti-cheat rests on the design and the local exploit test |
| Implementation rubric review (`harbor check`) | ❌ **not run** (cost). Command under *Reproducing* |

**What this does and doesn't show.** It shows a working, calibrated task that every agent tried failed, every time for the intended reason: Fable 5.1 (one probe), Opus 5.5 (three Harbor runs) and GPT-6 Astra (three Harbor runs). The full three-run standard matrix for Opus and Astra is present; the Fable evidence remains a resumed host probe, and the cheat runs were not done. The commands for the Harbor runs are under *Reproducing*.

An earlier candidate, a churn-forecasting task, did get the complete trial matrix (6 Claude Code runs and both cheat runs). The agents passed all six runs, which is why it was replaced by this task.

## 1. The task

**What the agent sees** (`/app/train/`): four photos from one camera. For each:

| File | Content |
|---|---|
| `NAME.raw.png` | the RAW capture: 16-bit RGGB Bayer mosaic, one value per photosite |
| `NAME.json` | black level, white level, CFA layout, as-shot white-balance gains |
| `NAME.out.png` | the 8-bit sRGB image the camera's own processing produced |

**What it must produce:** `/app/camera/pipeline.py`, run as `python3 pipeline.py IN.raw.png IN.json OUT.png`. It is graded on four other RAW photos from the same camera, of different scenes, lighting and exposure. On each, ≥95% of pixels must be within 2 levels on every channel and ≥99.5% within 8, ignoring a 16-px border, within 60 s per photo. The runtime is Python 3.13 with numpy, scipy, Pillow and scikit-image only. [Instruction](tasks/camera-pipeline-reconstruction/instruction.md).

**The hidden camera** (ground truth, never visible to the agent). Every stage is deterministic, closed-form and standard:
1. black/white level;
2. white balance with a small fixed tweak on top of the as-shot gains;
3. Malvar–He–Cutler demosaicing;
4. highlight clip with a smoothstep blend to neutral;
5. a 3×3 colour matrix;
6. **edge-aware local tone mapping**: guided filter on log luminance, base layer compressed, detail kept;
7. a filmic tone curve and the sRGB transfer function;
8. a luma unsharp mask with a soft threshold.

**Why it is hard, and why fairly so**
- **Fitting doesn't extrapolate.** Each training pair holds hundreds of thousands of pixels, so fitting the mapping looks promising. But the hidden photos push the same stages into regions the training pairs barely cover. A correct stage extrapolates (a colour matrix measured on moderate colours is exact on saturated ones); a fitted mapping guesses.
- **The crux is the local tone map.** On the training photos its effect resembles a global contrast and saturation change, so a global model fits them moderately well (85–96%) and then fails on scenes with different spatial structure.
- **Fairness rule.** Every mechanism leaves a measurable trace in the four training pairs: clipped highlights under two illuminants, colours saturated enough to pin the matrix, sharp edges, shadows. The hidden photos only push those mechanisms harder. The blind reconstruction (§2) proves the information is there. The instruction says openly that grading uses unseen photos with different scenes, lighting and exposure.
- **Real-world value.** Reproducing a camera's rendering from RAW is paid work: camera-matching profiles in raw converters, film-simulation recreations, ISP tuning.

## 2. Calibration

All controls are scored at the grading settings on the four hidden photos; the table shows the worst one ([script](tools/camera-pipeline/controls.py), [verdicts](tools/camera-pipeline/controls_verdicts.txt)).

| Candidate | Worst hidden photo, within ±2 | Verdict |
|---|---|---|
| Reference generator | 100% | pass |
| Independent float32 reimplementation (different filters and borders) | 100% | pass |
| **Blind reconstruction from the 4 training pairs** ([recover.py](tools/camera-pipeline/recover.py), [log](tools/camera-pipeline/recover.log)); also the oracle | **96.44%** | **pass** |
| Small fitting errors: gain, tone white point, LTM radius/ε/compression, sharpening amount/σ/threshold | 95.0–100% | pass |
| White-balance tweak off by 1% / highlight-blend start off by 0.05 / colour-matrix off-diagonals off by 5% | 89.8% / 84.4% / 80.4% | fail |
| Any single stage missing or replaced (bilinear demosaic, no highlight blend, no LTM, no sharpening, no WB tweak, identity matrix) | 1.1–79.2% | fail |
| 33³ colour lookup table fitted on the training pairs (training: 68–86%) | 32.1% | fail |
| Degree-3 polynomial colour regression | 2.2% | fail |
| U-Net trained 25 min on CPU on the training pairs (training: 8–31%) | 0.7% | fail |
| **Claude Fable 5.1's submission** (§4.2), re-scored in the real verifier | **66.1%** | **fail** |
| **Claude Opus 5.5's submissions** (§4.3), runs 1 / 2 / 3, in the real verifier | **37.9 / 69.0 / 75.0%** | **fail** |

**How the blind reconstruction works.** It chose its structure from candidates by training error alone:
- Malvar demosaic (MSE 3.2) over bilinear (~160);
- a guided-filter LTM (0.14) over a Gaussian one (3.2).

It then fitted all parameters jointly through a differentiable model. The tone curve is free-form (48 knots), not the generator's formula. It recovered the generator's parameters almost exactly (white-balance tweak 1.0298 vs 1.03; colour matrix to 3 decimals), and never read the generator or the hidden photos.

## 3. Verification and anti-cheat

- **The hidden RAW photos and the camera's images of them exist only in the verifier image,** under `/tests`, readable by root only (`chmod go-rwx`).
- **The agent's pipeline runs sandboxed.** Each hidden photo is copied to a fresh directory, and the agent's `pipeline.py` runs there as user `nobody`, with a 60 s timeout and a 3 GB address-space limit. `/logs/verifier` isn't writable by `nobody`, and root computes the reward.
- **Tests** ([test_outputs.py](tasks/camera-pipeline-reconstruction/tests/test_outputs.py)): the pipeline exists; the hidden set is intact; each of the four hidden photos meets both thresholds. The reward is 1 only if all pass.
- **Local exploit test.** A `pipeline.py` that tries to copy `/tests/**/*.out.png` and write `reward.txt` scored 0: `/tests` was unreadable and the reward write got `Permission denied` ([record](results/verifier-checks.md)).
- **Determinism.** The generator is float64 and the truth images are precomputed. Two correct implementations agree on 100% of pixels within ±2, so the tolerance absorbs implementation differences, not wrong stages.

## 4. Results

### 4.1 Automated checks

| Check | Command | Result |
|---|---|---|
| Static checks | `for c in scripts/checks/check-*.sh; do bash $c tasks/camera-pipeline-reconstruction; done` (Python ≥ 3.11 on `PATH`) | 26/26 |
| Docker build + oracle | `harbor run -p tasks/camera-pipeline-reconstruction --agent oracle --env docker` | reward 1.0 |
| Nop | `harbor run -p tasks/camera-pipeline-reconstruction --agent nop --env docker` | reward 0.0 |
| Exploit test, Fable re-score | verifier image + mounted `/app/camera` ([record](results/verifier-checks.md)) | 0 / 0 |
| Rubric review | `configs/run_trials.sh checks` | not run |

Environment: macOS arm64, Docker 29.5 on Colima (6 CPU, 10 GB).

### 4.2 Agent evidence: one Claude Fable 5.1 run

**Configuration.** Claude Code CLI with `--model claude-fable-5-1 --effort max` and `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, the CI default model and settings. It ran in a clean workspace holding only the four training pairs, with the same instruction text apart from paths. The run was a **probe on the host**, not a Harbor trial ([prompt](results/probe/PROMPT.md), [launcher](results/probe/launch_probe.sh)).

**The run.**
- **Session 1 (60 min, $21.72).** The agent started a long background fit and ended its turn expecting to be re-invoked when it finished. Non-interactive Claude Code doesn't do that, so the session ended with placeholder parameters.
- **Resume.** It was resumed once ([script](results/probe/resume_probe.sh)) with a neutral message: the fit had finished, and background jobs would not wake it.
- **Session 2 (82 min, $34.96).** It finished the pipeline.
- **Totals:** 142 min, 158 turns, $56.68 ([session 1](results/probe/probe_fable.json), [session 2](results/probe/probe_fable_resume.json)). The CLI's `total_cost_usd` after a resume includes the earlier session, so session 2's own cost is $56.68 − $21.72.

Under a real Harbor trial, session 1's early stop would itself have counted as a failure. The resume was given so the result reflects capability rather than the CLI's behaviour.

**Score** ([scorer output](results/probe/probe_score.txt)). Re-scoring its submitted `pipeline.py` in the real verifier gives the same numbers.

| Photo | Within ±2 | Within ±8 | Pass mark 95 / 99.5 |
|---|---|---|---|
| training: skin / astronaut / coffee / chart | 95.9 / 92.7 / 96.4 / 94.6% | 99.6 / 99.3 / 99.8 / 98.7% | 2 of 4 pass |
| hidden: deep-shadow starfield | 90.9% | 99.7% | fail |
| hidden: saturated chart | 66.1% | 80.8% | fail |
| hidden: overexposed motorcycle | 87.9% | 98.8% | fail |
| hidden: cat + zone plate (texture) | 84.9% | 97.0% | fail |

**Failure analysis.** This is drawn from its [submitted pipeline](results/probe/fable_submitted_pipeline.py) and its final report.
- **What it got right** (reverse-engineered stage by stage):
  - black level and white balance on the mosaic;
  - **Malvar–He–Cutler demosaicing**, identified from the data;
  - the colour matrix;
  - a smoothstep highlight blend (start 0.78 vs the true 0.80);
  - a luma unsharp mask (σ 1.01 vs 1.0; amount 0.55 vs 0.6).
- **What it missed:** the local tone-mapping stage entirely. It explained that stage's effect as a global "luminance-preserving saturation boost, exponent 1.42" plus a free tone curve. The global model fits the training photos to 93–96% and extrapolates worst where scene structure differs most: 66% on the saturated chart.
- **How it explained its remaining errors:** as "demosaic ringing" on the chart and "textured highlights and noisy dark areas" on the astronaut. It did not consider a missing spatial stage. It reported honestly that two training photos were below target and named saturated scenes as the main risk.
- **Is this conceptual or a precision miss?** Conceptual. Its pipeline lacks a stage, and the shortfall is large on the saturated chart (66% within ±2, 81% within ±8). On three photos it is moderate (85–91% within ±2), and the starfield passes the ±8 check, so a reviewer may weigh those three as closer misses.

This matches the pattern we found in Fable's own leaderboard failures (§5). It doesn't lack skill; it builds a self-consistent model and explains contradicting evidence away.

### 4.3 Agent evidence: three Claude Opus 5.5 runs

**Configuration.** Three standard Harbor trials using `--agent claude-code --model anthropic/claude-opus-5-5 --ak reasoning_effort=max --env docker`. Each trial ran in a fresh 2-CPU, 4-GB task container with only the task's four training pairs available; the hidden photos and solution remained confined to the separate verifier container.
- **Environment:** Python 3.13 with the pinned packages was on `PATH` in the task container.
- **Prompt:** the instruction exactly as Harbor sends it, with a 3-hour limit.
- **Grading:** Harbor passed `/app/camera` to the task's separate verifier image unchanged.

The commands, configuration, submissions, verifier output and run summaries are in [results/trials/opus-5.5/](results/trials/opus-5.5/).

| Run | Hidden photos, within ±2 (Hubble / saturated / motorcycle / texture) | Training photos, within ±2 | Task turn | Cost |
|---|---|---|---|---|
| 1 | 74.2 / 37.9 / 76.2 / 80.5% | 97.2 / 75.5 / 93.2 / 96.3% | 143 min | $7.52 |
| 2 | 83.7 / 69.0 / 77.2 / 71.6% | 93.1 / 82.5 / 82.5 / 92.6% | 136 min | $24.43 |
| 3 | 93.9 / 75.0 / 92.7 / 85.3% | 96.0 / 95.0 / 96.3 / 95.1% | 126 min | $32.06 |

All three scored reward 0, failing all four hidden photos, and finished with 37–54 minutes to spare.

**Failure analysis.**
- **What all three got right:** black level, as-shot white balance, Malvar–He–Cutler demosaicing and the colour matrix. Runs 2 and 3 also found the highlight blend to neutral. Run 3 found the luma unsharp mask (σ 1.0, amount 0.55; the camera's is σ 1.0, amount 0.6).
- **What all three missed: the edge-aware local tone map.**
  - **Run 3** modelled tone globally (a luminance exponent of 0.69, then a Hable filmic curve), as Fable did. It reached 95% on every training photo and still failed every hidden one.
  - **Run 1** noticed a local effect: "fine detail passes through with about 1.37× the gain of broad areas". It modelled it with a blurred-brightness gain, then patched the rest with a lookup table and a small CNN fitted on the four photos. Those did not carry over: 37.9% on the saturated chart.
  - **Run 2** also noticed it: "brightening depends on the brightness of its immediate neighbours ... with no halos at edges". It modelled it with a Gaussian of log-luminance and tried edge-aware blurs without finding one that fitted.
- **Is this conceptual or a precision miss?** Conceptual. On the saturated chart every run misses by 20–57 points within ±2, and only 73–88% of its pixels are within ±8. Run 3 came closest on the other photos (Hubble 93.9%, motorcycle 92.7%).

### 4.4 Agent evidence: Codex / GPT-6 Astra

**Configuration.** A standard Harbor trial: `--agent codex --model openai/gpt-6-astra --ak reasoning_effort=xhigh`, signed in with a ChatGPT Plus login as in the brief's sample command. The Codex session log records `gpt-6-astra` and `xhigh` on every turn. Full record: [results/trials/gpt-6-astra/](results/trials/gpt-6-astra/).

| Run | Hidden photos, within ±2 (Hubble / saturated / motorcycle / texture) | Training photos, within ±2 | Agent time | Usage |
|---|---|---|---|---|
| 1 | 96.7 / **44.8** / 90.1 / 89.1% (reward 0) | 98.7 / 96.9 / 98.2 / 99.4% | 176 min | 232.5 of 750 combined credits (Harbor estimate $42.13) |
| 2 | 96.6 / **44.8** / 90.0 / 88.8% (reward 0) | 98.7 / 96.9 / 98.3 / 99.4% | 170 min | 258.2 of 750 combined credits (Harbor estimate $39.84) |
| 3 | 96.6 / **44.7** / 90.2 / 89.3% (reward 0) | 98.7 / 97.0 / 98.2 / 99.4% | 174 min | 259.4 of 750 combined credits (Harbor estimate $44.76) |

The three runs overlapped, shared one five-hour plan allowance, and used 750 paid credits in total.

**Failure analysis.**
- **What it got right:** Malvar demosaicing with white balance applied first, two-component luma sharpening with a soft threshold, and the tone curve from the grey ramp.
- **What it missed:** the edge-aware local tone map and the 3×3 colour matrix. In their place it fitted a 2.2 MB B-spline colour profile directly to the training outputs. That fits the training photos better than any other agent's model (all four pass), but it cannot extrapolate to colours the training photos lack, hence 44.8% on the saturated chart (60.3% within ±8).
- **It saw the risk:** *"the fitted table can darken saturated colors as exposure rises"*. It ended with *"Accuracy on unseen photos remains unverified."*
- **Timing:** all three stopped on their own with 4–10 minutes remaining. The roughly 50-point miss on the saturated chart is far larger than any time pressure explains.

## 5. How the idea was chosen

We treated "find a task frontier agents genuinely fail" as an empirical question and changed method four times.

**Phase 1: design from first principles, then probe.** About a dozen ideas were prototyped with a hidden verifier and probed with Fable 5.1, and with Opus 5.5 where useful. They covered:
- a crash-safe key-value store;
- a billing engine graded on 300 hidden scenarios;
- test-impact selection;
- PromQL alerting rules;
- geospatial zonal statistics;
- impedance-spectroscopy fitting.

Every probed idea was solved, mostly in 8–25 minutes. Fable typically built its own checker (a crash enumerator, variant repositories, a brute-force reference) and verified itself against it.

**Phase 2: a full task with real trials.** We built subscription churn forecasting, which tests point-in-time feature leakage plus a contract-renewal wave. All six Claude Code trials passed it (Fable 5.1 and Opus 5.5, AUC 0.883–0.889 against a 0.86 pass mark). Both agents found both traps.

**Phase 3: mine what agents actually fail.** We analysed the Harbor Hub results of all 13 Terminal-Bench 4.0 leaderboard models (4,290 runs), and the run logs of Fable's failed runs on the tasks it never solves. Three findings stood out:
1. **Fable always builds its own checks.**
2. **Those checks share its interpretation.** When the interpretation is wrong, it passes its own tests, fails the hidden ones, and argues away contradicting evidence.
3. **Precisely specified targets get self-verified.** Three micro-probes built around requirements in tension were all solved.

**Phase 4: target that finding directly.** The task needed a definition of correct that isn't written anywhere the agent can check itself against, but that is fully recoverable from evidence. A hidden generator, observed only through its outputs and graded on unseen inputs, provides exactly that. Before building, the camera pipeline had to pass four gates:
1. generic fits fail;
2. a blind reconstruction passes;
3. every wrong stage fails;
4. Fable 5.1 fails for the predicted reason.

It passed all four (§2, §4.2).

## 6. Limitations

- **The standard trial matrix is complete for Opus 5.5 and GPT-6 Astra:** three independent Harbor container runs each. Fable 5.1 still has one resumed host probe rather than three Harbor trials. Cheat trials and the rubric review were not run.
- **Seven completed runs across three models are not evidence that every stronger run must fail.** All seven failed for the same reason, with large gaps on the saturated chart and moderate ones elsewhere (Opus run 3 reached 93.9% on the Hubble photo). A future run could still find the local stage.
- **Agent timeout.** It is set to 3 h. The Fable probe used 2 h 22 min including the resume; the Opus runs took 126–143 min; the GPT-6 Astra runs took 170–176 min.
- **The data is synthetic.** The scenes are real photos turned back into RAW (following Brooks et al. 2019, *Unprocessing Images for Learned Raw Denoising*) plus procedural charts, and the camera is a constructed pipeline rather than a real camera's firmware.
- **Licences.** The source photos are scikit-image sample images: astronaut, Hubble deep field and skin are public domain; coffee and cat are CC0. The Middlebury stereo motorcycle and scikit-image's colour wheel carry no stated licence, so they should be replaced before any upstream contribution.

## 7. Repository layout

| Path | What |
|---|---|
| `tasks/camera-pipeline-reconstruction/` | **The task**: instruction, agent environment with the 4 training pairs, oracle (blind reconstruction), separate verifier with the 4 hidden photos |
| `results/` | Static-check logs, oracle/nop outputs, exploit test, Fable 5.1 probe (prompt, scripts, run summaries, submitted pipeline), Opus 5.5 and GPT-6 Astra trials (run records, submissions, verifier outputs, agents' final reports; no agent transcripts) |
| `tools/camera-pipeline/` | Generator, data builder, independent implementation, controls, generic baselines, blind reconstruction |
| `run_codex_trial.sh`, `configs/pausing_codex.py` | Codex trial runner (standard or cheat; optional pausing mode that waits for ChatGPT-plan resets) |
| `run_claude_trials.sh`, `configs/` | Trial runner (checks, rubric review, standard and cheat trials, summary), CI defaults snapshot, hack prompt, rubric, analysis prompts |

## Reproducing

```bash
# static checks (from a terminal-bench checkout, Python >= 3.11 on PATH)
for c in scripts/checks/check-*.sh; do bash "$c" tasks/camera-pipeline-reconstruction; done

# oracle and nop
harbor run -p tasks/camera-pipeline-reconstruction --agent oracle --env docker
harbor run -p tasks/camera-pipeline-reconstruction --agent nop --env docker

# re-score an Opus 5.5 submission (results/trials/opus-5.5/) in the verifier image
docker build -t cam-verifier tasks/camera-pipeline-reconstruction/tests
docker run --rm -v "$PWD/results/trials/opus-5.5/trial-3/submission":/app/camera cam-verifier bash -c '/tests/test.sh; cat /logs/verifier/reward.txt'

# Codex / GPT-6 Astra trials
./run_codex_trial.sh run 3         # three standard trials; ./run_codex_trial.sh cheat for /cheat

# Harbor Claude trials (the Opus results above were produced this way)
./run_claude_trials.sh            # defaults: this task, jobs in ~/camera-jobs, results in results/

# regenerate data and calibration (Python 3.13 with numpy, scipy, Pillow, scikit-image; torch for the CNN/recovery)
cd tools/camera-pipeline && python make_data.py && python controls.py && python baseline_lut.py && python recover.py
```

## Acknowledgements and use of AI assistance

The brief permits extensive use of LLMs and coding agents, and this work used them throughout. Claude Code (Opus 5.5) assisted with prototyping, calibration, leaderboard and run-log analysis, drafting, external research and several earlier prototypes. All text, including the task instruction and the task README explanations, was drafted with AI assistance under the author's direction. The source photos are scikit-image sample images (see *Limitations* for licences).
