# Verifier checks run locally (no agent involved)

All run 2026-10-03 on macOS arm64, Docker 29.5 on Colima, Harbor 0.23.1.dev202609170426.

| Check | Command | Result |
|---|---|---|
| Static checks | `for c in scripts/checks/check-*.sh; do bash $c tasks/camera-pipeline-reconstruction; done` | 26/26 pass ([logs](checks/static/)) |
| Docker build + oracle | `harbor run -p tasks/camera-pipeline-reconstruction --agent oracle --env docker` | reward 1.0; hidden photos 99.67 / 96.44 / 99.996 / 99.998 % within ±2, 100 % within ±8 ([output](execution/oracle/verifier/test-stdout.txt)) |
| Nop | `harbor run -p tasks/camera-pipeline-reconstruction --agent nop --env docker` | reward 0.0; pipeline missing, 5 of 6 tests fail ([output](execution/nop/verifier/test-stdout.txt)) |
| Sandbox: cheating pipeline | verifier image, `/app/camera/pipeline.py` that tries to copy `/tests/**/*.out.png` and write `/logs/verifier/reward.txt` | reward 0; `/tests` unreadable as `nobody`, reward write `Permission denied` |
| Fable 5.1's probe submission in the real verifier | verifier image with Fable's `pipeline.py` mounted at `/app/camera` | reward 0; same scores as the probe scorer (e.g. h4_texture 84.899 % / 96.951 %), 4 of 4 hidden photos fail |

Commands for the last two (from the repo root, after `docker build -t cam-verifier tasks/camera-pipeline-reconstruction/tests`):

```bash
docker run --rm -v "$PWD/<dir-with-pipeline>":/app/camera cam-verifier bash -c '/tests/test.sh; cat /logs/verifier/reward.txt'
```
