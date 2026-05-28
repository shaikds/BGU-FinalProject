ORCHESTRATOR README
===================

What this does
---------------
`orchestrator.py` runs two existing pipeline scripts in parallel:

- `/home/shaikar/sn_pipe_trial/pipeline/run_all.sh` (uses conda env `sn_pipe` by default)
- `/home/shaikar/T-DEED-2/run_models.sh` (uses conda env `tdeed_inference2` by default)

After both finish successfully, it runs the Event-to-Player linker:
`/home/shaikar/T-DEED-2/EventToPlayer/EventToPlayerLinker.py` with the provided
tracking and events JSON files and writes a linked results JSON.

Prerequisites
-------------
- Python 3.8+ available on PATH.
- `conda` available (if you plan to use the default conda execution mode).
- The two pipeline scripts must be executable and present at the default
  locations above, or you can pass `--sn-script` / `--tdeed-script` to override.
- The conda environments `sn_pipe` and `tdeed_inference2` should be installed
  and contain the dependencies required by each pipeline.

Quick start
-----------
Run the orchestrator with only the video path (recommended):

```bash
python run_both.py --video /full/path/to/video.mp4
```
OR use:
python orchestrator.py --video /full/path/to/video.mp4

The orchestrator will:

- launch the SN pipeline (`sn_pipe` env) and the T-DEED pipeline (`tdeed_inference2` env) in parallel,
- wait for both to finish,
- then run the EventToPlayerLinker in the `tdeed_inference2` env and write `linked.json` (default).

Logs are written to `./logs_orchestrator/sn_pipe.log` and `./logs_orchestrator/tdeed.log`.

If you need to override produced file locations or other behaviour, you can pass optional flags:

```bash
python run_both.py --video /video.mp4 --output my_linked.json --no-conda
```
Or use :
python orchestrator.py --video /video.mp4 --output my_linked.json --no-conda

Customizing
-----------
- `--sn-script`: path to the SN pipeline script (default: `/home/shaikar/sn_pipe_trial/pipeline/run_all.sh`)
- `--tdeed-script`: path to the T-DEED script (default: `/home/shaikar/T-DEED-2/run_models.sh`)
- `--sn-env` / `--tdeed-env` / `--linker-env`: conda env names to use
- `--log-dir`: change logs directory (default `./logs_orchestrator`)
- `--window` / `--sigma` / `--summary`: passed to the EventToPlayerLinker

Notes
-----
- The orchestrator currently requires you to provide the final `--tracking` and
  `--events` JSON paths because pipeline output locations can vary. Point those
  to the files produced by the two pipelines (for example `reid.json` and
  `results_ensemble.json`).
- If either pipeline fails, the orchestrator prints the log file locations and
  exits with a non-zero code.

Troubleshooting
---------------
- If `conda run -n <env>` fails, ensure `conda` is on PATH and the env exists:
  `conda env list`.
- Check the per-pipeline logs under the configured log directory.

Want me to also add a small systemd unit or a shell wrapper? Open to add-ons.
