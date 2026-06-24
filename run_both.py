#!/usr/bin/env python3
"""
orchestrator to run both pipelines in parallel and then link events to players.

Usage example:
  python run_both.py \
    --video /full/path/to/video.mp4 \
    --tracking /full/path/to/reid.json \
    --events /full/path/to/results_ensemble.json \
    --output linked.json

This script launches the two pipeline scripts in parallel using their
conda environments, captures logs, waits for completion, then runs
`EventToPlayerLinker.py` to produce a linked output JSON.
"""
import argparse
import subprocess
import shlex
from pathlib import Path
import sys
import os


def run_parallel(sn_script: Path, tdeed_script: Path, video: str, sn_env: str, tdeed_env: str, logdir: Path, use_conda: bool):
    logdir.mkdir(parents=True, exist_ok=True)

    sn_log = logdir / "sn_pipe.log"
    tdeed_log = logdir / "tdeed.log"
    if use_conda:
        if not sn_env:
            raise RuntimeError("sn_env is empty but --no-conda not set; provide --sn-env or use --no-conda")
        if not tdeed_env:
            raise RuntimeError("tdeed_env is empty but --no-conda not set; provide --tdeed-env or use --no-conda")

        # Use a bash login shell that sources conda.sh and activates the env,
        # then run the script. This avoids `conda run` quoting issues on some systems.
        sn_cmd_list = [
            "bash",
            "-lc",
            f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {shlex.quote(sn_env)} && bash {shlex.quote(str(sn_script))} {shlex.quote(video)}",
        ]
        tdeed_cmd_list = [
            "bash",
            "-lc",
            f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {shlex.quote(tdeed_env)} && bash {shlex.quote(str(tdeed_script))} {shlex.quote(video)}",
        ]

        print("Launching SN pipeline:", shlex.join(sn_cmd_list))
        print("Launching T-DEED pipeline:", shlex.join(tdeed_cmd_list))

        sn_proc = subprocess.Popen(sn_cmd_list, stdout=open(sn_log, "wb"), stderr=subprocess.STDOUT)
        tdeed_proc = subprocess.Popen(tdeed_cmd_list, stdout=open(tdeed_log, "wb"), stderr=subprocess.STDOUT)
    else:
        sn_cmd_list = ["bash", str(sn_script), video]
        tdeed_cmd_list = ["bash", str(tdeed_script), video]

        print("Launching SN pipeline:", shlex.join(sn_cmd_list))
        print("Launching T-DEED pipeline:", shlex.join(tdeed_cmd_list))

        sn_proc = subprocess.Popen(sn_cmd_list, stdout=open(sn_log, "wb"), stderr=subprocess.STDOUT)
        tdeed_proc = subprocess.Popen(tdeed_cmd_list, stdout=open(tdeed_log, "wb"), stderr=subprocess.STDOUT)

    print(f"SN PID={sn_proc.pid}, log={sn_log}")
    print(f"T-DEED PID={tdeed_proc.pid}, log={tdeed_log}")

    rc1 = sn_proc.wait()
    rc2 = tdeed_proc.wait()

    return rc1, rc2, sn_log, tdeed_log


def run_linker(linker_py: Path, tracking: str, events: str, output: str, window: int, sigma: float, summary: bool, use_conda: bool, env: str):
    base_cmd = f"python {shlex.quote(str(linker_py))} --tracking {shlex.quote(tracking)} --events {shlex.quote(events)} --output {shlex.quote(output)} --window {int(window)} --sigma {float(sigma)}"
    if summary:
        base_cmd += " --summary"

    if use_conda and env:
        cmd_list = [
            "bash",
            "-lc",
            f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {shlex.quote(env)} && {base_cmd}",
        ]
        print("Running EventToPlayerLinker:", shlex.join(cmd_list))
        completed = subprocess.run(cmd_list)
        return completed.returncode

    print("Running EventToPlayerLinker:", base_cmd)
    completed = subprocess.run(base_cmd, shell=True)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description="Run both pipelines in parallel and link events to players.")
    parser.add_argument("--video", required=True, help="Path to input video")
    # Optional overrides for produced files; if omitted, defaults below are used.
    parser.add_argument("--tracking", help="Optional path to tracking/reid JSON (for linker)")
    parser.add_argument("--events", help="Optional path to events JSON (for linker)")
    parser.add_argument("--output", default="linked.json", help="Output path for linked results JSON")
    parser.add_argument("--sn-script", default="/home/shaikar/sn_pipe_trial/pipeline/run_all.sh", help="Path to sn pipeline script")
    parser.add_argument("--tdeed-script", default="/home/shaikar/T-DEED-2/run_models.sh", help="Path to tdeed pipeline script")
    parser.add_argument("--sn-env", default="sn_pipe", help="Conda env name for sn pipeline")
    parser.add_argument("--tdeed-env", default="tdeed_inference2", help="Conda env name for tdeed pipeline")
    parser.add_argument("--linker-env", default="", help="Optional conda env to run EventToPlayerLinker (empty = use current env)")
    parser.add_argument("--log-dir", default="./logs_orchestrator", help="Directory to write pipeline logs")
    parser.add_argument("--no-conda", action="store_true", help="Do not use conda run; run scripts directly")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--summary", action="store_true")

    args = parser.parse_args()

    video = args.video
    sn_script = Path(args.sn_script).expanduser().resolve()
    tdeed_script = Path(args.tdeed_script).expanduser().resolve()
    linker_py = Path("/home/shaikar/T-DEED-2/EventToPlayer/EventToPlayerLinker.py")  # Linker script path
    logdir = Path(args.log_dir)
    use_conda = not args.no_conda

    # Default produced file locations (can be overridden with --tracking/--events)
    tracking_default = Path("/home/shaikar/sn_pipe_trial/outputs/reid_v2/reid_observations.json")
    events_default = Path("/home/shaikar/T-DEED-2/inference_output/results_ensemble.json")

    if not sn_script.exists():
        print(f"SN script not found: {sn_script}")
        sys.exit(2)
    if not tdeed_script.exists():
        print(f"T-DEED script not found: {tdeed_script}")
        sys.exit(2)
    if not linker_py.exists():
        print(f"EventToPlayerLinker not found: {linker_py}")
        sys.exit(2)

    rc1, rc2, sn_log, tdeed_log = run_parallel(sn_script, tdeed_script, video, args.sn_env, args.tdeed_env, logdir, use_conda)

    print(f"SN exit code: {rc1}, T-DEED exit code: {rc2}")
    if rc1 != 0 or rc2 != 0:
        print("One or both pipelines failed. Check logs:")
        print(f"  {sn_log}")
        print(f"  {tdeed_log}")
        sys.exit(3)

    # Both succeeded — run linker (in T-DEED env by default)
    tracking_path = args.tracking if args.tracking else str(tracking_default)
    events_path = args.events if args.events else str(events_default)

    if not Path(tracking_path).exists():
        print(f"Warning: expected tracking file not found: {tracking_path}")
    if not Path(events_path).exists():
        print(f"Warning: expected events file not found: {events_path}")

    rc_link = run_linker(linker_py, tracking_path, events_path, args.output, args.window, args.sigma, args.summary, use_conda, args.tdeed_env)
    if rc_link != 0:
        print("EventToPlayerLinker failed")
        sys.exit(4)

    print(f"Linking completed. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
