#!/usr/bin/env bash
set -euo pipefail

# Run from project root
cd "$(dirname "$0")/.."

python pipeline/detect_video.py
python pipeline/inspect_predictions.py
python pipeline/track_video.py
python pipeline/reid_video.py
