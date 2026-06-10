#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
	echo "Usage: $0 <video_path>"
	exit 1
fi

VIDEO_PATH="$1"
VIDEO_NAME="$(basename "$VIDEO_PATH")"

# Run from project root
cd "$(dirname "$0")/.."

# Ensure data directory exists and link the provided video as the expected
# pipeline input: data/seconds_video.mp4
mkdir -p data
TARGET=data/seconds_video.mp4
if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then
	ts=$(date +%Y%m%d_%H%M%S)
	mv "$TARGET" "${TARGET}.bak_${ts}"
fi
ln -sfn "$VIDEO_PATH" "$TARGET"
echo "Using video: $VIDEO_PATH -> $TARGET"

python pipeline/detect_video.py --video "$VIDEO_PATH"
python pipeline/inspect_predictions.py
python pipeline/visualize_detections.py --video "$VIDEO_PATH"

python pipeline/track_video.py --video "$VIDEO_PATH"
python pipeline/visualize_tracks.py --video "$VIDEO_PATH"

python pipeline/pteam_assignment_v2.py --video "$VIDEO_PATH"

python pipeline/reid_prtreid_v2.py --video "$VIDEO_PATH"
