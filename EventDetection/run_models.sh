#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_path>"
    exit 1
fi

VIDEO_PATH="$1"
VIDEO_NAME="$(basename "$VIDEO_PATH")"

cd ~/T-DEED-2
# This script runs the two models creates an ensemble, filters by best thresholds, and visualizes the results. Adjust thresholds as needed based on your validation set performance.
## First Model: SoccerNetBall_challenge1 - Best model for ball detection and ball-player interactions:
python inference.py --model SoccerNetBall_challenge1 \
    --video_path "$VIDEO_PATH" \
    --frame_width 796 --frame_height 448 \
    --inference_threshold 0.05
mv inference_output/results_inference.json inference_output/results_snball.json

# Second Model: SoccerNet_small - Best model of SoccerNet:
# For MOBILE VERSION, we use different script - replace inference.py with run_inference_no_vcap.py in all places.
python inference.py --model SoccerNet_small \
    --video_path "$VIDEO_PATH" \
    --frame_width 796 --frame_height 448 \
    --inference_threshold 0.05
mv inference_output/results_inference.json inference_output/results_soccernet.json


# Step 1: Re-run ensemble with correct FPS
python specialist_ensemble.py \
    --snball inference_output/results_snball.json \
    --soccernet inference_output/results_soccernet.json \
    --output inference_output/results_ensemble.json \
    --video "$VIDEO_NAME"

# Filter by best thresholds per event:
python3 scripts/filter_by_confidence.py \
    --json inference_output/results_ensemble.json \
    --output inference_output/results_ensemble_filtered.json \
    --threshold PASS=0.3 \
    --threshold "HIGH PASS=0.25" \
    --threshold "THROW IN=0.40" \
    --threshold "CROSS=0.55" \
    --threshold "HEADER=0.15" \
    --threshold "SHOT=0.35" \
    --threshold "GOAL=0.70" \
    --threshold "FOUL=0.99" \
    --threshold "BALL PLAYER BLOCK=0.90" \
    --threshold "OUT=0.99" \
    --map-tackle \
    --remove-drive

# Create Video of Event Detections:
python3 scripts/visualize_events.py    \
    --video "$VIDEO_PATH"   \
    --json inference_output/results_ensemble_filtered.json \
    --output inference_output/visualized_ensemble.mp4




# # Step 2: Re-evaluate with correct FPS
# python evaluation/full_evaluation.py \
#     inference_output/results_ensemble.json \
#     evaluation/tdeed_evaluation_6.5.csv \
#     0.3 \
#     --video evaluation_tdeed_vid1_v2.mp4

# # Step 3: Optional — with replay exclusion
# python evaluation/full_evaluation.py \
#     inference_output/results_ensemble.json \
#     evaluation/ground_truth.csv \
#     0.3 \
#     --video "$VIDEO_NAME" \
#     --exclude-replays evaluation/tdeed_eval_v3_replay_windows.csv

# # Step 4: Optional — deep diagnostics
# python evaluation/model_analysis.py \
#     inference_output/results_ensemble.json \
#     evaluation/tdeed_evaluation_6.5.csv \
#     --video evaluation_tdeed_vid1_v2.mp4