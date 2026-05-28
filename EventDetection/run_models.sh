#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_path>"
    exit 1
fi

VIDEO_PATH="$1"
VIDEO_NAME="$(basename "$VIDEO_PATH")"

cd ~/T-DEED-2

# 1. הרץ inference עם SoccerNetBall (אם עוד לא רצת):
python inference.py --model SoccerNetBall_challenge1 \
    --video_path "$VIDEO_PATH" \
    --frame_width 796 --frame_height 448 \
    --inference_threshold 0.05
mv inference_output/results_inference.json inference_output/results_snball.json

# 2. הרץ inference עם SoccerNet (small שעובד טוב על GOAL):
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