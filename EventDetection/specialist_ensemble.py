"""
Specialist Ensemble: combine predictions from two T-DEED models.

USAGE:
    python specialist_ensemble.py \
        --snball inference_output/results_snball.json \
        --soccernet inference_output/results_soccernet.json \
        --output inference_output/results_ensemble.json \
        --video path/to/video.mp4

The script:
    1. Normalizes label vocabularies across models (Title Case -> UPPERCASE)
    2. Takes GOAL/SHOT predictions from SoccerNet (its specialty)
    3. Takes everything else from SoccerNetBall
    4. Applies per-class thresholds
    5. Resolves temporal conflicts via NMS
"""

import json
import argparse
import subprocess
from fractions import Fraction
from collections import defaultdict


DEFAULT_FPS = 25.0


# ==========================================================
# Label normalization: SoccerNet Title Case -> SoccerNetBall UPPERCASE
# ==========================================================
SOCCERNET_LABEL_MAP = {
    "Goal": "GOAL",
    "Throw-in": "THROW IN",
    "Ball out of play": "OUT",
    "Shots on target": "SHOT",
    "Shots off target": "SHOT",
    "Indirect free-kick": "FREE KICK",
    "Direct free-kick": "FREE KICK",
    "Corner": "CORNER",
    "Kick-off": "KICK OFF",
    "Clearance": "CLEARANCE",
    "Yellow card": "YELLOW CARD",
    "Red card": "RED CARD",
    "Foul": "FOUL",
}

# Classes where SoccerNet model is the expert
SOCCERNET_SPECIALIST_CLASSES = {
    "GOAL",
    "FOUL",
}

# Per-class thresholds (low so evaluator can do its own sweep)
DEFAULT_THRESHOLDS = {
    "CROSS": 0.05,
    "THROW IN": 0.05,
    "HIGH PASS": 0.05,
    "PASS": 0.05,
    "HEADER": 0.05,
    "OUT": 0.05,
    "DRIVE": 0.05,
    "BALL PLAYER BLOCK": 0.05,
    "PLAYER SUCCESSFUL TACKLE": 0.05,
    "SHOT": 0.05,
    "FREE KICK": 0.05,
    "GOAL": 0.05,
}


# ============================================================
# FPS detection
# ============================================================

def get_video_fps(video_path):
    """Probe video file with ffprobe and return its FPS as float."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0',
             video_path],
            capture_output=True, text=True, check=True
        )
        rate_str = result.stdout.strip()
        if not rate_str:
            raise RuntimeError("ffprobe returned empty FPS")
        return float(Fraction(rate_str))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Could not parse FPS from ffprobe: {e}")


# ============================================================
# Load / filter / threshold / NMS
# ============================================================

def load_predictions(path, label="model", label_map=None):
    """Load T-DEED JSON, normalize labels."""
    with open(path, 'r') as f:
        data = json.load(f)

    preds = []
    for p in data['predictions']:
        raw_label = p['label']
        if label_map and raw_label in label_map:
            norm = label_map[raw_label]
        else:
            norm = raw_label.upper()

        preds.append({
            'frame': p['frame'],
            'label': norm,
            'score': p.get('confidence', p.get('score', 0)),
            'source': label,
        })
    return preds


def filter_by_class(predictions, class_set, source_name):
    filtered = [p for p in predictions if p['label'] in class_set]
    print(f"  {source_name}: kept {len(filtered)} predictions for classes {class_set}")
    return filtered


def filter_by_class_excluding(predictions, excluded_set, source_name):
    filtered = [p for p in predictions if p['label'] not in excluded_set]
    print(f"  {source_name}: kept {len(filtered)} predictions (excluded {excluded_set})")
    return filtered


def apply_thresholds(predictions, thresholds, default_threshold=0.05):
    filtered = []
    drop_counts = defaultdict(int)
    keep_counts = defaultdict(int)
    for p in predictions:
        thresh = thresholds.get(p['label'], default_threshold)
        if p['score'] >= thresh:
            filtered.append(p)
            keep_counts[p['label']] += 1
        else:
            drop_counts[p['label']] += 1
    return filtered, dict(keep_counts), dict(drop_counts)


def resolve_temporal_conflicts(predictions, window_frames):
    """NMS: drop lower-confidence predictions within window_frames of a kept one."""
    by_class = defaultdict(list)
    for p in predictions:
        by_class[p['label']].append(p)

    resolved = []
    for cls, preds in by_class.items():
        preds_sorted = sorted(preds, key=lambda x: -x['score'])
        kept = []
        for p in preds_sorted:
            too_close = any(abs(p['frame'] - k['frame']) <= window_frames for k in kept)
            if not too_close:
                kept.append(p)
        resolved.extend(kept)

    resolved.sort(key=lambda x: x['frame'])
    return resolved


def ensemble(snball_preds, soccernet_preds, fps,
             thresholds=None, apply_nms=True, nms_window=None,
             specialist_classes=None):
    """Build ensemble predictions."""
    if specialist_classes is None:
        specialist_classes = SOCCERNET_SPECIALIST_CLASSES
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if nms_window is None:
        nms_window = int(fps)  # 1 second of frames

    print("\n" + "=" * 80)
    print("  SPECIALIST ENSEMBLE")
    print("=" * 80)

    print(f"\nFPS in use: {fps}")
    print(f"NMS window: {nms_window} frames ({nms_window/fps:.2f}s)")

    print(f"\nInput:")
    print(f"   SoccerNetBall predictions:  {len(snball_preds)}")
    print(f"   SoccerNet predictions:      {len(soccernet_preds)}")

    print(f"\nRouting strategy:")
    print(f"   SoccerNet handles:     {specialist_classes}")
    print(f"   SoccerNetBall handles: everything else")

    print(f"\nFiltering by source:")
    soccernet_specialist = filter_by_class(
        soccernet_preds, specialist_classes, "SoccerNet")
    snball_general = filter_by_class_excluding(
        snball_preds, specialist_classes, "SoccerNetBall")

    combined = soccernet_specialist + snball_general
    print(f"\n   Combined (before threshold): {len(combined)}")

    print(f"\nApplying per-class thresholds:")
    combined_filtered, kept, dropped = apply_thresholds(combined, thresholds)

    print(f"\n{'Class':<28} {'Threshold':>10} {'Kept':>5} {'Dropped':>8}")
    print("-" * 55)
    all_classes = sorted(set(list(kept.keys()) + list(dropped.keys())))
    for cls in all_classes:
        thresh = thresholds.get(cls, 0.05)
        print(f"{cls:<28} {thresh:>10.2f} {kept.get(cls, 0):>5} {dropped.get(cls, 0):>8}")

    print(f"\n   After thresholding: {len(combined_filtered)}")

    if apply_nms:
        print(f"\nApplying NMS (window={nms_window} frames = {nms_window/fps:.2f}s)")
        combined_filtered = resolve_temporal_conflicts(combined_filtered, window_frames=nms_window)
        print(f"   After NMS: {len(combined_filtered)}")

    combined_filtered.sort(key=lambda x: x['frame'])

    by_source = defaultdict(int)
    by_class = defaultdict(int)
    for p in combined_filtered:
        by_source[p['source']] += 1
        by_class[p['label']] += 1

    print(f"\nFinal ensemble breakdown:")
    print(f"   Total: {len(combined_filtered)}")
    print(f"\n   By source:")
    for src, cnt in sorted(by_source.items()):
        print(f"     {src}: {cnt}")

    print(f"\n   By class:")
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"     {cls:<28} {cnt}")

    return combined_filtered


def save_predictions(predictions, output_path, fps):
    output = {
        'fps': fps,
        'predictions': [
            {
                'frame': p['frame'],
                'label': p['label'],
                'confidence': p['score'],
                'source': p.get('source', 'unknown'),
            }
            for p in predictions
        ]
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved ensemble predictions to: {output_path}")


def print_top_predictions(predictions, fps, n=20):
    print(f"\nTOP {n} PREDICTIONS BY CONFIDENCE:")
    print(f"{'Time':<10} {'Class':<28} {'Score':>6} {'Source':>15}")
    print("-" * 65)

    sorted_preds = sorted(predictions, key=lambda x: -x['score'])[:n]
    for p in sorted_preds:
        time_sec = p['frame'] / fps
        time_str = f"{int(time_sec//60):02d}:{time_sec%60:05.2f}"
        print(f"{time_str:<10} {p['label']:<28} {p['score']:>6.2f} {p.get('source', 'unknown'):>15}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specialist ensemble of T-DEED models')
    parser.add_argument('--snball', required=True, help='SoccerNetBall predictions JSON')
    parser.add_argument('--soccernet', required=True, help='SoccerNet predictions JSON')
    parser.add_argument('--output', default='inference_output/results_ensemble.json',
                        help='Output path for ensemble predictions')
    parser.add_argument('--video', default=None,
                        help='Path to source video for FPS auto-detection')
    parser.add_argument('--no-nms', action='store_true', help='Skip NMS step')
    parser.add_argument('--nms-window', type=int, default=None,
                        help='NMS window in frames (default: 1 second worth of frames)')
    parser.add_argument('--specialist-classes', nargs='+', default=None,
                        help='Classes for SoccerNet specialist (default: GOAL, SHOT)')

    args = parser.parse_args()

    # Resolve FPS
    if args.video:
        try:
            fps = get_video_fps(args.video)
            print(f"Auto-detected FPS from {args.video}: {fps}")
        except RuntimeError as e:
            print(f"FPS detection failed: {e}")
            print(f"Falling back to default FPS {DEFAULT_FPS}")
            fps = DEFAULT_FPS
    else:
        fps = DEFAULT_FPS
        print(f"No --video provided. Using default FPS: {fps}")

    print(f"Loading SoccerNetBall from: {args.snball}")
    snball_preds = load_predictions(args.snball, label="SoccerNetBall")

    print(f"Loading SoccerNet from: {args.soccernet}")
    soccernet_preds = load_predictions(
        args.soccernet, label="SoccerNet", label_map=SOCCERNET_LABEL_MAP)

    specialist = set(args.specialist_classes) if args.specialist_classes else None

    ensemble_preds = ensemble(
        snball_preds, soccernet_preds,
        fps=fps,
        apply_nms=(not args.no_nms),
        nms_window=args.nms_window,
        specialist_classes=specialist,
    )

    save_predictions(ensemble_preds, args.output, fps)
    print_top_predictions(ensemble_preds, fps, n=20)

    print("\n" + "=" * 80)
    print("  NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Evaluate ensemble against your ground truth:
   python evaluation/full_evaluation.py {args.output} evaluation/ground_truth.csv 0.05 --video {args.video or 'YOUR_VIDEO.mp4'}

2. Compare to single-model baselines.
""")