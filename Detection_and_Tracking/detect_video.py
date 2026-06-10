import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import cv2
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Project root
# Example: /home/shaikar/sn_pipe_trial
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Add the cloned Hugging Face repo to PYTHONPATH so that:
#   from inference import RFDETRSoccerNet
# works without modifying the model repo itself.
# ============================================================
HF_DIR = ROOT / "hf_rfdetr_soccernet"
sys.path.insert(0, str(HF_DIR))

# The detector wrapper exposed by the cloned model repo.
# According to the repo's example and inference guide, the intended
# usage is:
#   model = RFDETRSoccerNet()
#   df = model.process_video(...)
from inference import RFDETRSoccerNet  # type: ignore

# ============================================================
# Detection config
# ============================================================
SCORE_THR = 0.50
MAX_DET = 60

# Process every STRIDE-th frame.
# 1 = every frame, 2 = every other frame, etc.
STRIDE = 1

# None = process the full video
MAX_FRAMES: Optional[int] = None

# If predictions.json already exists, move it aside before overwriting.
BACKUP_ON_OVERWRITE = True


def _backup_if_exists(path: Path) -> None:
    """
    If the output file already exists, move it to a timestamped backup.
    This is useful when re-running experiments so old outputs are not lost.
    """
    if not BACKUP_ON_OVERWRITE:
        return
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        path.replace(bak)


def _get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Read FPS and total frame count directly from the video file.

    We still use the model repo for inference, but we read metadata ourselves
    because our downstream pipeline expects these values inside predictions.json.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
    }


def _normalize_detection_row(row: pd.Series) -> Dict[str, Any]:
    """
    Convert one detection row returned by RFDETRSoccerNet into the JSON format
    expected by the tracking stage.

    Expected columns from the detector wrapper include:
      - frame
      - x1, y1, x2, y2
      - confidence
      - class_id

    Output format per detection:
      {
        "bbox_xyxy": [x1, y1, x2, y2],
        "score": confidence,
        "label": class_id
      }
    """
    return {
        "bbox_xyxy": [
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        ],
        "score": float(row["confidence"]),
        "label": int(row["class_id"]),
    }


def _build_sparse_frame_map(df: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
    """
    Convert the detector DataFrame into a mapping:
      frame_index -> list of detections

    This is called "sparse" because it only includes frames that actually
    appear in the detector output.
    """
    frames_dict: Dict[int, List[Dict[str, Any]]] = {}

    if len(df) == 0:
        return frames_dict

    # Sort rows by frame first, then by confidence descending,
    # so any per-frame capping is deterministic.
    df_sorted = df.sort_values(["frame", "confidence"], ascending=[True, False])

    for frame_idx, group in df_sorted.groupby("frame", sort=True):
        frame_idx = int(frame_idx)

        dets: List[Dict[str, Any]] = []
        for _, row in group.head(MAX_DET).iterrows():
            dets.append(_normalize_detection_row(row))

        frames_dict[frame_idx] = dets

    return frames_dict


def _expected_frame_indices(total_frames: int, stride: int, max_frames: Optional[int]) -> List[int]:
    """
    Reconstruct the list of frame indices that *should* be represented in the
    output payload according to our pipeline settings.

    Why do this?
    -----------
    RFDETRSoccerNet.process_video(...) may return rows only for frames where
    detections were found. But the old pipeline structure wrote one entry per
    processed frame, even if there were zero detections.

    To keep downstream behavior stable, we explicitly create frame entries for:
      0, stride, 2*stride, ...

    and attach an empty detection list when nothing was detected.
    """
    if total_frames <= 0:
        return []

    frame_indices = list(range(0, total_frames, stride))

    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    return frame_indices


def _build_frames_payload(
    frames_dict: Dict[int, List[Dict[str, Any]]],
    total_frames: int,
    stride: int,
    max_frames: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Build the final 'frames' list written into predictions.json.

    Each element has the form:
      {
        "frame_index": <int>,
        "detections": [...]
      }

    We include empty detections for processed frames with no detections,
    because that is usually the safest behavior for downstream tracking code.
    """
    frame_indices = _expected_frame_indices(total_frames, stride, max_frames)

    payload: List[Dict[str, Any]] = []
    for fi in frame_indices:
        payload.append(
            {
                "frame_index": fi,
                "detections": frames_dict.get(fi, []),
            }
        )

    return payload


def _validate_detector_output(df: pd.DataFrame) -> None:
    """
    Sanity-check that the DataFrame returned by the detector wrapper contains
    the columns we need for downstream conversion.
    """
    required_cols = {"frame", "x1", "y1", "x2", "y2", "confidence", "class_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Detector output is missing required columns: "
            f"{sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def main():
    """Run object detection on a video and save pipeline-formatted JSON.

    Accepts optional `--video` to override the default `data/seconds_video.mp4`.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    # Also accept a positional video path so the script can be called as:
    #   python detect_video.py /path/to/video.mp4
    parser.add_argument("pos_video", nargs="?", default=None, help="Positional path to input video (alternative to --video)")
    args, _ = parser.parse_known_args()

    # Prefer the explicit --video flag, fall back to the positional argument.
    video_arg = args.video or args.pos_video
    if video_arg is None:
        parser.error("Missing input video. Provide --video or pass the video path as a positional argument.")

    video_path = Path(video_arg)
    out_dir = ROOT / "outputs" / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert HF_DIR.exists(), f"Missing detector repo directory: {HF_DIR}"
    assert video_path.exists(), f"Missing video: {video_path}"

    print("=" * 70)
    print("DETECTION STAGE")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"HF detector repo: {HF_DIR}")
    print(f"Score threshold: {SCORE_THR}")
    print(f"Frame stride: {STRIDE}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"Max detections per frame: {MAX_DET}")

    meta = _get_video_metadata(video_path)
    fps = meta["fps"]
    total_frames = meta["total_frames"]

    print(f"Video metadata -> fps={fps:.3f}, total_frames={total_frames}")

    # ------------------------------------------------------------
    # Create the detector exactly through the model repo's API.
    # We are not rebuilding the model manually and not changing its code.
    # ------------------------------------------------------------
    print("Loading RFDETRSoccerNet detector...")
    model = RFDETRSoccerNet(model_path=str(ROOT / "weights" / "checkpoint_best_regular.pth"))

    # ------------------------------------------------------------
    # Run inference through the repo's intended public API.
    #
    # The HF repo documentation shows process_video(...) returning a
    # pandas DataFrame with columns like frame, x1, y1, x2, y2, confidence,
    # class_id, class_name, etc.
    # ------------------------------------------------------------
    print("Running detector on video...")
    df = model.process_video(
        video_path=str(video_path),
        confidence_threshold=SCORE_THR,
        frame_skip=STRIDE,
        max_frames=MAX_FRAMES,
        save_results=False,
        output_dir=None,
    )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected detector to return pandas.DataFrame, got: {type(df)}"
        )

    _validate_detector_output(df)

    print(f"Raw detector rows returned: {len(df)}")

    # Convert detector DataFrame -> per-frame detection lists
    frames_dict = _build_sparse_frame_map(df)

    # Reconstruct full processed-frame payload, including empty frames
    frames_payload = _build_frames_payload(
        frames_dict=frames_dict,
        total_frames=total_frames,
        stride=STRIDE,
        max_frames=MAX_FRAMES,
    )

    out_path = out_dir / "predictions.json"
    _backup_if_exists(out_path)

    total_dets_kept = sum(len(f["detections"]) for f in frames_payload)

    payload = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "score_thr": SCORE_THR,
        "stride": STRIDE,
        "max_frames": MAX_FRAMES,
        "max_det": MAX_DET,
        "num_frames_written": len(frames_payload),
        "frames": frames_payload,
    }

    out_path.write_text(json.dumps(payload, indent=2))

    print("-" * 70)
    print(f"Detections saved: {out_path}")
    print(f"Frames written: {len(frames_payload)}")
    print(f"Total detections kept: {total_dets_kept}")

    if frames_payload:
        first = frames_payload[0]
        last = frames_payload[-1]
        print(
            f"First written frame -> index={first['frame_index']} "
            f"dets={len(first['detections'])}"
        )
        print(
            f"Last written frame  -> index={last['frame_index']} "
            f"dets={len(last['detections'])}"
        )

    if len(df) > 0:
        class_summary = (
            df["class_name"].value_counts().to_dict()
            if "class_name" in df.columns
            else {}
        )
        if class_summary:
            print(f"Class summary: {class_summary}")

    print("=" * 70)
    print("DETECTION STAGE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()