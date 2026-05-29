# track_video.py
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]

BACKUP_ON_OVERWRITE = True


def _backup_if_exists(path: Path) -> None:
    if not BACKUP_ON_OVERWRITE:
        return
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        path.replace(bak)


def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", type=str, default=None, help="Optional override video path")
    args, _ = parser.parse_known_args()

    det_path = ROOT / "outputs" / "detections" / "predictions.json"
    out_dir = ROOT / "outputs" / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tracks.json"

    assert det_path.exists(), f"Missing detections file: {det_path}"

    data = json.loads(det_path.read_text())

    video_path = args.video if args.video else data.get("video", None)
    fps = float(data.get("fps", 30.0) or 30.0)
    total_frames = int(data.get("total_frames", 0) or 0)
    stride = int(data.get("stride", 1) or 1)

    frames = data.get("frames", [])
    if not frames:
        raise RuntimeError("No frames found in predictions.json")

    # sort by frame_index (important)
    frames = sorted(frames, key=lambda x: int(x.get("frame_index", -1)))

    print("Using ByteTrack with supervision:", sv.__version__)
    
    # tracking parameters (can change to test better fits)
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=float(fps),
        minimum_consecutive_frames=2,
    )

    rows: List[Dict[str, Any]] = []
    total_in = 0
    total_out = 0

    empty = sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int64),
    )

    prev_fi = None

    for f in frames:
        frame_index = int(f.get("frame_index", -1))
        dets_list = f.get("detections", [])

        if frame_index < 0:
            continue

        # If we sampled detections (stride>1) or there are missing indices,
        # we must advance ByteTrack time-steps for the gap using empty detections.
        if prev_fi is not None and frame_index > prev_fi + 1:
            gap = frame_index - prev_fi - 1
            for _ in range(gap):
                tracker.update_with_detections(empty)

        prev_fi = frame_index

        if not isinstance(dets_list, list):
            dets_list = []

        total_in += len(dets_list)

        if not dets_list:
            tracker.update_with_detections(empty)
            continue

        xyxy = np.array([d["bbox_xyxy"] for d in dets_list], dtype=np.float32)
        conf = np.array([d.get("score", 0.0) for d in dets_list], dtype=np.float32)
        cls = np.array([d.get("label", -1) for d in dets_list], dtype=np.int64)

        dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracked = tracker.update_with_detections(dets)

        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None:
            raise RuntimeError(
                "supervision ByteTrack did not attach tracker_id. "
                "Please confirm supervision version and ByteTrack API."
            )

        total_out += len(tracker_ids)

        for i in range(len(tracker_ids)):
            rows.append(
                {
                    "frame_index": frame_index,
                    "track_id": int(tracker_ids[i]),
                    "bbox_xyxy": [float(x) for x in tracked.xyxy[i].tolist()],
                    "score": float(tracked.confidence[i]) if tracked.confidence is not None else None,
                    "label": int(tracked.class_id[i]) if tracked.class_id is not None else None,
                }
            )



    # Preserve every raw ball detection from predictions.json WITHOUT adding it to rows.
    # This keeps the original ByteTrack output and track IDs exactly as in FIRST,
    # while downstream stages can still access all raw label=0 ball detections.
    ball_detections = []
    for f in frames:
        fi = int(f.get("frame_index", -1))
        for j, d in enumerate(f.get("detections", []) or []):
            if int(d.get("label", -1)) == 0:
                ball_detections.append({
                    "frame_index": fi,
                    "ball_detection_id": int(fi * 1000 + j),
                    "bbox_xyxy": [float(x) for x in d["bbox_xyxy"]],
                    "score": float(d.get("score", 0.0)),
                    "label": 0,
                    "source": "raw_detection_sidecar",
                    "is_tracked": False,
                })

    payload = {
        "video": str(video_path) if video_path is not None else None,
        "fps": float(fps),
        "total_frames": int(total_frames),
        "stride_in_detections": int(stride),
        "num_detection_frames_in": int(len(frames)),
        "num_detections_in": int(total_in),
        "num_tracked_rows": int(len(rows)),
        "num_raw_ball_detections_kept": int(len(ball_detections)),
        "ball_policy": "sidecar_only: tracks list is unchanged from FIRST ByteTrack output; all raw label=0 detections are in ball_detections",
        "ball_detections": ball_detections,
        "tracks": rows,
    }

    _backup_if_exists(out_path)
    out_path.write_text(json.dumps(payload, indent=2))

    print("Tracks saved:", out_path)
    print("num detection frames:", len(frames), "| stride:", stride)
    print("num dets in:", total_in)
    print("num tracked rows:", len(rows))
    print("num raw ball detections kept:", len(ball_detections))
    if rows:
        print("unique track_ids:", len(set(r["track_id"] for r in rows)))
        print("sample row:", rows[0])


if __name__ == "__main__":
    main()
