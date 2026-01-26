# visualize_detections.py
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2

ROOT = Path(__file__).resolve().parents[1]

# Render every frame by default (full video)
VIS_STRIDE = 1
SAVE_FRAMES = False  # for full video this can be huge; set True only for short debugging


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def draw_bbox(
    img_bgr,
    box_xyxy: List[float],
    text: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
):
    x1, y1, x2, y2 = [int(round(x)) for x in box_xyxy]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img_bgr,
        text,
        (x1, max(0, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    det_path = ROOT / "outputs" / "detections" / "predictions.json"
    assert det_path.exists(), f"Missing detections file: {det_path}"
    d = read_json(det_path)

    video_path = Path(d.get("video", ROOT / "data" / "video_test.mp4"))
    assert video_path.exists(), f"Missing video: {video_path}"

    frames = d.get("frames", [])
    assert isinstance(frames, list) and frames, "predictions.json has no frames"

    dets_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for fr in frames:
        fi = int(fr.get("frame_index", -1))
        if fi < 0:
            continue
        dets = fr.get("detections", [])
        if isinstance(dets, list):
            dets_by_frame[fi] = dets

    out_dir = ROOT / "outputs" / "vis_det"
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_FRAMES:
        frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_fps = in_fps / max(1, VIS_STRIDE)
    out_mp4 = out_dir / "detections_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(out_fps), (w, h))

    print(f"Visualizing detections FULL video: total_frames={total_frames} VIS_STRIDE={VIS_STRIDE} -> {out_mp4}")

    for fi in range(total_frames):
        if fi % VIS_STRIDE != 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"[WARN] failed reading frame {fi}")
            continue

        dets = dets_by_frame.get(fi, [])

        cv2.putText(
            frame_bgr,
            f"DETECTION | frame={fi} | dets={len(dets)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for det in dets:
            box = det.get("bbox_xyxy", None)
            if not isinstance(box, list) or len(box) != 4:
                continue
            score = float(det.get("score", 0.0))
            label = int(det.get("label", -1))
            draw_bbox(frame_bgr, box, f"cls={label} s={score:.2f}", color=(0, 255, 0))

        if SAVE_FRAMES:
            jpg_path = frames_dir / f"frame_{fi:06d}.jpg"
            cv2.imwrite(str(jpg_path), frame_bgr)

        vw.write(frame_bgr)

        if fi % max(1, (VIS_STRIDE * 200)) == 0:
            print(f"  rendered frame {fi}/{total_frames}")

    cap.release()
    vw.release()
    print("Saved:", out_mp4)
    if SAVE_FRAMES:
        print("Frames saved under:", frames_dir)


if __name__ == "__main__":
    main()
