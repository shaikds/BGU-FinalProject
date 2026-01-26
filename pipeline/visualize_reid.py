# visualize_reid.py
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

VIS_STRIDE = 1
SAVE_FRAMES = False  # full video can be huge


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def clip_box_xyxy(box: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = clamp_int(x1, 0, w - 1)
    y1 = clamp_int(y1, 0, h - 1)
    x2 = clamp_int(x2, 0, w - 1)
    y2 = clamp_int(y2, 0, h - 1)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def gid_to_color(gid: int) -> Tuple[int, int, int]:
    # deterministic BGR color per global_id
    x = (gid * 2654435761) & 0xFFFFFFFF
    b = 64 + (x & 127)
    g = 64 + ((x >> 8) & 127)
    r = 64 + ((x >> 16) & 127)
    return (int(b), int(g), int(r))


def draw_box(frame_bgr: np.ndarray, box: List[float], text: str, color: Tuple[int, int, int]) -> None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clip_box_xyxy(box, w, h)

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    tracks_path = ROOT / "outputs" / "tracks" / "tracks.json"
    map_path = ROOT / "outputs" / "reid" / "trackid_to_globalid.json"

    assert tracks_path.exists(), f"Missing tracks file: {tracks_path}"
    assert map_path.exists(), f"Missing mapping file: {map_path} (run reid_video.py first)"

    tracks_data = read_json(tracks_path)
    video_path = Path(tracks_data.get("video", ROOT / "data" / "video_test.mp4"))
    assert video_path.exists(), f"Missing video: {video_path}"

    rows = tracks_data.get("tracks", [])
    assert isinstance(rows, list), "tracks.json expected 'tracks' list"

    m = read_json(map_path)
    trackid_to_gid = m.get("trackid_to_globalid", {})
    if not isinstance(trackid_to_gid, dict):
        raise ValueError("trackid_to_globalid.json has unexpected schema")

    by_frame = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        fi = int(r.get("frame_index", -1))
        if fi < 0:
            continue
        by_frame[fi].append(r)

    out_dir = ROOT / "outputs" / "vis_reid"
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_FRAMES:
        frames_dir.mkdir(parents=True, exist_ok=True)

    out_video = out_dir / "reid_overlay.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if vw <= 0 or vh <= 0:
        raise RuntimeError("Could not read video width/height")

    out_fps = fps / max(1, VIS_STRIDE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, float(out_fps), (vw, vh))

    print(f"Overlay REID FULL video: total_frames={total_frames} VIS_STRIDE={VIS_STRIDE}")

    rendered = 0
    for fi in range(total_frames):
        if fi % VIS_STRIDE != 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"[WARN] Failed reading frame {fi}, skipping")
            continue

        rows_f = by_frame.get(fi, [])

        cv2.putText(
            frame_bgr,
            f"REID(global_id) | frame={fi} | tracks={len(rows_f)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for r in rows_f:
            box = r.get("bbox_xyxy")
            tid = int(r.get("track_id", -1))
            if not isinstance(box, list) or len(box) != 4:
                continue

            gid = int(trackid_to_gid.get(str(tid), trackid_to_gid.get(tid, tid)))
            color = gid_to_color(gid)

            draw_box(frame_bgr, [float(x) for x in box], f"gid={gid} (tid={tid})", color)

        writer.write(frame_bgr)

        if SAVE_FRAMES:
            frame_path = frames_dir / f"frame_{fi:06d}.jpg"
            cv2.imwrite(str(frame_path), frame_bgr)

        rendered += 1
        if rendered % 200 == 0:
            print(f"  rendered {rendered} frames... latest={fi}")

    cap.release()
    writer.release()

    print("Wrote:", out_video)
    if SAVE_FRAMES:
        print("Frames saved under:", frames_dir)


if __name__ == "__main__":
    main()
