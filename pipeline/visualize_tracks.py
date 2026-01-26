# visualize_tracks.py
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2

ROOT = Path(__file__).resolve().parents[1]

VIS_STRIDE = 1
SAVE_FRAMES = False  # full video can be huge


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    x = (track_id * 2654435761) & 0xFFFFFFFF
    b = 64 + (x & 127)
    g = 64 + ((x >> 8) & 127)
    r = 64 + ((x >> 16) & 127)
    return (int(b), int(g), int(r))


def draw_bbox(
    img_bgr,
    box_xyxy: List[float],
    text: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    x1, y1, x2, y2 = [int(round(x)) for x in box_xyxy]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img_bgr,
        text,
        (x1, max(0, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    tracks_path = ROOT / "outputs" / "tracks" / "tracks.json"
    assert tracks_path.exists(), f"Missing tracks file: {tracks_path}"
    d = read_json(tracks_path)

    video_path = Path(d.get("video", ROOT / "data" / "video_test.mp4"))
    assert video_path.exists(), f"Missing video: {video_path}"

    rows = d.get("tracks", [])
    assert isinstance(rows, list), "tracks.json expected 'tracks' list"

    by_frame: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        fi = int(r.get("frame_index", -1))
        if fi >= 0:
            by_frame[fi].append(r)

    out_dir = ROOT / "outputs" / "vis_tracks"
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
    out_mp4 = out_dir / "tracks_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(out_fps), (w, h))

    print(f"Visualizing tracks FULL video: total_frames={total_frames} VIS_STRIDE={VIS_STRIDE} -> {out_mp4}")

    for fi in range(total_frames):
        if fi % VIS_STRIDE != 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"[WARN] failed reading frame {fi}")
            continue

        rows_f = by_frame.get(fi, [])

        cv2.putText(
            frame_bgr,
            f"TRACKING | frame={fi} | tracks={len(rows_f)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for r in rows_f:
            box = r.get("bbox_xyxy", None)
            if not isinstance(box, list) or len(box) != 4:
                continue
            tid = int(r.get("track_id", -1))
            score = r.get("score", None)
            label = r.get("label", None)
            color = id_to_color(tid)
            s_txt = f"{float(score):.2f}" if score is not None else "NA"
            l_txt = f"{int(label)}" if label is not None else "NA"
            draw_bbox(frame_bgr, box, f"id={tid} cls={l_txt} s={s_txt}", color=color)

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
