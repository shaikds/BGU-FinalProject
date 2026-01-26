# inspect_predictions.py
#!/usr/bin/env python3
"""
Inspect predictions.json schema produced by detect_video.py.

Run:
  python pipeline/inspect_predictions.py
"""

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / "outputs" / "detections" / "predictions.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p} (run detect_video.py first)")

    data = json.loads(p.read_text())

    print("Top-level type:", type(data))

    if isinstance(data, list):
        print("List length:", len(data))
        if len(data) == 0:
            return
        first = data[0]
        print("First element type:", type(first))
        if isinstance(first, dict):
            print("First element keys:", list(first.keys()))
        print("First element sample:", str(first)[:1200])

    elif isinstance(data, dict):
        keys = list(data.keys())
        print("Dict keys (first 50):", keys[:50])
        if len(keys) == 0:
            return
        k0 = keys[0]
        v0 = data[k0]
        print("Sample key:", k0)
        print("Sample value type:", type(v0))
        if isinstance(v0, dict):
            print("Sample value keys:", list(v0.keys()))
        print("Sample value sample:", str(v0)[:1200])

        # extra helpful info
        print("\n---- summary ----")
        print("video:", data.get("video"))
        print("fps:", data.get("fps"))
        print("total_frames:", data.get("total_frames"))
        print("stride:", data.get("stride"))
        frames = data.get("frames", [])
        print("frames_len:", len(frames))
        if frames:
            print("first frame_index:", frames[0].get("frame_index"))
            print("last frame_index:", frames[-1].get("frame_index"))

    else:
        print("Unsupported JSON top-level type:", type(data))


if __name__ == "__main__":
    main()
