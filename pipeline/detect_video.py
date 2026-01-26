import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Project root: ~/sn_pipe_trial
ROOT = Path(__file__).resolve().parents[1]

# Allow importing code from the HF repo folder (your existing setup)
HF_DIR = ROOT / "hf_rfdetr_soccernet"
sys.path.insert(0, str(HF_DIR))

# Model code from HF repo
from rfdetr.models.lwdetr import build_model, PostProcess  # type: ignore
from rfdetr.main import get_args_parser  # type: ignore

# ===== config =====
SCORE_THR = 0.50
MAX_DET = 60

STRIDE = 1                 # 1 = full video
MAX_FRAMES: Optional[int] = None  # None = run all frames
BACKUP_ON_OVERWRITE = True


def _backup_if_exists(path: Path) -> None:
    if not BACKUP_ON_OVERWRITE:
        return
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        path.replace(bak)


def ensure(ns, name, default):
    if not hasattr(ns, name):
        setattr(ns, name, default)


def load_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    ckpt_args = ckpt["args"]
    args = get_args_parser().parse_args([])

    for k, v in vars(ckpt_args).items():
        setattr(args, k, v)

    ensure(args, "device", device)
    ensure(args, "segmentation_head", False)
    ensure(args, "mask_downsample_ratio", 4)

    # logits=4 => num_classes=3 (background is added internally)
    args.num_classes = 3

    dev = torch.device(device)
    model = build_model(args).to(dev).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    post = PostProcess()
    return model, post, dev


def round_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


@torch.no_grad()
def infer_one_image(model, post, device, pil_img: Image.Image):
    x = TF.to_tensor(pil_img).unsqueeze(0)  # [1,3,H,W]

    # DINOv2 windowed backbone requires H,W divisible by 56
    block = 56
    _, _, h, w = x.shape
    new_h = round_to_multiple(h, block)
    new_w = round_to_multiple(w, block)
    if new_h != h or new_w != w:
        x = TF.resize(x, [new_h, new_w])

    x = x.to(device)
    outputs = model(x)

    target_sizes = torch.tensor([[pil_img.height, pil_img.width]], device=device)
    res = post(outputs, target_sizes)[0]

    return res["boxes"], res["scores"], res["labels"]


def filter_and_cap(boxes, scores, labels, score_thr: float, max_det: int):
    keep = scores >= score_thr
    boxes_k = boxes[keep]
    scores_k = scores[keep]
    labels_k = labels[keep]

    if scores_k.numel() > max_det:
        top_idx = torch.argsort(scores_k, descending=True)[:max_det]
        boxes_k = boxes_k[top_idx]
        scores_k = scores_k[top_idx]
        labels_k = labels_k[top_idx]

    return boxes_k, scores_k, labels_k


def dets_to_json_list(boxes, scores, labels) -> List[Dict[str, Any]]:
    return [
        {
            "bbox_xyxy": [float(x) for x in b.detach().cpu().tolist()],
            "score": float(s.detach().cpu().item()),
            "label": int(l.detach().cpu().item()),
        }
        for b, s, l in zip(boxes, scores, labels)
    ]


def main():
    video_path = ROOT / "data" / "video_test.mp4"
    ckpt_path = ROOT / "weights" / "checkpoint_best_regular.pth"
    out_dir = ROOT / "outputs" / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert video_path.exists(), f"Missing video: {video_path}"
    assert ckpt_path.exists(), f"Missing ckpt: {ckpt_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video info: fps={fps:.0f} total_frames={total_frames}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} from: {ckpt_path}")
    model, post, dev = load_model_from_ckpt(ckpt_path, device)

    frames_payload: List[Dict[str, Any]] = []

    processed = 0
    fi = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if STRIDE > 1 and (fi % STRIDE != 0):
            fi += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        boxes, scores, labels = infer_one_image(model, post, dev, img)
        boxes_k, scores_k, labels_k = filter_and_cap(boxes, scores, labels, SCORE_THR, MAX_DET)

        frames_payload.append(
            {
                "frame_index": fi,
                "detections": dets_to_json_list(boxes_k, scores_k, labels_k),
            }
        )

        processed += 1
        if MAX_FRAMES is not None and processed >= MAX_FRAMES:
            break

        fi += 1

    cap.release()

    out_path = out_dir / "predictions.json"
    _backup_if_exists(out_path)

    payload = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "score_thr": SCORE_THR,
        "stride": STRIDE,
        "max_frames": MAX_FRAMES,
        "max_det": MAX_DET,
        "num_frames_written": processed,
        "frames": frames_payload,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    total_dets = sum(len(f["detections"]) for f in frames_payload)
    print("Detections saved:", out_path)
    print("frames written:", processed)
    print("total detections kept:", total_dets)
    if frames_payload:
        print("frame0 index:", frames_payload[0]["frame_index"], "dets:", len(frames_payload[0]["detections"]))
        print("last frame index:", frames_payload[-1]["frame_index"], "dets:", len(frames_payload[-1]["detections"]))


if __name__ == "__main__":
    main()
