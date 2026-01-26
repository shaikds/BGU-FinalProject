# reid_video.py
# NOTE: this keeps your existing ReID+linking logic, but it’s already compatible with full video
# as long as tracks.json covers the full timeline (which track_video.py now supports).

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

BACKUP_ON_OVERWRITE = True


def _backup_if_exists(path: Path) -> None:
    if not BACKUP_ON_OVERWRITE:
        return
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        path.replace(bak)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clip_xyxy(box: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(clamp(x1, 0, w - 1))
    y1 = int(clamp(y1, 0, h - 1))
    x2 = int(clamp(x2, 0, w - 1))
    y2 = int(clamp(y2, 0, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float(torch.dot(a, b) / denom)


def build_reid_model(device: torch.device) -> nn.Module:
    from torchvision.models import resnet50, ResNet50_Weights
    m = resnet50(weights=ResNet50_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.eval().to(device)
    return m


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup_if_exists(path)
    path.write_text(json.dumps(obj, indent=2))


def get_video_info(cap: cv2.VideoCapture) -> Tuple[float, int, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return fps, total, w, h


def uf_find(parent: Dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def uf_union(parent: Dict[int, int], a: int, b: int):
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


@torch.no_grad()
def main():
    tracks_path = ROOT / "outputs" / "tracks" / "tracks.json"
    assert tracks_path.exists(), f"Missing tracks file: {tracks_path}"

    tracks_data = read_json(tracks_path)
    video_path = Path(tracks_data.get("video", ROOT / "data" / "video_test.mp4"))
    assert video_path.exists(), f"Missing video: {video_path}"

    rows = tracks_data.get("tracks", [])
    assert isinstance(rows, list), "tracks.json expected 'tracks' to be a list"

    out_dir = ROOT / "outputs" / "reid"
    crops_dir = out_dir / "crops"
    out_pt = out_dir / "reid_embeddings.pt"
    out_obs = out_dir / "reid_observations.json"
    out_report = out_dir / "reid_report.json"
    out_map = out_dir / "trackid_to_globalid.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_crops = False  # for full video keep this OFF unless you debug
    SIM_THR = 0.90
    MAX_GAP = 150
    ALLOW_OVERLAP = 0

    rows_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        fi = int(r.get("frame_index", -1))
        if fi >= 0:
            rows_by_frame[fi].append(r)

    frame_indices = sorted(rows_by_frame.keys())
    print(f"ReID: frames with tracks: {len(frame_indices)} (from {tracks_path})")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps, total_frames, vw, vh = get_video_info(cap)
    print(f"Video info: fps={fps:.2f} total_frames={total_frames} size={vw}x{vh}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_reid_model(device)

    tfm = T.Compose(
        [
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    embeddings_list: List[torch.Tensor] = []
    observations: List[dict] = []
    track_embs: Dict[int, List[Tuple[int, torch.Tensor]]] = defaultdict(list)

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        crops = []
        metas = []
        for r in rows_by_frame[fi]:
            tid = int(r.get("track_id", -1))
            box = r.get("bbox_xyxy", None)
            if tid < 0 or not isinstance(box, list) or len(box) != 4:
                continue

            x1, y1, x2, y2 = clip_xyxy([float(x) for x in box], vw, vh)
            crop = pil.crop((x1, y1, x2, y2))

            if save_crops:
                fd = crops_dir / f"frame_{fi:06d}"
                fd.mkdir(parents=True, exist_ok=True)
                crop.save(fd / f"track_{tid:04d}.jpg")

            crops.append(tfm(crop))
            metas.append(
                (tid, [float(x1), float(y1), float(x2), float(y2)], float(r.get("score", 0.0)), int(r.get("label", -1)))
            )

        if not crops:
            continue

        batch = torch.stack(crops, dim=0).to(device)
        feats = model(batch).detach().float().cpu()

        for j, (tid, bbox, score, label) in enumerate(metas):
            emb = feats[j].contiguous()
            idx = len(embeddings_list)
            embeddings_list.append(emb)

            observations.append(
                {
                    "obs_index": idx,
                    "frame_index": int(fi),
                    "track_id": int(tid),
                    "bbox_xyxy": bbox,
                    "score": float(score),
                    "label": int(label),
                }
            )

            track_embs[int(tid)].append((int(fi), emb))

    cap.release()

    if embeddings_list:
        E = torch.stack(embeddings_list, dim=0).to(torch.float32)
        emb_dim = int(E.shape[1])
    else:
        E = torch.empty((0, 2048), dtype=torch.float32)
        emb_dim = 2048

    torch.save(
        {
            "video": str(video_path),
            "source_tracks_json": str(tracks_path),
            "embedding_dim": emb_dim,
            "num_observations": int(E.shape[0]),
            "embeddings": E,
            "frame_index": torch.tensor([o["frame_index"] for o in observations], dtype=torch.int32),
            "track_id": torch.tensor([o["track_id"] for o in observations], dtype=torch.int32),
        },
        out_pt,
    )
    print("Embeddings saved (binary):", out_pt)

    write_json(
        out_obs,
        {
            "video": str(video_path),
            "source_tracks_json": str(tracks_path),
            "embedding_dim": emb_dim,
            "num_frames_with_tracks": len(frame_indices),
            "num_observations_embedded": len(observations),
            "observations": observations,
            "notes": [
                "Embeddings are stored in reid_embeddings.pt (Torch binary).",
                "Each observation has obs_index that maps into embeddings tensor rows.",
            ],
        },
    )
    print("Observations saved (json):", out_obs)

    # ===== Linking by mean embedding + time constraints =====
    track_stats = {}
    track_means = {}

    for tid, items in track_embs.items():
        items = sorted(items, key=lambda x: x[0])
        frames_t = [f for f, _ in items]
        frame_min, frame_max = min(frames_t), max(frames_t)

        embs = torch.stack([e for _, e in items], dim=0)
        mean_emb = embs.mean(dim=0)
        track_means[tid] = mean_emb

        track_stats[tid] = {
            "track_id": tid,
            "num_embeddings": len(items),
            "frame_min": int(frame_min),
            "frame_max": int(frame_max),
        }

    tids = sorted(track_means.keys())
    print("Tracks for linking:", len(tids))

    parent = {tid: tid for tid in tids}
    merge_edges = []

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            t1, t2 = tids[i], tids[j]

            a = track_stats[t1]
            b = track_stats[t2]

            # overlap constraint
            overlap = min(a["frame_max"], b["frame_max"]) - max(a["frame_min"], b["frame_min"]) + 1
            if overlap > ALLOW_OVERLAP:
                continue

            # gap constraint
            gap = max(a["frame_min"], b["frame_min"]) - min(a["frame_max"], b["frame_max"])
            if gap > MAX_GAP:
                continue

            sim = cosine_sim(track_means[t1], track_means[t2])
            if sim >= SIM_THR:
                uf_union(parent, t1, t2)
                merge_edges.append({"track_id_a": t1, "track_id_b": t2, "cosine_sim": float(sim)})

    # compress & assign global ids
    root_to_gid: Dict[int, int] = {}
    trackid_to_gid: Dict[int, int] = {}
    next_gid = 1
    for tid in tids:
        r = uf_find(parent, tid)
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        trackid_to_gid[tid] = root_to_gid[r]

    write_json(out_map, {"trackid_to_globalid": trackid_to_gid})
    print("Mapping saved:", out_map)
    print("num_global_ids:", len(set(trackid_to_gid.values())), "merge_edges:", len(merge_edges))

    # ===== report =====
    per_track_stats = []
    for tid, items in track_embs.items():
        items = sorted(items, key=lambda x: x[0])
        gid = int(trackid_to_gid.get(tid, tid))

        if len(items) < 2:
            per_track_stats.append(
                {"track_id": tid, "global_id": gid, "num_embeddings": len(items), "mean_consecutive_cosine": None, "min_consecutive_cosine": None}
            )
            continue

        sims = []
        for (_, e1), (_, e2) in zip(items, items[1:]):
            sims.append(cosine_sim(e1, e2))

        per_track_stats.append(
            {
                "track_id": tid,
                "global_id": gid,
                "num_embeddings": len(items),
                "mean_consecutive_cosine": float(sum(sims) / len(sims)),
                "min_consecutive_cosine": float(min(sims)),
            }
        )

    report = {
        "video": str(video_path),
        "source_tracks_json": str(tracks_path),
        "outputs": {
            "reid_embeddings_pt": str(out_pt),
            "reid_observations_json": str(out_obs),
            "mapping_json": str(out_map),
        },
        "params": {"sim_thr": SIM_THR, "max_gap": MAX_GAP, "allow_overlap": ALLOW_OVERLAP},
        "num_tracks": len(tids),
        "num_tracks_with_2plus": sum(1 for v in track_embs.values() if len(v) >= 2),
        "num_observations": len(observations),
        "num_global_ids": len(set(trackid_to_gid.values())),
        "num_merge_edges": len(merge_edges),
        "merge_edges": merge_edges[:200],  # avoid huge report
        "per_track_stats": per_track_stats,
        "notes": [
            "This is a baseline ReID using ImageNet ResNet50 embeddings. For real player ReID, swap to a sports-trained ReID model.",
            "Color consistency in visualize_reid depends on successful merging of tracklets into global_id.",
        ],
    }
    write_json(out_report, report)
    print("ReID report saved:", out_report)

    valid = [x for x in per_track_stats if x["mean_consecutive_cosine"] is not None]
    if valid:
        means = sorted([x["mean_consecutive_cosine"] for x in valid])
        print(
            f"Tracks with >=2 embs: {len(valid)} | mean cosine p50={means[len(means)//2]:.3f} "
            f"min={min(means):.3f} max={max(means):.3f}"
        )

    print("Next: run visualize_reid.py to see colors by global_id.")


if __name__ == "__main__":
    main()
