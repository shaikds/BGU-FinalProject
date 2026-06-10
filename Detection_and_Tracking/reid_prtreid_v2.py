import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# ============================================================================
# ReID V2 - balanced + targeted guards + two-stage component repair
# ============================================================================
# This version is designed for longer clips (for example 3.5 minutes / ~5400
# frames), where ByteTrack can fragment one real player into many local IDs.
#
# The previous attempts showed a clear precision/recall tradeoff:
# - aggressive merging: higher Recall but many false merges
# - conservative merging: high Precision but very low Recall
#
# Main strategy in this file:
# 1. Stage 1: run a conservative/balanced pairwise ReID merge, with team, spatial,
#    temporal, mutual-best, component-size and long-gap safety guards.
# 2. Stage 2: run a narrow component-repair pass. Instead of trusting one pair,
#    it connects a track/component to an existing identity component only if
#    multiple members of that component support the merge.
#
# This is meant to improve Recall without going back to broad, risky merging.
# ============================================================================

# ============================================================================
# Project paths
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "data" / "seconds_video.mp4"
TRACKS_PATH = ROOT / "outputs" / "tracks" / "tracks.json"
# TEAM_ASSIGNMENT_PATH = ROOT / "outputs" / "team_assignment" / "team_assignment.json"
TEAM_ASSIGNMENT_PATH = ROOT / "outputs" / "team_assignment_v2" / "team_assignment_v2.json"
OUT_DIR = ROOT / "outputs" / "reid_v2"

REID_ASSETS_DIR = ROOT / "reid"
PRTREID_DIR = REID_ASSETS_DIR / "prtreid"
CHECKPOINT_PATH = REID_ASSETS_DIR / "prtreid-soccernet-baseline.pth.tar"
HRNET_PRETRAINED_PATH = REID_ASSETS_DIR / "hrnetv2_w32_imagenet_pretrained.pth"


# ============================================================================
# Config
# ============================================================================
VALID_REID_LABELS = {1, 3}
BATCH_SIZE = 64

# Dynamic similarity thresholds by temporal gap
# ---------------------------------------------------------------------------
# Important change for longer videos:
# In a 20-second clip, two same-team players may not have many opportunities to
# be confused. In a 3.5-minute clip, however, many same-team players can look
# similar and reappear after long gaps. Therefore, long-gap merges must be much
# stricter than short-gap merges.
SHORT_GAP_THR = 10          # almost adjacent tracklets
MID_GAP_THR = 40            # short re-entry / short occlusion
LONG_GAP_THR = 250          # medium temporal distance
VERY_LONG_GAP_THR = 750     # long temporal distance

SIM_THR_SHORT = 0.80        # allow short-gap recovery
SIM_THR_MID = 0.82          # balanced: recover medium-gap same-player fragments
SIM_THR_LONG = 0.85         # softer than strict version, still safer than short gaps
SIM_THR_VERY_LONG = 0.89    # very long gaps remain strict
SIM_THR_ULTRA_LONG = 0.92   # beyond VERY_LONG_GAP_THR

MAX_GAP = 1200              # keep candidates, but require high similarity for long gaps

# Overlap handling
ENABLE_SMALL_OVERLAP_ALLOWANCE = True
MAX_OVERLAP_FOR_REID = 40
OVERLAP_MIN_SIM = 0.85
OVERLAP_MAX_SPATIAL_DIST = 1.0
OVERLAP_ALLOWANCE = 40
OVERLAP_RELAX_FACTOR = 0.05

# Spatial gating
SPATIAL_GATE_ENABLED = True
SPATIAL_GATE_SHORT = 1.2
SPATIAL_GATE_MID = 1.60     # slightly relaxed to recover plausible medium-gap fragments
SPATIAL_GATE_LONG = 2.7      # slightly relaxed, still controlled by high similarity

# Merge policy - same as V1 style
USE_MUTUAL_BEST_MATCH = True
BLOCK_COMPONENT_TIME_CONFLICTS = True

ALLOW_NON_MUTUAL_SHORT_GAP = True
NON_MUTUAL_MAX_GAP = 30
NON_MUTUAL_MIN_SIM = 0.82

ALLOW_NON_MUTUAL_HIGH_SIM = True
NON_MUTUAL_HIGH_SIM = 0.88

# ---------------------------------------------------------------------------
# Targeted false-merge guards for long videos
# ---------------------------------------------------------------------------
# In the previous balanced run we improved Recall a bit, but still got several
# false merges. The main risk comes from allowing non-mutual high-similarity
# merges across long temporal gaps. In a 3.5-minute football clip, many players
# from the same team can look similar, so a single high visual score is not
# enough unless the pair is also mutual-best or extremely convincing.
NON_MUTUAL_HIGH_SIM_MAX_GAP = LONG_GAP_THR      # normal non-mutual high-sim only up to medium gaps
NON_MUTUAL_VERY_HIGH_SIM = 0.92                 # above this, allow non-mutual even on longer gaps
NON_MUTUAL_VERY_HIGH_TOPK = 0.91                # but require strong crop-level evidence too

# Guard very long gaps even after the normal similarity threshold passed.
# This is not a full block; it just requires both mean and top-k evidence to be strong.
ENABLE_VERY_LONG_GAP_GUARD = True
VERY_LONG_GAP_GUARD_MIN_GAP = VERY_LONG_GAP_THR
VERY_LONG_GAP_MIN_MEAN_SIM = 0.84
VERY_LONG_GAP_MIN_TOPK_SIM = 0.91

# Targeted Recall recovery: allow a short fragment to reconnect when the evidence
# looks like a real re-entry. This is safer than lowering global thresholds because
# it only applies to short tracklets, moderate gaps, same-team candidates, and
# spatially plausible pairs.
ENABLE_FRAGMENT_REENTRY_RELAXATION = True
FRAGMENT_MAX_LEN = 35
FRAGMENT_REENTRY_MAX_GAP = 350
FRAGMENT_REENTRY_MIN_TOPK_SIM = 0.88
FRAGMENT_REENTRY_MIN_MEAN_SIM = 0.78
FRAGMENT_REENTRY_MAX_SPATIAL_DIST = 2.2

# ---------------------------------------------------------------------------
# Stage-2 component repair pass
# ---------------------------------------------------------------------------
# Stage 1 is pairwise. A single false pair can be dangerous, so it remains
# conservative. However, after Stage 1 creates small identity components, we can
# safely recover Recall by asking a stronger question:
#
#   "Does this unmerged tracklet look similar to several members of an existing
#    component, not just one?"
#
# This is much safer than lowering all thresholds because the evidence must be
# component-level. It is especially useful for patterns such as:
#   player_1 -> tracklets [1, 49, 103, 116, 234, 281, ...]
# where Stage 1 may merge [49, 234, 281] but miss [1, 103, 116].
ENABLE_COMPONENT_REPAIR_PASS = True
COMPONENT_REPAIR_MAX_ROUNDS = 2
COMPONENT_REPAIR_MAX_GAP = 900
COMPONENT_REPAIR_MIN_SUPPORT_PAIRS = 2
COMPONENT_REPAIR_MIN_TARGET_COMPONENT_SIZE = 2
COMPONENT_REPAIR_MAX_SOURCE_COMPONENT_SIZE = 3
COMPONENT_REPAIR_MIN_COMBINED_SIM = 0.815
COMPONENT_REPAIR_MIN_MEAN_SIM = 0.755
COMPONENT_REPAIR_MIN_TOPK_SIM = 0.875
COMPONENT_REPAIR_STRONG_SINGLE_COMBINED_SIM = 0.90
COMPONENT_REPAIR_STRONG_SINGLE_TOPK_SIM = 0.92
COMPONENT_REPAIR_MAX_SPATIAL_DIST = 3.0
COMPONENT_REPAIR_BLOCK_OVERLAP = True

# Team constraints
BLOCK_DIFFERENT_TEAMS = True
BLOCK_REFEREE_LIKE_TRACKS = True

# Long-video safety controls
# ---------------------------------------------------------------------------
# These are intended to improve Precision on long videos.
# They reduce false merges caused by visually similar same-team players.
MIN_TRACK_LEN_FOR_STRONG_MERGE = 3
LOW_TEAM_CONFIDENCE_FOR_REID = 0.55
BLOCK_LOW_CONFIDENCE_TEAM_ON_LONG_GAP = True
LOW_CONFIDENCE_LONG_GAP_MIN = LONG_GAP_THR  # only block low-confidence teams on longer gaps

# Component-size penalty:
# When a global component already contains many local track IDs, adding another
# track is dangerous. One bad merge contaminates the whole identity component.
COMPONENT_SIZE_PENALTY_ENABLED = True
COMPONENT_SIZE_PENALTY_START = 5
COMPONENT_SIZE_PENALTY_PER_EXTRA_TRACK = 0.007
COMPONENT_SIZE_PENALTY_MAX = 0.04

# Multi-crop similarity:
# Instead of relying only on the mean embedding of each tracklet, we compare
# several crops from each pair and combine mean similarity with top-k similarity.
# This is more robust to blur, occlusion, and bad crops in long videos.
USE_MULTI_CROP_SIMILARITY = True
MAX_EMB_SAMPLES_PER_TRACK = 12
TOPK_SIM_K = 5
PAIR_SIM_MEAN_WEIGHT = 0.60
PAIR_SIM_TOPK_WEIGHT = 0.40

# Outputs
SAVE_OVERLAY_VIDEO = True
OVERLAY_VIDEO_NAME = "reid_overlay.mp4"
BACKUP_ON_OVERWRITE = True

BALL_LABELS = {0}
BALL_COLOR = (0, 255, 0)


# ============================================================================
# Utilities
# ============================================================================
def _backup_if_exists(path: Path) -> None:
    if not BACKUP_ON_OVERWRITE:
        return
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        path.replace(bak)


def _ensure_paths_exist() -> None:
    assert VIDEO_PATH.exists(), f"Missing video: {VIDEO_PATH}"
    assert TRACKS_PATH.exists(), f"Missing tracks file: {TRACKS_PATH}"
    assert TEAM_ASSIGNMENT_PATH.exists(), f"Missing team assignment file: {TEAM_ASSIGNMENT_PATH}"
    assert PRTREID_DIR.exists(), f"Missing PRTReID repo dir: {PRTREID_DIR}"
    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}"
    assert HRNET_PRETRAINED_PATH.exists(), f"Missing HRNet pretrained file: {HRNET_PRETRAINED_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())


# ============================================================================
# Data structures
# ============================================================================
@dataclass
class TrackInfo:
    track_id: int
    label: int
    start_frame: int
    end_frame: int
    count: int

    # Main tracklet descriptor: average of all normalized crop embeddings.
    mean_embedding: torch.Tensor

    # Small deterministic sample of crop embeddings from this tracklet.
    # Used for top-k crop-to-crop similarity so we are not relying only on the mean.
    obs_embeddings: torch.Tensor

    first_bbox: List[float]
    last_bbox: List[float]
    mean_bbox_wh: Tuple[float, float]
    mean_center_xy: Tuple[float, float]

    # Per-frame boxes let us compute spatial distance correctly for overlap cases.
    frame_to_bbox: Dict[int, List[float]]

    # Optional team-assignment metadata must appear after non-default fields.
    team_id: Optional[int] = None
    team_confidence: Optional[float] = None
    referee_like: bool = False
    min_team_color_dist: Optional[float] = None
    mean_color_lab: Optional[np.ndarray] = None


# ============================================================================
# Loading tracks + team assignment
# ============================================================================
def load_tracks(tracks_path: Path) -> Tuple[dict, List[dict], List[dict]]:
    with open(tracks_path, "r") as f:
        raw_data = json.load(f)

    all_tracks = raw_data["tracks"]
    reid_tracks = [t for t in all_tracks if int(t["label"]) in VALID_REID_LABELS]

    print(f"Loaded total track observations: {len(all_tracks)}")
    print(f"Loaded ReID-eligible observations: {len(reid_tracks)}")
    print(f"Unique track IDs (all): {len(sorted(set(t['track_id'] for t in all_tracks)))}")
    print(f"Unique track IDs (ReID labels only): {len(sorted(set(t['track_id'] for t in reid_tracks)))}")

    return raw_data, all_tracks, reid_tracks


def load_team_assignment(team_assignment_path: Path) -> Dict[int, dict]:
    with open(team_assignment_path, "r") as f:
        data = json.load(f)

    by_tid: Dict[int, dict] = {}
    for row in data["tracks"]:
        by_tid[int(row["track_id"])] = row

    print(f"Loaded team-assignment metadata for {len(by_tid)} tracks")
    return by_tid


# ============================================================================
# Crop extraction from video
# ============================================================================
def crop_persons_from_video(video_path: Path, tracks: List[dict]) -> Tuple[List[int], List[np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened for crop extraction: {total_frames} frames")

    frame_to_tracks = defaultdict(list)
    for i, t in enumerate(tracks):
        frame_to_tracks[int(t["frame_index"])].append((i, t))

    crops: List[Optional[np.ndarray]] = [None] * len(tracks)
    current_frame_idx = -1

    for frame_idx in sorted(frame_to_tracks.keys()):
        if frame_idx != current_frame_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame_bgr = cap.read()
        current_frame_idx = frame_idx
        if not ret:
            print(f"Warning: failed to read frame {frame_idx}")
            continue

        h, w = frame_bgr.shape[:2]
        for i, t in frame_to_tracks[frame_idx]:
            x1, y1, x2, y2 = t["bbox_xyxy"]

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop_bgr = frame_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            crops[i] = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    cap.release()

    valid = [(i, crops[i]) for i in range(len(crops)) if crops[i] is not None]
    valid_indices = [v[0] for v in valid]
    valid_crops = [v[1] for v in valid]

    print(f"Cropped valid ReID images: {len(valid_crops)} / {len(tracks)}")
    return valid_indices, valid_crops


# ============================================================================
# Build PRTReID model from checkpoint
# ============================================================================
def build_model_from_checkpoint(
    checkpoint_path: Path,
    hrnet_pretrained_path: Path,
    device: torch.device,
    prtreid_dir: Path,
):
    print(f"Loading ReID checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt["config"]
    print(
        f"Checkpoint info -> epoch={ckpt.get('epoch', '?')}, "
        f"rank1={ckpt.get('rank1', '?')}, mAP={ckpt.get('mAP', '?')}"
    )

    if hrnet_pretrained_path:
        cfg.model.hrnet_pretrained_path = str(hrnet_pretrained_path)

    prtreid_repo = os.path.abspath(str(prtreid_dir))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if script_dir in sys.path:
        sys.path.remove(script_dir)
    if prtreid_repo not in sys.path:
        sys.path.insert(0, prtreid_repo)
    sys.path.append(script_dir)

    import prtreid
    import prtreid.models

    print(f"prtreid loaded from: {prtreid.__file__}")

    state_dict = ckpt["state_dict"]
    num_classes = 1343
    for key in state_dict:
        if "global_identity_classifier.classifier.weight" in key:
            num_classes = state_dict[key].shape[0]
            break

    print(f"Identity classes inferred from checkpoint: {num_classes}")

    model = prtreid.models.build_model(
        name=cfg.model.name,
        num_classes=num_classes,
        loss="softmax",
        pretrained=False,
        use_gpu=device.type != "cpu",
        config=cfg,
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    if not missing and not unexpected:
        print("All checkpoint keys loaded successfully.")

    model = model.to(device)
    model.eval()

    img_h = getattr(cfg.data, "height", 256)
    img_w = getattr(cfg.data, "width", 128)
    print(f"Model ready on {device} | input size = {img_h}x{img_w}")

    return model, cfg, img_h, img_w


# ============================================================================
# Feature extraction
# ============================================================================
def get_test_transform(img_h: int, img_w: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(
    model,
    crops: List[np.ndarray],
    img_h: int,
    img_w: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    transform = get_test_transform(img_h, img_w)
    all_embeddings = []

    num_batches = (len(crops) + batch_size - 1) // batch_size
    for bi in range(num_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(crops))
        batch_crops = crops[start:end]

        tensors = [transform(c) for c in batch_crops]
        batch = torch.stack(tensors).to(device)
        output = model(batch)

        if isinstance(output, tuple):
            embeddings_dict = output[0]
            if isinstance(embeddings_dict, dict):
                chosen = None
                for key in ["bn_foreg", "bn_glob", "foreg", "glob"]:
                    if key in embeddings_dict:
                        chosen = embeddings_dict[key]
                        break
                if chosen is None:
                    chosen = list(embeddings_dict.values())[0]
                emb = chosen
            else:
                emb = embeddings_dict
        else:
            emb = output

        if emb.dim() > 2:
            emb = emb.view(emb.size(0), -1)

        emb = _normalize_embeddings(emb)
        all_embeddings.append(emb.cpu())

        if (bi + 1) % 100 == 0 or bi == num_batches - 1:
            print(f"  Extracted batch {bi + 1}/{num_batches}")

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Final embedding tensor shape: {tuple(all_embeddings.shape)}")
    return all_embeddings


# ============================================================================
# Track-level aggregation
# ============================================================================
def build_track_summaries(
    valid_tracks: List[dict],
    embeddings: torch.Tensor,
    team_assignment_by_tid: Dict[int, dict],
) -> Dict[int, TrackInfo]:
    by_track = defaultdict(list)
    for obs, emb in zip(valid_tracks, embeddings):
        by_track[int(obs["track_id"])].append((obs, emb))

    track_infos: Dict[int, TrackInfo] = {}

    for tid, items in by_track.items():
        items = sorted(items, key=lambda x: int(x[0]["frame_index"]))

        label = int(items[0][0]["label"])
        frames = [int(obs["frame_index"]) for obs, _ in items]

        emb_stack = torch.stack([emb for _, emb in items], dim=0)
        mean_emb = _normalize_embeddings(emb_stack.mean(dim=0, keepdim=True))[0]

        # Keep a small deterministic sample of crop embeddings per tracklet.
        # This avoids comparing only the average embedding, which may be noisy
        # when the tracklet contains blur, occlusions, or bad crops.
        if emb_stack.size(0) <= MAX_EMB_SAMPLES_PER_TRACK:
            sample_embs = emb_stack
        else:
            sample_idx = torch.linspace(0, emb_stack.size(0) - 1, steps=MAX_EMB_SAMPLES_PER_TRACK).long()
            sample_embs = emb_stack[sample_idx]
        sample_embs = _normalize_embeddings(sample_embs)

        first_bbox = list(items[0][0]["bbox_xyxy"])
        last_bbox = list(items[-1][0]["bbox_xyxy"])

        ws, hs, cxs, cys = [], [], [], []
        frame_to_bbox = {}

        for obs, _ in items:
            frame_idx = int(obs["frame_index"])
            bbox = [float(v) for v in obs["bbox_xyxy"]]
            frame_to_bbox[frame_idx] = bbox

            x1, y1, x2, y2 = bbox
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            ws.append(w)
            hs.append(h)
            cxs.append((x1 + x2) / 2.0)
            cys.append((y1 + y2) / 2.0)

        team_meta = team_assignment_by_tid.get(tid, {})
        mean_color_lab = team_meta.get("mean_color_lab", None)
        if mean_color_lab is not None:
            mean_color_lab = np.array(mean_color_lab, dtype=np.float32)

        track_infos[tid] = TrackInfo(
            track_id=tid,
            label=label,
            start_frame=min(frames),
            end_frame=max(frames),
            count=len(items),

            mean_embedding=mean_emb,
            obs_embeddings=sample_embs,
            team_confidence=team_meta.get("team_confidence", None),

            first_bbox=first_bbox,
            last_bbox=last_bbox,
            mean_bbox_wh=(float(np.mean(ws)), float(np.mean(hs))),
            mean_center_xy=(float(np.mean(cxs)), float(np.mean(cys))),
            frame_to_bbox=frame_to_bbox,

            team_id=team_meta.get("team_id", None),
            referee_like=bool(team_meta.get("referee_like", False)),
            min_team_color_dist=team_meta.get("min_team_color_dist", None),
            mean_color_lab=mean_color_lab,
        )

    print(f"Built track summaries for {len(track_infos)} local track IDs")
    return track_infos


# ============================================================================
# Pairwise helpers (V1 logic + team hard constraint)
# ============================================================================
class UnionFind:
    def __init__(self, items: List[int]):
        self.parent = {x: x for x in items}
        self.members = {x: {x} for x in items}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def get_members(self, x: int) -> set:
        return self.members[self.find(x)]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.members[ra].update(self.members[rb])
        del self.members[rb]


def temporal_overlap(a: TrackInfo, b: TrackInfo) -> bool:
    return not (a.end_frame < b.start_frame or b.end_frame < a.start_frame)


def temporal_overlap_size(a: TrackInfo, b: TrackInfo) -> int:
    left = max(a.start_frame, b.start_frame)
    right = min(a.end_frame, b.end_frame)
    if right < left:
        return 0
    return right - left + 1


def temporal_gap(a: TrackInfo, b: TrackInfo) -> int:
    if temporal_overlap(a, b):
        return 0
    if a.end_frame < b.start_frame:
        return b.start_frame - a.end_frame - 1
    return a.start_frame - b.end_frame - 1


def get_dynamic_sim_threshold(gap_frames: int) -> float:
    """
    Gap-aware similarity threshold.

    Long videos create many more false-positive opportunities. Therefore,
    the longer the temporal gap, the stronger the visual evidence required.
    """
    if gap_frames <= SHORT_GAP_THR:
        return SIM_THR_SHORT
    if gap_frames <= MID_GAP_THR:
        return SIM_THR_MID
    if gap_frames <= LONG_GAP_THR:
        return SIM_THR_LONG
    if gap_frames <= VERY_LONG_GAP_THR:
        return SIM_THR_VERY_LONG
    return SIM_THR_ULTRA_LONG


def pairwise_track_similarity(a: TrackInfo, b: TrackInfo) -> Tuple[float, dict]:
    """
    Robust tracklet-to-tracklet similarity.

    Old behavior used only cosine(mean_embedding_a, mean_embedding_b).
    That is sometimes too noisy for long videos. Here we combine:
    1. mean embedding similarity
    2. top-k crop-to-crop similarity between sampled crops

    The returned score is still a single scalar, so the rest of the pipeline can
    continue working as before, but the debug report also stores both parts.
    """
    mean_sim = _cosine_similarity(a.mean_embedding, b.mean_embedding)

    if not USE_MULTI_CROP_SIMILARITY:
        return mean_sim, {
            "mean_similarity": round(mean_sim, 4),
            "topk_similarity": None,
            "similarity_mode": "mean_only",
        }

    if a.obs_embeddings is None or b.obs_embeddings is None:
        return mean_sim, {
            "mean_similarity": round(mean_sim, 4),
            "topk_similarity": None,
            "similarity_mode": "mean_fallback_no_samples",
        }

    # obs_embeddings are already normalized, so matrix multiplication gives cosine similarities.
    sim_matrix = torch.matmul(a.obs_embeddings, b.obs_embeddings.T).flatten()
    if sim_matrix.numel() == 0:
        return mean_sim, {
            "mean_similarity": round(mean_sim, 4),
            "topk_similarity": None,
            "similarity_mode": "mean_fallback_empty_samples",
        }

    k = min(TOPK_SIM_K, sim_matrix.numel())
    topk_sim = float(torch.topk(sim_matrix, k=k).values.mean().item())

    combined = (PAIR_SIM_MEAN_WEIGHT * mean_sim) + (PAIR_SIM_TOPK_WEIGHT * topk_sim)
    return float(combined), {
        "mean_similarity": round(mean_sim, 4),
        "topk_similarity": round(topk_sim, 4),
        "similarity_mode": "mean_topk_combined",
    }


def is_fragment_reentry_candidate(candidate: dict) -> bool:
    """
    Targeted recall-recovery rule for long clips.

    Why this exists:
    ----------------
    In longer football videos, ByteTrack often splits one real player into many
    short local tracklets after occlusions, camera motion, and re-entry. If we
    require every merge to be mutual-best, we miss many correct reconnects.

    But lowering thresholds globally creates false merges between visually
    similar players from the same team. Therefore this rule is intentionally
    narrow: it only allows a non-mutual merge when one side is a short fragment,
    the temporal gap is not huge, top-k crop evidence is strong, mean similarity
    is still reasonable, and the spatial jump is plausible.
    """
    if not ENABLE_FRAGMENT_REENTRY_RELAXATION:
        return False

    # Do not use this relaxation for overlap cases; overlap duplicates are handled separately.
    if candidate.get("overlap", False):
        return False

    gap = int(candidate.get("gap_frames", 10**9))
    if gap > FRAGMENT_REENTRY_MAX_GAP:
        return False

    min_len = candidate.get("min_track_len", 10**9)
    if min_len is None or int(min_len) > FRAGMENT_MAX_LEN:
        return False

    mean_sim = candidate.get("mean_similarity", None)
    topk_sim = candidate.get("topk_similarity", None)
    if mean_sim is None or topk_sim is None:
        return False

    if float(mean_sim) < FRAGMENT_REENTRY_MIN_MEAN_SIM:
        return False
    if float(topk_sim) < FRAGMENT_REENTRY_MIN_TOPK_SIM:
        return False

    spatial_dist = candidate.get("spatial_distance", None)
    if spatial_dist is None:
        return False
    if float(spatial_dist) > FRAGMENT_REENTRY_MAX_SPATIAL_DIST:
        return False

    return True


def is_merge_eligible(candidate: dict) -> bool:
    """
    Decide whether a pair that passed the basic gates should actually be merged.

    Important idea:
    ---------------
    The basic gates already checked label/team/threshold/spatial constraints.
    This function decides how permissive we are about non-mutual matches.

    For long videos, the dangerous case is:
        same team + high visual similarity + not mutual best + long time gap
    because there are many visually similar same-team players.

    Therefore:
    - mutual-best remains the safest and preferred merge path.
    - short-gap non-mutual merges are still allowed because they usually reflect
      a small tracking break.
    - long-gap non-mutual merges require very strong evidence.
    - short fragment re-entry has a narrow relaxation rule to improve Recall.
    """
    if not candidate["passed_threshold"]:
        return False

    if not candidate["spatial_pass"]:
        return False

    # Mutual best is the strongest evidence that each tracklet chose the other.
    if candidate["mutual_best"]:
        return True

    gap = int(candidate["gap_frames"])
    sim = float(candidate["similarity"])
    topk = candidate.get("topk_similarity", None)

    # Short tracking gaps are usually safe even if the match is not mutual-best.
    if ALLOW_NON_MUTUAL_SHORT_GAP and gap <= NON_MUTUAL_MAX_GAP and sim >= NON_MUTUAL_MIN_SIM:
        return True

    # Fragment re-entry: narrow rule for short tracklets that likely belong to
    # a player who briefly disappeared/reappeared.
    if is_fragment_reentry_candidate(candidate):
        return True

    # Non-mutual high similarity is allowed only for medium gaps.
    # For longer gaps, require a much stronger combined score and top-k evidence.
    if ALLOW_NON_MUTUAL_HIGH_SIM and sim >= NON_MUTUAL_HIGH_SIM:
        if gap <= NON_MUTUAL_HIGH_SIM_MAX_GAP:
            return True
        if topk is not None and sim >= NON_MUTUAL_VERY_HIGH_SIM and float(topk) >= NON_MUTUAL_VERY_HIGH_TOPK:
            return True

    return False

def bbox_center_xyxy(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size_xyxy(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def get_spatial_gate_threshold(gap_frames: int) -> float:
    if gap_frames <= SHORT_GAP_THR:
        return SPATIAL_GATE_SHORT
    elif gap_frames <= MID_GAP_THR:
        return SPATIAL_GATE_MID
    else:
        return SPATIAL_GATE_LONG


def normalized_bbox_center_distance(bbox_a: List[float], bbox_b: List[float]) -> float:
    ax, ay = bbox_center_xyxy(bbox_a)
    bx, by = bbox_center_xyxy(bbox_b)

    aw, ah = bbox_size_xyxy(bbox_a)
    bw, bh = bbox_size_xyxy(bbox_b)

    avg_scale = max(1.0, (aw + ah + bw + bh) / 4.0)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return float(dist / avg_scale)


def endpoint_tracklet_distance(a: TrackInfo, b: TrackInfo) -> float:
    """
    Spatial distance for non-overlap pairs:
    compare end of earlier tracklet to start of later tracklet.
    """
    if a.end_frame <= b.start_frame:
        bbox_a = a.last_bbox
        bbox_b = b.first_bbox
    else:
        bbox_a = a.first_bbox
        bbox_b = b.last_bbox

    return normalized_bbox_center_distance(bbox_a, bbox_b)


def overlap_tracklet_distance(a: TrackInfo, b: TrackInfo) -> float:
    """
    Spatial distance for overlap pairs:
    compare the two tracks on the actual overlapping frames.

    We use a low percentile instead of a mean so that a few noisy frames
    do not completely dominate the distance.
    """
    common_frames = sorted(set(a.frame_to_bbox.keys()) & set(b.frame_to_bbox.keys()))
    if not common_frames:
        return endpoint_tracklet_distance(a, b)

    dists = []
    for f in common_frames:
        bbox_a = a.frame_to_bbox[f]
        bbox_b = b.frame_to_bbox[f]
        dists.append(normalized_bbox_center_distance(bbox_a, bbox_b))

    return float(np.percentile(np.array(dists, dtype=np.float32), 25))


def normalized_tracklet_distance(a: TrackInfo, b: TrackInfo) -> float:
    """
    Unified spatial distance:
    - overlap pair  -> compare on shared frames
    - non-overlap   -> compare endpoints
    """
    if temporal_overlap(a, b):
        return overlap_tracklet_distance(a, b)
    return endpoint_tracklet_distance(a, b)


def passes_spatial_gate(a: TrackInfo, b: TrackInfo, gap_frames: int) -> Tuple[bool, float, float]:
    ndist = normalized_tracklet_distance(a, b)
    thr = get_spatial_gate_threshold(gap_frames)
    return ndist <= thr, ndist, thr


def tracks_overlap_in_time(a: TrackInfo, b: TrackInfo) -> bool:
    return not (a.end_frame < b.start_frame or b.end_frame < a.start_frame)


def is_duplicate_overlap_case(a: TrackInfo, b: TrackInfo, candidate: dict) -> bool:
    """
    Detect cases like 22-23-24:
    - strong similarity
    - small spatial distance
    - one track is short (fragment)
    - overlap exists
    """

    if not candidate["overlap"]:
        return False

    overlap_frames = candidate.get("overlap_frames", 0)
    if overlap_frames is None or overlap_frames > 30:
        return False

    sim = candidate["similarity"]
    if sim < 0.88:
        return False

    spatial_dist = candidate.get("spatial_distance", None)
    if spatial_dist is None or spatial_dist > 0.6:
        return False

    # one must be short fragment
    if min(a.count, b.count) > 15:
        return False

    return True


def merged_component_has_time_conflict(
    component_a: set,
    component_b: set,
    track_infos: Dict[int, TrackInfo],
) -> bool:
    merged_ids = list(component_a | component_b)
    for i in range(len(merged_ids)):
        for j in range(i + 1, len(merged_ids)):
            ta = track_infos[merged_ids[i]]
            tb = track_infos[merged_ids[j]]
            if tracks_overlap_in_time(ta, tb):
                return True
    return False


def pair_passes_basic_gates(a: TrackInfo, b: TrackInfo) -> Tuple[bool, dict]:
    """
    V1 logic + two extra hard constraints:
    1. never merge across different teams
    2. never merge tracks already marked referee-like
    """
    if a.label != b.label:
        return False, {"reason": "label_mismatch"}

    if BLOCK_DIFFERENT_TEAMS and a.team_id is not None and b.team_id is not None:
        if int(a.team_id) != int(b.team_id):
            return False, {
                "reason": "team_mismatch",
                "team_id_a": a.team_id,
                "team_id_b": b.team_id,
            }

    if BLOCK_REFEREE_LIKE_TRACKS and (a.referee_like or b.referee_like):
        return False, {"reason": "referee_like_track"}

    overlap = temporal_overlap(a, b)
    gap = temporal_gap(a, b)
    overlap_frames = temporal_overlap_size(a, b) if overlap else 0

    if not overlap and gap > MAX_GAP:
        return False, {
            "reason": "gap_too_large",
            "gap_frames": gap,
            "overlap": overlap,
        }

    if overlap and overlap_frames > MAX_OVERLAP_FOR_REID:
        return False, {
            "reason": "overlap_too_large",
            "gap_frames": gap,
            "overlap": overlap,
            "overlap_frames": overlap_frames,
        }

    sim, sim_debug = pairwise_track_similarity(a, b)

    # Same threshold logic as V1, but with stricter long-gap thresholds.
    if overlap:
        thr = SIM_THR_LONG
    else:
        thr = get_dynamic_sim_threshold(gap)

    effective_thr = thr
    if overlap and overlap_frames <= OVERLAP_ALLOWANCE:
        effective_thr = max(0.0, thr - OVERLAP_RELAX_FACTOR)

    # Long-gap + low team confidence is risky. Team assignment is one of the
    # strongest guards against false same-team merges, so if either track has
    # very weak team confidence on a non-trivial temporal gap, block it.
    if BLOCK_LOW_CONFIDENCE_TEAM_ON_LONG_GAP and gap >= LOW_CONFIDENCE_LONG_GAP_MIN:
        conf_a = a.team_confidence
        conf_b = b.team_confidence
        if conf_a is not None and conf_b is not None:
            if float(conf_a) < LOW_TEAM_CONFIDENCE_FOR_REID or float(conf_b) < LOW_TEAM_CONFIDENCE_FOR_REID:
                return False, {
                    "reason": "low_team_confidence_long_gap",
                    "similarity": round(sim, 4),
                    "mean_similarity": sim_debug.get("mean_similarity"),
                    "topk_similarity": sim_debug.get("topk_similarity"),
                    "similarity_mode": sim_debug.get("similarity_mode"),
                    "threshold_used": round(thr, 4),
                    "effective_threshold_used": round(effective_thr, 4),
                    "gap_frames": gap,
                    "overlap": overlap,
                    "overlap_frames": overlap_frames,
                    "team_confidence_a": conf_a,
                    "team_confidence_b": conf_b,
                }

    # Very-long-gap guard:
    # Even if the dynamic threshold passed, very long gaps are risky because two
    # players from the same team may look similar across minutes. We therefore
    # require both the average descriptor and the clean-crop top-k evidence to be
    # strong. This targets false merges without lowering general Recall rules.
    if ENABLE_VERY_LONG_GAP_GUARD and (not overlap) and gap >= VERY_LONG_GAP_GUARD_MIN_GAP:
        mean_sim = sim_debug.get("mean_similarity", None)
        topk_sim = sim_debug.get("topk_similarity", None)
        if mean_sim is not None and topk_sim is not None:
            if float(mean_sim) < VERY_LONG_GAP_MIN_MEAN_SIM or float(topk_sim) < VERY_LONG_GAP_MIN_TOPK_SIM:
                return False, {
                    "reason": "very_long_gap_weak_evidence",
                    "similarity": round(sim, 4),
                    "mean_similarity": sim_debug.get("mean_similarity"),
                    "topk_similarity": sim_debug.get("topk_similarity"),
                    "similarity_mode": sim_debug.get("similarity_mode"),
                    "threshold_used": round(thr, 4),
                    "effective_threshold_used": round(effective_thr, 4),
                    "gap_frames": gap,
                    "overlap": overlap,
                    "overlap_frames": overlap_frames,
                }

    if sim < effective_thr:
        return False, {
            "reason": "below_threshold",
            "similarity": round(sim, 4),
            "mean_similarity": sim_debug.get("mean_similarity"),
            "topk_similarity": sim_debug.get("topk_similarity"),
            "similarity_mode": sim_debug.get("similarity_mode"),
            "threshold_used": round(thr, 4),
            "effective_threshold_used": round(effective_thr, 4),
            "gap_frames": gap,
            "overlap": overlap,
            "overlap_frames": overlap_frames,
        }

    spatial_pass = True
    spatial_dist = None
    spatial_thr = None

    if SPATIAL_GATE_ENABLED:
        if overlap:
            spatial_dist = normalized_tracklet_distance(a, b)
            spatial_thr = OVERLAP_MAX_SPATIAL_DIST
            spatial_pass = spatial_dist <= spatial_thr
        else:
            spatial_pass, spatial_dist, spatial_thr = passes_spatial_gate(a, b, gap)

        if not spatial_pass:
            return False, {
                "reason": "failed_spatial_gate",
                "similarity": round(sim, 4),
                "threshold_used": round(thr, 4),
                "effective_threshold_used": round(effective_thr, 4),
                "gap_frames": gap,
                "overlap": overlap,
                "overlap_frames": overlap_frames,
                "spatial_distance": round(spatial_dist, 4) if spatial_dist is not None else None,
                "spatial_threshold_used": round(spatial_thr, 4) if spatial_thr is not None else None,
            }

    if overlap and ENABLE_SMALL_OVERLAP_ALLOWANCE:
        if sim < OVERLAP_MIN_SIM and sim < effective_thr:
            return False, {
                "reason": "overlap_similarity_too_low",
                "similarity": round(sim, 4),
                "threshold_used": round(thr, 4),
                "effective_threshold_used": round(effective_thr, 4),
                "gap_frames": gap,
                "overlap": overlap,
                "overlap_frames": overlap_frames,
                "spatial_distance": round(spatial_dist, 4) if spatial_dist is not None else None,
                "spatial_threshold_used": round(spatial_thr, 4) if spatial_thr is not None else None,
            }

    return True, {
        "reason": "passed",
        "similarity": round(sim, 4),
        "mean_similarity": sim_debug.get("mean_similarity"),
        "topk_similarity": sim_debug.get("topk_similarity"),
        "similarity_mode": sim_debug.get("similarity_mode"),
        "threshold_used": round(thr, 4),
        "effective_threshold_used": round(effective_thr, 4),
        "gap_frames": gap,
        "overlap": overlap,
        "overlap_frames": overlap_frames,
        "spatial_distance": round(spatial_dist, 4) if spatial_dist is not None else None,
        "spatial_threshold_used": round(spatial_thr, 4) if spatial_thr is not None else None,
        "team_id_a": a.team_id,
        "team_id_b": b.team_id,
    }


# ============================================================================
# Component-level merge safety
# ============================================================================
def component_size_penalty(component_a: set, component_b: set) -> float:
    """
    Return an additional similarity requirement for merging into large components.

    Rationale:
    In long videos, a correct identity can be split into many tracklets. But once
    a component is already large, adding a wrong player is very damaging because
    it creates many false same-identity pairs. Therefore, require stronger
    similarity when either side is already a large component.
    """
    if not COMPONENT_SIZE_PENALTY_ENABLED:
        return 0.0

    largest = max(len(component_a), len(component_b))
    if largest < COMPONENT_SIZE_PENALTY_START:
        return 0.0

    extra = largest - COMPONENT_SIZE_PENALTY_START + 1
    return float(min(COMPONENT_SIZE_PENALTY_MAX, extra * COMPONENT_SIZE_PENALTY_PER_EXTRA_TRACK))


def candidate_required_similarity_for_component(candidate: dict, component_a: set, component_b: set) -> float:
    """
    Base threshold + component-size penalty.
    """
    base_thr = candidate.get("effective_threshold_used")
    if base_thr is None:
        base_thr = candidate.get("threshold_used")
    if base_thr is None:
        base_thr = 0.0
    return float(base_thr) + component_size_penalty(component_a, component_b)


# ============================================================================
# Stage-2 component repair helpers
# ============================================================================
def component_has_any_time_conflict(component_a: set, component_b: set, track_infos: Dict[int, TrackInfo]) -> bool:
    """
    Check whether merging two components would put two local tracks that exist
    at the same time into one global identity.

    This is stricter than only checking the proposed pair and is important in a
    two-stage repair pass. If any two members overlap in time, the repair merge
    is unsafe and should be blocked.
    """
    for tid_a in component_a:
        for tid_b in component_b:
            if tracks_overlap_in_time(track_infos[tid_a], track_infos[tid_b]):
                return True
    return False


def repair_pair_evidence(a: TrackInfo, b: TrackInfo) -> Tuple[bool, dict]:
    """
    Compute whether a single pair provides enough evidence for the Stage-2
    component repair pass.

    This is intentionally NOT the same as the Stage-1 pairwise merge rule:
    - Stage 1 decides whether one pair alone is safe enough to merge.
    - Stage 2 uses this as one vote among multiple component-level votes.

    Therefore, thresholds can be slightly lower than Stage 1, but the merge will
    only happen if several pair votes agree or if there is one extremely strong
    vote with very high top-k similarity.
    """
    if a.label != b.label:
        return False, {"reason": "label_mismatch"}

    # Team is still a hard constraint. This is the main protection against
    # connecting visually similar players from different teams.
    if BLOCK_DIFFERENT_TEAMS and a.team_id is not None and b.team_id is not None:
        if int(a.team_id) != int(b.team_id):
            return False, {"reason": "team_mismatch"}

    if BLOCK_REFEREE_LIKE_TRACKS and (a.referee_like or b.referee_like):
        return False, {"reason": "referee_like_track"}

    overlap = temporal_overlap(a, b)
    if COMPONENT_REPAIR_BLOCK_OVERLAP and overlap:
        return False, {"reason": "repair_overlap_blocked"}

    gap = temporal_gap(a, b)
    if gap > COMPONENT_REPAIR_MAX_GAP:
        return False, {"reason": "repair_gap_too_large", "gap_frames": gap}

    sim, sim_debug = pairwise_track_similarity(a, b)
    spatial_dist = normalized_tracklet_distance(a, b)

    if spatial_dist > COMPONENT_REPAIR_MAX_SPATIAL_DIST:
        return False, {
            "reason": "repair_spatial_too_far",
            "similarity": round(sim, 4),
            "mean_similarity": sim_debug.get("mean_similarity"),
            "topk_similarity": sim_debug.get("topk_similarity"),
            "gap_frames": gap,
            "spatial_distance": round(spatial_dist, 4),
        }

    mean_sim = sim_debug.get("mean_similarity", None)
    topk_sim = sim_debug.get("topk_similarity", None)
    if mean_sim is None or topk_sim is None:
        return False, {"reason": "repair_missing_similarity_parts"}

    normal_support = (
        sim >= COMPONENT_REPAIR_MIN_COMBINED_SIM
        and float(mean_sim) >= COMPONENT_REPAIR_MIN_MEAN_SIM
        and float(topk_sim) >= COMPONENT_REPAIR_MIN_TOPK_SIM
    )

    # Strong support can allow a repair merge even when there is only one pair,
    # but only with very high combined and top-k evidence. This remains rare.
    strong_support = (
        sim >= COMPONENT_REPAIR_STRONG_SINGLE_COMBINED_SIM
        and float(topk_sim) >= COMPONENT_REPAIR_STRONG_SINGLE_TOPK_SIM
    )

    if not normal_support and not strong_support:
        return False, {
            "reason": "repair_similarity_too_low",
            "similarity": round(sim, 4),
            "mean_similarity": mean_sim,
            "topk_similarity": topk_sim,
            "gap_frames": gap,
            "spatial_distance": round(spatial_dist, 4),
        }

    return True, {
        "reason": "repair_pair_support",
        "similarity": round(sim, 4),
        "mean_similarity": mean_sim,
        "topk_similarity": topk_sim,
        "gap_frames": gap,
        "spatial_distance": round(spatial_dist, 4),
        "strong_support": bool(strong_support),
    }


def component_repair_evidence(component_a: set, component_b: set, track_infos: Dict[int, TrackInfo]) -> Tuple[bool, dict]:
    """
    Decide whether two current components should be merged during Stage 2.

    The repair pass is designed to recover missed same-player fragments that
    Stage 1 did not merge. To keep it safe, it prefers merging a small source
    component into an already established target component, and it requires
    multiple supporting pairwise links between the two components.
    """
    size_a = len(component_a)
    size_b = len(component_b)

    # Do not let two large components merge in repair. If both are large, one
    # wrong decision creates many false same-player pairs.
    if min(size_a, size_b) > COMPONENT_REPAIR_MAX_SOURCE_COMPONENT_SIZE:
        return False, {"reason": "repair_both_components_too_large"}

    # At least one side should already be a small identity component created by
    # Stage 1; otherwise repair is just pairwise merging again.
    if max(size_a, size_b) < COMPONENT_REPAIR_MIN_TARGET_COMPONENT_SIZE:
        return False, {"reason": "repair_no_established_target_component"}

    if BLOCK_COMPONENT_TIME_CONFLICTS and component_has_any_time_conflict(component_a, component_b, track_infos):
        return False, {"reason": "repair_component_time_conflict"}

    support_pairs = []
    strong_support_pairs = []

    for tid_a in component_a:
        for tid_b in component_b:
            ok, dbg = repair_pair_evidence(track_infos[tid_a], track_infos[tid_b])
            if not ok:
                continue
            pair_row = {
                "track_id_a": tid_a,
                "track_id_b": tid_b,
                **dbg,
            }
            support_pairs.append(pair_row)
            if dbg.get("strong_support", False):
                strong_support_pairs.append(pair_row)

    # Main rule: require at least two supporting pair links. This is the key
    # difference from normal pairwise merging and should reduce false positives.
    enough_multi_support = len(support_pairs) >= COMPONENT_REPAIR_MIN_SUPPORT_PAIRS

    # Backup rule: allow one extremely strong support only when one side is a
    # singleton. This catches isolated fragments with very clear visual evidence.
    singleton_strong_support = (
        len(strong_support_pairs) >= 1
        and min(size_a, size_b) == 1
    )

    if not enough_multi_support and not singleton_strong_support:
        return False, {
            "reason": "repair_not_enough_component_support",
            "num_support_pairs": len(support_pairs),
            "num_strong_support_pairs": len(strong_support_pairs),
        }

    best = max(support_pairs, key=lambda x: x["similarity"])
    avg_sim = float(np.mean([p["similarity"] for p in support_pairs]))
    avg_topk = float(np.mean([p["topk_similarity"] for p in support_pairs if p.get("topk_similarity") is not None]))

    return True, {
        "reason": "repair_component_supported",
        "component_size_a": size_a,
        "component_size_b": size_b,
        "num_support_pairs": len(support_pairs),
        "num_strong_support_pairs": len(strong_support_pairs),
        "avg_support_similarity": round(avg_sim, 4),
        "avg_support_topk_similarity": round(avg_topk, 4),
        "best_support_pair": best,
        "support_pairs_preview": support_pairs[:8],
    }


def run_component_repair_pass(
    uf: UnionFind,
    tids: List[int],
    track_infos: Dict[int, TrackInfo],
    candidates: List[dict],
) -> None:
    """
    Stage 2: component-level repair pass.

    This function modifies the UnionFind object in-place and appends synthetic
    repair rows to candidates for debugging/reporting.

    It performs a small number of rounds. Each round scans current components
    and merges the strongest safe repair candidate. Multiple rounds allow a
    component to gradually absorb missed fragments, but the round limit prevents
    uncontrolled growth.
    """
    if not ENABLE_COMPONENT_REPAIR_PASS:
        return

    print("Running Stage-2 component repair pass...")

    for repair_round in range(1, COMPONENT_REPAIR_MAX_ROUNDS + 1):
        roots = sorted({uf.find(tid) for tid in tids})
        components = [set(uf.members[root]) for root in roots]

        repair_candidates = []

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp_a = components[i]
                comp_b = components[j]
                ok, dbg = component_repair_evidence(comp_a, comp_b, track_infos)
                if not ok:
                    continue

                # Ranking: prefer stronger average support, then more support pairs.
                score = (
                    float(dbg.get("avg_support_similarity", 0.0))
                    + 0.01 * int(dbg.get("num_support_pairs", 0))
                    + 0.02 * int(dbg.get("num_strong_support_pairs", 0))
                )
                repair_candidates.append((score, comp_a, comp_b, dbg))

        if not repair_candidates:
            print(f"  Repair round {repair_round}: no safe component repairs found")
            break

        repair_candidates.sort(key=lambda x: x[0], reverse=True)

        merges_this_round = 0
        used_roots = set()

        for score, comp_a, comp_b, dbg in repair_candidates:
            # Components may have changed after earlier repair merges in this round.
            root_a = uf.find(next(iter(comp_a)))
            root_b = uf.find(next(iter(comp_b)))
            if root_a == root_b:
                continue
            if root_a in used_roots or root_b in used_roots:
                continue

            current_a = uf.get_members(next(iter(comp_a)))
            current_b = uf.get_members(next(iter(comp_b)))
            ok_now, dbg_now = component_repair_evidence(current_a, current_b, track_infos)
            if not ok_now:
                continue

            # Use the best support pair as representative in the report row.
            best_pair = dbg_now.get("best_support_pair", {})
            tid_a = int(best_pair.get("track_id_a", min(current_a)))
            tid_b = int(best_pair.get("track_id_b", min(current_b)))

            uf.union(tid_a, tid_b)
            used_roots.add(root_a)
            used_roots.add(root_b)
            merges_this_round += 1

            candidates.append({
                "track_id_a": tid_a,
                "track_id_b": tid_b,
                "label": track_infos[tid_a].label,
                "similarity": best_pair.get("similarity", dbg_now.get("avg_support_similarity")),
                "mean_similarity": best_pair.get("mean_similarity"),
                "topk_similarity": best_pair.get("topk_similarity"),
                "similarity_mode": "stage2_component_repair",
                "gap_frames": best_pair.get("gap_frames"),
                "overlap": False,
                "overlap_frames": 0,
                "threshold_used": None,
                "effective_threshold_used": None,
                "passed_threshold": True,
                "spatial_pass": True,
                "spatial_distance": best_pair.get("spatial_distance"),
                "spatial_threshold_used": COMPONENT_REPAIR_MAX_SPATIAL_DIST,
                "team_id_a": track_infos[tid_a].team_id,
                "team_id_b": track_infos[tid_b].team_id,
                "min_track_len": min(track_infos[tid_a].count, track_infos[tid_b].count),
                "max_track_len": max(track_infos[tid_a].count, track_infos[tid_b].count),
                "mutual_best": False,
                "merged": True,
                "blocked_reason": None,
                "merge_stage": "component_repair_pass",
                "repair_round": repair_round,
                "repair_score": round(float(score), 4),
                "repair_debug": dbg_now,
            })

        print(f"  Repair round {repair_round}: merged {merges_this_round} component pairs")
        if merges_this_round == 0:
            break

# ============================================================================
# Linking
# ============================================================================
def merge_tracklets_v2(track_infos: Dict[int, TrackInfo]) -> Tuple[Dict[int, int], List[dict]]:
    """
    Stable V1-style linking with two additions:
    - team mismatch blocking
    - referee-like blocking
    """
    tids = sorted(track_infos.keys())
    candidates: List[dict] = []

    min_report_thr = min(SIM_THR_SHORT, SIM_THR_MID, SIM_THR_LONG) - 0.05

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            tid_a = tids[i]
            tid_b = tids[j]
            a = track_infos[tid_a]
            b = track_infos[tid_b]

            passed, dbg = pair_passes_basic_gates(a, b)

            sim = dbg.get("similarity", pairwise_track_similarity(a, b)[0])
            gap = dbg.get("gap_frames", temporal_gap(a, b))
            overlap = dbg.get("overlap", temporal_overlap(a, b))

            candidate = {
                "track_id_a": tid_a,
                "track_id_b": tid_b,
                "label": a.label,
                "similarity": round(sim, 4),
                "mean_similarity": dbg.get("mean_similarity", None),
                "topk_similarity": dbg.get("topk_similarity", None),
                "similarity_mode": dbg.get("similarity_mode", None),
                "gap_frames": gap,
                "overlap": overlap,
                "overlap_frames": dbg.get("overlap_frames", None),
                "threshold_used": dbg.get("threshold_used", None),
                "effective_threshold_used": dbg.get("effective_threshold_used", None),
                "passed_threshold": dbg.get("reason") != "below_threshold",
                "spatial_pass": dbg.get("reason") != "failed_spatial_gate",
                "spatial_distance": dbg.get("spatial_distance", None),
                "spatial_threshold_used": dbg.get("spatial_threshold_used", None),
                "team_id_a": dbg.get("team_id_a", a.team_id),
                "team_id_b": dbg.get("team_id_b", b.team_id),
                "min_track_len": min(a.count, b.count),
                "max_track_len": max(a.count, b.count),
                "mutual_best": False,
                "merged": False,
                "blocked_reason": None if passed else dbg.get("reason"),
            }

            if sim >= min_report_thr or candidate["blocked_reason"] in {"team_mismatch", "referee_like_track"}:
                candidates.append(candidate)

    best_for = {}
    for c in candidates:
        if not c["passed_threshold"]:
            continue
        if not c["spatial_pass"]:
            continue
        if c["blocked_reason"] in {"team_mismatch", "referee_like_track"}:
            continue

        a = c["track_id_a"]
        b = c["track_id_b"]
        sim = c["similarity"]

        if a not in best_for or sim > best_for[a]["similarity"]:
            best_for[a] = {"other": b, "similarity": sim}
        if b not in best_for or sim > best_for[b]["similarity"]:
            best_for[b] = {"other": a, "similarity": sim}

    for c in candidates:
        if not c["passed_threshold"]:
            continue
        if not c["spatial_pass"]:
            continue
        if c["blocked_reason"] in {"team_mismatch", "referee_like_track"}:
            continue

        a = c["track_id_a"]
        b = c["track_id_b"]

        mutual = (
            a in best_for and
            b in best_for and
            best_for[a]["other"] == b and
            best_for[b]["other"] == a
        )

        if not USE_MUTUAL_BEST_MATCH:
            mutual = True

        c["mutual_best"] = mutual

    uf = UnionFind(tids)

    merge_order = sorted(
        [c for c in candidates if is_merge_eligible(c)],
        key=lambda x: x["similarity"],
        reverse=True,
    )

    for c in merge_order:
        a_tid = c["track_id_a"]
        b_tid = c["track_id_b"]

        ra = uf.find(a_tid)
        rb = uf.find(b_tid)
        if ra == rb:
            c["blocked_reason"] = "already_same_component"
            continue

        comp_a = uf.get_members(a_tid)
        comp_b = uf.get_members(b_tid)

        # Component-size penalty: if one component is already large, require a
        # stronger similarity score before adding another tracklet. This mainly
        # improves Precision on long videos.
        required_sim = candidate_required_similarity_for_component(c, comp_a, comp_b)
        penalty = component_size_penalty(comp_a, comp_b)
        c["component_size_penalty"] = round(penalty, 4)
        c["component_required_similarity"] = round(required_sim, 4)
        if c["similarity"] < required_sim:
            c["blocked_reason"] = "component_size_penalty"
            continue

        if BLOCK_COMPONENT_TIME_CONFLICTS:
            if merged_component_has_time_conflict(comp_a, comp_b, track_infos):

                # Allow duplicate overlap cases
                a = track_infos[a_tid]
                b = track_infos[b_tid]

                if is_duplicate_overlap_case(a, b, c):
                    pass  # allow merge
                else:
                    c["blocked_reason"] = "component_time_conflict"
                    continue

        uf.union(a_tid, b_tid)
        c["merged"] = True
        c["blocked_reason"] = None

    for c in candidates:
        if c["merged"]:
            continue
        if c["blocked_reason"] is not None:
            continue

        if not c["passed_threshold"]:
            c["blocked_reason"] = "below_threshold"
        elif not c["spatial_pass"]:
            c["blocked_reason"] = "failed_spatial_gate"
        elif not is_merge_eligible(c):
            c["blocked_reason"] = "not_merge_eligible"
        else:
            c["blocked_reason"] = "not_merged"

    # ------------------------------------------------------------------
    # Stage 2: component-level repair pass
    # ------------------------------------------------------------------
    # Stage 1 above is intentionally conservative. Now that initial identity
    # components exist, we try to recover missed fragments using component-level
    # evidence. This pass only merges when several links support the same
    # component-level decision, which is safer than lowering global thresholds.
    run_component_repair_pass(
        uf=uf,
        tids=tids,
        track_infos=track_infos,
        candidates=candidates,
    )

    root_to_gid = {}
    next_gid = 1
    trackid_to_globalid = {}

    for tid in tids:
        root = uf.find(tid)
        if root not in root_to_gid:
            root_to_gid[root] = next_gid
            next_gid += 1
        trackid_to_globalid[tid] = root_to_gid[root]

    merge_candidates_report = sorted(
        candidates,
        key=lambda x: (x["merged"], x["similarity"]),
        reverse=True,
    )
    return trackid_to_globalid, merge_candidates_report


# ============================================================================
# Save outputs
# ============================================================================
def save_observations(all_tracks: List[dict], trackid_to_globalid: Dict[int, int], out_path: Path, ball_detections: Optional[List[dict]] = None) -> None:
    _backup_if_exists(out_path)

    observations = []
    for i, t in enumerate(all_tracks):
        tid = int(t["track_id"])
        gid = trackid_to_globalid.get(tid)

        observations.append({
            "obs_index": i,
            "frame_index": int(t["frame_index"]),
            "track_id": tid,
            "global_id": int(gid) if gid is not None else None,
            "bbox_xyxy": t["bbox_xyxy"],
            "score": float(t.get("score", 0.0)),
            "label": int(t["label"]),
        })

    payload = {
        "video": str(VIDEO_PATH),
        "num_observations": len(observations),
        "num_ball_detections": int(len(ball_detections or [])),
        "ball_policy": "sidecar_only: ReID observations and trackid_to_globalid are unchanged; all raw label=0 detections are copied from tracking output",
        "ball_detections": ball_detections or [],
        "observations": observations,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved observation-level ReID output: {out_path}")


def save_trackid_to_globalid(trackid_to_globalid: Dict[int, int], out_path: Path) -> None:
    _backup_if_exists(out_path)

    payload = {
        "trackid_to_globalid": {str(k): int(v) for k, v in sorted(trackid_to_globalid.items())}
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved track->global mapping: {out_path}")


def save_report(
    raw_tracks_meta: dict,
    reid_tracks: List[dict],
    valid_tracks: List[dict],
    track_infos: Dict[int, TrackInfo],
    trackid_to_globalid: Dict[int, int],
    merge_candidates_report: List[dict],
    out_path: Path,
) -> None:
    _backup_if_exists(out_path)

    num_local_tracks = len(track_infos)
    num_global_ids = len(set(trackid_to_globalid.values()))
    num_merges = num_local_tracks - num_global_ids

    by_gid = defaultdict(list)
    for tid, gid in trackid_to_globalid.items():
        by_gid[gid].append(tid)

    groups = []
    for gid, tids in sorted(by_gid.items()):
        groups.append({
            "global_id": gid,
            "track_ids": sorted(tids),
            "num_tracks_merged": len(tids),
        })

    track_summaries = []
    for tid, info in sorted(track_infos.items()):
        track_summaries.append({
            "track_id": tid,
            "label": info.label,
            "frames": [info.start_frame, info.end_frame],
            "length": info.count,
            "team_id": info.team_id,
            "team_confidence": info.team_confidence,
            "referee_like": info.referee_like,
            "min_team_color_dist": info.min_team_color_dist,
        })

    report = {
        "video": str(VIDEO_PATH),
        "tracks_json": str(TRACKS_PATH),
        "team_assignment_json": str(TEAM_ASSIGNMENT_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "prtreid_dir": str(PRTREID_DIR),
        "config": {
            "valid_reid_labels": sorted(list(VALID_REID_LABELS)),
            "batch_size": BATCH_SIZE,
            "max_gap": MAX_GAP,
            "short_gap_thr": SHORT_GAP_THR,
            "mid_gap_thr": MID_GAP_THR,
            "long_gap_thr": LONG_GAP_THR,
            "very_long_gap_thr": VERY_LONG_GAP_THR,
            "sim_thr_short": SIM_THR_SHORT,
            "sim_thr_mid": SIM_THR_MID,
            "sim_thr_long": SIM_THR_LONG,
            "sim_thr_very_long": SIM_THR_VERY_LONG,
            "sim_thr_ultra_long": SIM_THR_ULTRA_LONG,
            "use_multi_crop_similarity": USE_MULTI_CROP_SIMILARITY,
            "component_size_penalty_enabled": COMPONENT_SIZE_PENALTY_ENABLED,
            "component_size_penalty_start": COMPONENT_SIZE_PENALTY_START,
            "component_size_penalty_max": COMPONENT_SIZE_PENALTY_MAX,
            "non_mutual_high_sim_max_gap": NON_MUTUAL_HIGH_SIM_MAX_GAP,
            "non_mutual_very_high_sim": NON_MUTUAL_VERY_HIGH_SIM,
            "non_mutual_very_high_topk": NON_MUTUAL_VERY_HIGH_TOPK,
            "enable_very_long_gap_guard": ENABLE_VERY_LONG_GAP_GUARD,
            "very_long_gap_min_mean_sim": VERY_LONG_GAP_MIN_MEAN_SIM,
            "very_long_gap_min_topk_sim": VERY_LONG_GAP_MIN_TOPK_SIM,
            "enable_fragment_reentry_relaxation": ENABLE_FRAGMENT_REENTRY_RELAXATION,
            "fragment_max_len": FRAGMENT_MAX_LEN,
            "fragment_reentry_max_gap": FRAGMENT_REENTRY_MAX_GAP,
            "fragment_reentry_min_topk_sim": FRAGMENT_REENTRY_MIN_TOPK_SIM,
            "fragment_reentry_min_mean_sim": FRAGMENT_REENTRY_MIN_MEAN_SIM,
            "fragment_reentry_max_spatial_dist": FRAGMENT_REENTRY_MAX_SPATIAL_DIST,
            "enable_component_repair_pass": ENABLE_COMPONENT_REPAIR_PASS,
            "component_repair_max_rounds": COMPONENT_REPAIR_MAX_ROUNDS,
            "component_repair_max_gap": COMPONENT_REPAIR_MAX_GAP,
            "component_repair_min_support_pairs": COMPONENT_REPAIR_MIN_SUPPORT_PAIRS,
            "component_repair_min_target_component_size": COMPONENT_REPAIR_MIN_TARGET_COMPONENT_SIZE,
            "component_repair_max_source_component_size": COMPONENT_REPAIR_MAX_SOURCE_COMPONENT_SIZE,
            "component_repair_min_combined_sim": COMPONENT_REPAIR_MIN_COMBINED_SIM,
            "component_repair_min_mean_sim": COMPONENT_REPAIR_MIN_MEAN_SIM,
            "component_repair_min_topk_sim": COMPONENT_REPAIR_MIN_TOPK_SIM,
            "component_repair_max_spatial_dist": COMPONENT_REPAIR_MAX_SPATIAL_DIST,
            "block_low_confidence_team_on_long_gap": BLOCK_LOW_CONFIDENCE_TEAM_ON_LONG_GAP,
            "low_team_confidence_for_reid": LOW_TEAM_CONFIDENCE_FOR_REID,
            "spatial_gate_short": SPATIAL_GATE_SHORT,
            "spatial_gate_mid": SPATIAL_GATE_MID,
            "spatial_gate_long": SPATIAL_GATE_LONG,
            "block_different_teams": BLOCK_DIFFERENT_TEAMS,
            "block_referee_like_tracks": BLOCK_REFEREE_LIKE_TRACKS,
        },
        "input_summary": {
            "fps": raw_tracks_meta.get("fps"),
            "total_frames": raw_tracks_meta.get("total_frames"),
            "num_detection_frames_in": raw_tracks_meta.get("num_detection_frames_in"),
            "num_detections_in": raw_tracks_meta.get("num_detections_in"),
            "num_tracked_rows": raw_tracks_meta.get("num_tracked_rows"),
            "num_reid_observations_before_crop_filter": len(reid_tracks),
            "num_reid_observations_after_crop_filter": len(valid_tracks),
        },
        "reid_summary": {
            "num_local_track_ids": num_local_tracks,
            "num_global_ids": num_global_ids,
            "num_merges": num_merges,
        },
        "track_summaries": track_summaries,
        "groups": groups,
        "merge_candidates": merge_candidates_report[:300],
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved ReID report: {out_path}")


def save_embeddings(
    valid_tracks: List[dict],
    obs_embeddings: torch.Tensor,
    track_infos: Dict[int, TrackInfo],
    out_path: Path,
) -> None:
    _backup_if_exists(out_path)

    payload = {
        "obs_embeddings": obs_embeddings,
        "obs_track_ids": torch.tensor([int(t["track_id"]) for t in valid_tracks], dtype=torch.long),
        "obs_frame_indices": torch.tensor([int(t["frame_index"]) for t in valid_tracks], dtype=torch.long),
        "track_mean_embeddings": {int(tid): info.mean_embedding for tid, info in track_infos.items()},
    }

    torch.save(payload, out_path)
    print(f"Saved embeddings: {out_path}")


# ============================================================================
# Overlay video
# ============================================================================
# def save_overlay_video(
#     video_path: Path,
#     all_tracks: List[dict],
#     trackid_to_globalid: Dict[int, int],
#     out_path: Path,
# ) -> None:
#     """
#     Save a video with local track_id + merged global_id drawn on each box.
#     """
#     _backup_if_exists(out_path)

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

#     unique_gids = sorted(set(trackid_to_globalid.values()))
#     np.random.seed(42)
#     gid_colors = {}
#     for gid in unique_gids:
#         gid_colors[gid] = tuple(int(c) for c in np.random.randint(50, 255, size=3))

#     frame_to_tracks = defaultdict(list)
#     for t in all_tracks:
#         frame_to_tracks[int(t["frame_index"])].append(t)

#     for frame_idx in range(total):
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_idx % 100 == 0:
#             print(f"Video overlay progress: frame {frame_idx}/{total}", flush=True)

#         entries = frame_to_tracks.get(frame_idx, [])
#         for t in entries:
#             tid = int(t["track_id"])

#             label = int(t.get("label", -1))

#             if label in BALL_LABELS:
#                 gid = None
#             else:
#                 gid = trackid_to_globalid.get(tid)
#                 if gid is None:
#                     continue

#             # label = int(t.get("label", -1))

#             # if label == 0:
#             #     gid = None
#             # else:
#             #     gid = trackid_to_globalid.get(tid)
#             #     if gid is None:
#             #         continue

#             x1, y1, x2, y2 = [int(v) for v in t["bbox_xyxy"]]
#             x1 = max(0, x1)
#             y1 = max(0, y1)
#             x2 = min(w, x2)
#             y2 = min(h, y2)

#             if label in BALL_LABELS:
#                 color = BALL_COLOR
#                 label_txt = "BALL"
#             else:
#                 color = gid_colors[gid]
#                 label_txt = f"TID:{tid} GID:{gid}"

#             # if label == 0:
#             #     color = (0, 255, 0)
#             #     label_txt = "BALL"
#             # else:
#             #     color = gid_colors[gid]
#             #     label_txt = f"TID:{tid} GID:{gid}"

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 0.45
#             thickness = 1
#             (tw, th), _ = cv2.getTextSize(label_txt, font, font_scale, thickness)

#             y_top = max(0, y1 - th - 6)
#             cv2.rectangle(frame, (x1, y_top), (x1 + tw + 4, y1), color, -1)

#             brightness = sum(color) / 3
#             text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
#             cv2.putText(frame, label_txt, (x1 + 2, y1 - 4), font, font_scale, text_color, thickness)

#         cv2.putText(
#             frame,
#             f"Frame {frame_idx}/{total}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2,
#         )

#         out.write(frame)

#     cap.release()
#     out.release()
#     print(f"Saved overlay video: {out_path}")


def save_overlay_video(
    video_path: Path,
    all_tracks: List[dict],
    trackid_to_globalid: Dict[int, int],
    out_path: Path,
) -> None:
    """
    Save a video with local track_id + merged global_id drawn on each player.
    Uses a static open ellipse under each player instead of bounding boxes.
    """
    _backup_if_exists(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    unique_gids = sorted(set(trackid_to_globalid.values()))
    np.random.seed(42)

    gid_colors = {}
    for gid in unique_gids:
        gid_colors[gid] = tuple(int(c) for c in np.random.randint(50, 255, size=3))

    frame_to_tracks = defaultdict(list)
    for t in all_tracks:
        frame_to_tracks[int(t["frame_index"])].append(t)

    # Players
    ELLIPSE_W = 80
    ELLIPSE_H = 25
    ELLIPSE_THICKNESS = 5

    # Ball
    BALL_ELLIPSE_W = 42
    BALL_ELLIPSE_H = 14
    BALL_ELLIPSE_THICKNESS = 5

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 100 == 0:
            print(f"Video overlay progress: frame {frame_idx}/{total}", flush=True)

        entries = frame_to_tracks.get(frame_idx, [])

        for t in entries:
            tid = int(t["track_id"])
            label = int(t.get("label", -1))

            if label == 0:
                gid = None
            else:
                gid = trackid_to_globalid.get(tid)
                if gid is None:
                    continue

            x1, y1, x2, y2 = [int(v) for v in t["bbox_xyxy"]]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if label == 0:
                color = (0, 255, 0)
                label_txt = "BALL"
                ellipse_w = BALL_ELLIPSE_W
                ellipse_h = BALL_ELLIPSE_H
                ellipse_thickness = BALL_ELLIPSE_THICKNESS
            else:
                color = gid_colors[gid]
                label_txt = f"TID:{tid} GID:{gid}"
                ellipse_w = ELLIPSE_W
                ellipse_h = ELLIPSE_H
                ellipse_thickness = ELLIPSE_THICKNESS

            cx = int((x1 + x2) / 2)
            cy = int(y2)

            cv2.ellipse(
                frame,
                (cx, cy),
                (ellipse_w, ellipse_h),
                0,
                15,
                345,
                color,
                ellipse_thickness,
                cv2.LINE_AA,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (tw, th), _ = cv2.getTextSize(
                label_txt,
                font,
                font_scale,
                thickness,
            )

            text_x = max(0, min(w - tw - 6, cx - tw // 2))
            text_y = min(h - 4, cy + ellipse_h + th + 10)

            box_y1 = max(0, text_y - th - 4)
            box_y2 = min(h, text_y + 4)

            cv2.rectangle(
                frame,
                (text_x, box_y1),
                (text_x + tw + 4, box_y2),
                color,
                -1,
            )

            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            cv2.putText(
                frame,
                label_txt,
                (text_x + 2, text_y),
                font,
                font_scale,
                text_color,
                thickness,
            )

        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        out.write(frame)

    cap.release()
    out.release()

    print(f"Saved overlay video: {out_path}")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    print("=" * 72)
    print("PRTREID REID STAGE V2")
    print("=" * 72)
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", type=str, default=None, help="Optional override video path")
    args, _ = parser.parse_known_args()

    # # Allow overriding the global VIDEO_PATH from CLI while keeping default otherwise.
    # if args.video:
    #     global VIDEO_PATH
    #     VIDEO_PATH = Path(args.video)

    # Force VIDEO_PATH to the CLI parameter (required)
    global VIDEO_PATH
    VIDEO_PATH = Path(args.video)

    print(f"Video path: {VIDEO_PATH}")

    _ensure_paths_exist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_tracks_meta, all_tracks, reid_tracks = load_tracks(TRACKS_PATH)
    team_assignment_by_tid = load_team_assignment(TEAM_ASSIGNMENT_PATH)

    valid_indices, valid_crops = crop_persons_from_video(VIDEO_PATH, reid_tracks)
    valid_tracks = [reid_tracks[i] for i in valid_indices]

    model, cfg, img_h, img_w = build_model_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        hrnet_pretrained_path=HRNET_PRETRAINED_PATH,
        device=device,
        prtreid_dir=PRTREID_DIR,
    )

    print("\nExtracting ReID embeddings...")
    obs_embeddings = extract_features(
        model=model,
        crops=valid_crops,
        img_h=img_h,
        img_w=img_w,
        batch_size=BATCH_SIZE,
        device=device,
    )

    track_infos = build_track_summaries(
        valid_tracks=valid_tracks,
        embeddings=obs_embeddings,
        team_assignment_by_tid=team_assignment_by_tid,
    )

    print("\nLinking tracklets with V1 logic + team constraints...")
    trackid_to_globalid, merge_candidates_report = merge_tracklets_v2(track_infos)

    num_local = len(track_infos)
    num_global = len(set(trackid_to_globalid.values()))
    num_merges = num_local - num_global
    print(f"Local track IDs: {num_local}")
    print(f"Global IDs after merging: {num_global}")
    print(f"Merges performed: {num_merges}")

    obs_out_path = OUT_DIR / "reid_observations.json"
    map_out_path = OUT_DIR / "trackid_to_globalid.json"
    report_out_path = OUT_DIR / "reid_report.json"
    emb_out_path = OUT_DIR / "reid_embeddings.pt"

    save_observations(all_tracks, trackid_to_globalid, obs_out_path, raw_tracks_meta.get("ball_detections", []))
    save_trackid_to_globalid(trackid_to_globalid, map_out_path)
    save_report(
        raw_tracks_meta=raw_tracks_meta,
        reid_tracks=reid_tracks,
        valid_tracks=valid_tracks,
        track_infos=track_infos,
        trackid_to_globalid=trackid_to_globalid,
        merge_candidates_report=merge_candidates_report,
        out_path=report_out_path,
    )
    save_embeddings(valid_tracks, obs_embeddings, track_infos, emb_out_path)

    if SAVE_OVERLAY_VIDEO:
        overlay_out_path = OUT_DIR / OVERLAY_VIDEO_NAME
        save_overlay_video(VIDEO_PATH, all_tracks, trackid_to_globalid, overlay_out_path)

    print("=" * 72)
    print("PRTREID REID STAGE V2 COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()