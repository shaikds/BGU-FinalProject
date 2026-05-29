"""
team_assignment_v2.py

A new, separate team-assignment stage inspired by the football-ai notebook/video.

Important design choice:
------------------------
This file DOES NOT change detection or tracking.
It consumes the existing tracking output:

    outputs/tracks/tracks.json

and produces a new independent output directory:

    outputs/team_assignment_v2/

Pipeline position:
------------------
Object Detection -> Tracking -> Team Assignment V2 -> ReID

Main idea:
----------
Instead of assigning teams using only a simple LAB mean color descriptor,
this version learns team clusters from many player crops using an image
embedding model, then assigns each local track_id to a team by voting over
its crops.

Preferred embedding backend:
----------------------------
1. SigLIP from Hugging Face transformers, if installed and available.
2. Fallback to torchvision ResNet18 embeddings, if transformers/SigLIP is not available.
3. Fallback to color descriptors only, if neither deep model can be loaded.

Why fallback exists:
--------------------
On the cluster, installing transformers / downloading SigLIP may not always be
available. This lets us still run the script and compare.

Outputs:
--------
outputs/team_assignment_v2/team_assignment_v2.json
outputs/team_assignment_v2/team_assignment_v2_overlay.mp4
outputs/team_assignment_v2/team_assignment_v2_debug.json

Expected input labels from tracks.json:
---------------------------------------
The pipeline use:
    label=1 -> player
    label=3 -> goalkeeper
"""

import json
import math
import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np


# =============================================================================
# Paths
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data" / "seconds_video.mp4"
TRACKS_PATH = ROOT / "outputs" / "tracks" / "tracks.json"
OUT_DIR = ROOT / "outputs" / "team_assignment_v2"


# =============================================================================
# Labels from your tracking output
# =============================================================================
PLAYER_LABELS = {1}
GOALKEEPER_LABELS = {3}
TEAM_ASSIGNMENT_LABELS = PLAYER_LABELS | GOALKEEPER_LABELS
BALL_LABELS = {0}
PASS_THROUGH_LABELS = BALL_LABELS

# =============================================================================
# Config
# =============================================================================
BACKUP_ON_OVERWRITE = True
SAVE_OVERLAY_VIDEO = True # Set True only when you need the overlay video
OVERLAY_VIDEO_NAME = "team_assignment_v2_overlay.mp4"

# Crop sampling for fitting the team classifier.
# We use existing tracking rows, not a new detector.
TRAIN_STRIDE_FRAMES = 50          # approximately 1 crop round per second at 25 FPS
MAX_TRAIN_CROPS = 1500            # cap for speed/memory
MIN_TRAIN_CROP_H = 35
MIN_TRAIN_CROP_W = 15

# Track-level voting.
MAX_CROPS_PER_TRACK_FOR_PREDICT = 50
MIN_CROPS_PER_TRACK_FOR_CONFIDENT_TEAM = 3
MIN_TEAM_VOTE_CONFIDENCE = 0.58

# Exclude extremely short/noisy tracks from training. They can still be predicted.
MIN_PLAYER_TRACK_LEN_FOR_TRAIN = 20

# UMAP/KMeans settings.
USE_UMAP_IF_AVAILABLE = True
UMAP_N_COMPONENTS = 3
KMEANS_N_CLUSTERS = 2
RANDOM_SEED = 7

# Goalkeeper assignment method.
# The notebook assigns GK by distance to per-frame team centroids.
# Here we do a track-level version using average field position from player tracks.
ASSIGN_GOALKEEPERS_BY_POSITION = True

# Referee-like detection is intentionally conservative here.
# We do NOT remove anything from tracks.json. We only flag suspicious outliers.
ENABLE_REFEREE_LIKE_FLAG = True
REFEREE_MIN_TRACK_LEN = 5
REFEREE_LOW_CONFIDENCE_THRESHOLD = 0.55
REFEREE_DARK_LAB_L_THRESHOLD = 90.0

# Drawing colors in BGR for OpenCV.
TEAM_COLORS = {
    0: (255, 80, 80),   # team 0
    1: (80, 80, 255),   # team 1
}
REFEREE_COLOR = (0, 255, 255)
UNKNOWN_COLOR = (255, 255, 255)


# =============================================================================
# Utilities
# =============================================================================
def _backup_if_exists(path: Path) -> None:
    if not BACKUP_ON_OVERWRITE:
        return
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        path.replace(bak)


def _ensure_paths_exist() -> None:
    assert VIDEO_PATH.exists(), f"Missing video: {VIDEO_PATH}"
    assert TRACKS_PATH.exists(), f"Missing tracks JSON: {TRACKS_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def bbox_wh(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bottom_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def crop_from_frame(frame_bgr: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

def torso_crop_from_frame(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # upper body / jersey area
    nx1 = x1 + 0.15 * w
    nx2 = x2 - 0.15 * w
    ny1 = y1 + 0.10 * h
    ny2 = y1 + 0.60 * h

    return crop_from_frame(frame_bgr, [nx1, ny1, nx2, ny2])

def upper_body_lab_mean(crop_bgr: np.ndarray) -> np.ndarray:
    """Small helper for debug/referee-like heuristics only."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = crop_rgb.shape[:2]
    if h < 6 or w < 6:
        patch = crop_rgb
    else:
        y1 = int(0.15 * h)
        y2 = int(0.55 * h)
        x1 = int(0.20 * w)
        x2 = int(0.80 * w)
        patch = crop_rgb[y1:y2, x1:x2]
        if patch.size == 0:
            patch = crop_rgb

    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    mask = (pixels[:, 0] > 25) & (pixels[:, 0] < 240)
    if mask.any():
        pixels = pixels[mask]
    return pixels.mean(axis=0).astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class CropRecord:
    crop_bgr: np.ndarray
    track_id: int
    frame_index: int
    label: int
    bbox_xyxy: List[float]


@dataclass
class TrackAssignment:
    track_id: int
    label: int
    start_frame: int
    end_frame: int
    count: int
    team_id: Optional[int]
    team_confidence: Optional[float]
    referee_like: bool
    mean_color_lab: List[float]
    mean_center_xy: List[float]
    mean_bottom_center_xy: List[float]
    mean_bbox_wh: List[float]
    votes: Dict[str, int] = field(default_factory=dict)
    method: str = "unknown"


# =============================================================================
# Lightweight KMeans fallback
# =============================================================================
class SimpleKMeans:
    """Tiny KMeans implementation so the script can run without sklearn."""

    def __init__(self, n_clusters: int = 2, seed: int = 7, n_iter: int = 100):
        self.n_clusters = n_clusters
        self.seed = seed
        self.n_iter = n_iter
        self.cluster_centers_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        if len(X) < self.n_clusters:
            raise RuntimeError("Not enough samples for KMeans")

        init_ids = rng.choice(len(X), size=self.n_clusters, replace=False)
        centers = X[init_ids].astype(np.float32).copy()

        labels = np.zeros(len(X), dtype=np.int32)
        for _ in range(self.n_iter):
            dists = np.stack([np.linalg.norm(X - c, axis=1) for c in centers], axis=1)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)

            new_centers = []
            for k in range(self.n_clusters):
                pts = X[new_labels == k]
                if len(pts) == 0:
                    new_centers.append(centers[k])
                else:
                    new_centers.append(pts.mean(axis=0))
            new_centers = np.stack(new_centers, axis=0).astype(np.float32)

            if np.array_equal(labels, new_labels):
                centers = new_centers
                labels = new_labels
                break
            centers = new_centers
            labels = new_labels

        self.cluster_centers_ = centers
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("SimpleKMeans must be fitted first")
        dists = np.stack([np.linalg.norm(X - c, axis=1) for c in self.cluster_centers_], axis=1)
        return np.argmin(dists, axis=1).astype(np.int32)

    def distances(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("SimpleKMeans must be fitted first")
        return np.stack([np.linalg.norm(X - c, axis=1) for c in self.cluster_centers_], axis=1)


# =============================================================================
# Embedding model wrapper
# =============================================================================
class EmbeddingBackend:
    """
    Tries to mimic the notebook's SigLIP-based idea.

    Priority:
    1. SigLIP from transformers
    2. torchvision ResNet18
    3. color histogram fallback
    """

    def __init__(self, device: str):
        self.device = device
        self.backend_name = "color_hist_fallback"
        self.model = None
        self.processor = None
        self.transform = None
        self._load_backend()

    def _load_backend(self) -> None:
        # Try SigLIP first.
        try:
            import torch
            from transformers import AutoProcessor, SiglipVisionModel

            model_name = os.environ.get("SIGLIP_MODEL_PATH", "google/siglip-base-patch16-224")
            print(f"Trying to load SigLIP: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = SiglipVisionModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.backend_name = "siglip"
            print("Embedding backend: SigLIP")
            return
        except Exception as e:
            print(f"SigLIP backend unavailable, falling back. Reason: {e}")

        # Try torchvision ResNet18 second.
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            print("Trying to load torchvision ResNet18")
            weights = None
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None

            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model = model.to(self.device)
            model.eval()

            self.model = model
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.backend_name = "resnet18"
            print("Embedding backend: torchvision ResNet18")
            return
        except Exception as e:
            print(f"ResNet18 backend unavailable, using color histogram fallback. Reason: {e}")

        print("Embedding backend: color histogram fallback")
        self.backend_name = "color_hist_fallback"

    def embed(self, crops_bgr: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        if self.backend_name == "siglip":
            return self._embed_siglip(crops_bgr, batch_size=batch_size)
        if self.backend_name == "resnet18":
            return self._embed_resnet(crops_bgr, batch_size=batch_size)
        return self._embed_color_hist(crops_bgr)

    def _embed_siglip(self, crops_bgr: List[np.ndarray], batch_size: int) -> np.ndarray:
        import torch
        from PIL import Image

        all_embeddings = []
        for start in range(0, len(crops_bgr), batch_size):
            batch = crops_bgr[start:start + batch_size]
            images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in batch]
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                emb = torch.mean(outputs.last_hidden_state, dim=1).detach().cpu().numpy()
            all_embeddings.append(emb)
        X = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return l2_normalize(X)

    def _embed_resnet(self, crops_bgr: List[np.ndarray], batch_size: int) -> np.ndarray:
        import torch

        all_embeddings = []
        for start in range(0, len(crops_bgr), batch_size):
            batch = crops_bgr[start:start + batch_size]
            tensors = []
            for c in batch:
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(rgb))
            x = torch.stack(tensors, dim=0).to(self.device)
            with torch.no_grad():
                emb = self.model(x).detach().cpu().numpy()
            all_embeddings.append(emb)
        X = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return l2_normalize(X)

    def _embed_color_hist(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Color-only fallback. Not as good as SigLIP, but useful if deep models
        are unavailable on the cluster.
        """
        feats = []
        for crop in crops_bgr:
            h, w = crop.shape[:2]
            if h >= 6 and w >= 6:
                y1 = int(0.15 * h)
                y2 = int(0.55 * h)
                x1 = int(0.20 * w)
                x2 = int(0.80 * w)
                patch = crop[y1:y2, x1:x2]
                if patch.size == 0:
                    patch = crop
            else:
                patch = crop

            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [24, 12], [0, 180, 0, 256]).astype(np.float32)
            hist = hist.flatten()
            hist /= (hist.sum() + 1e-6)
            feats.append(hist)
        X = np.stack(feats, axis=0).astype(np.float32)
        return l2_normalize(X)


# =============================================================================
# Team classifier
# =============================================================================
class TeamClassifierV2:
    """
    Similar spirit to the notebook's TeamClassifier:
    - extract image embeddings for player crops
    - optionally project with UMAP
    - cluster into 2 teams
    - predict team for new crops
    """

    def __init__(self, device: str):
        self.device = device
        self.embedding_backend = EmbeddingBackend(device=device)
        self.reducer = None
        self.cluster_model = None
        self.use_projection = False
        self.debug_info: Dict[str, Any] = {}

    def fit(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        print(f"Fitting TeamClassifierV2 on {len(crops_bgr)} crops")
        X = self.embedding_backend.embed(crops_bgr)
        X_for_cluster = X

        self.use_projection = False
        if USE_UMAP_IF_AVAILABLE:
            try:
                import umap
                print("Using UMAP projection before KMeans")
                self.reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, random_state=RANDOM_SEED)
                X_for_cluster = self.reducer.fit_transform(X).astype(np.float32)
                self.use_projection = True
            except Exception as e:
                print(f"UMAP unavailable, clustering directly on embeddings. Reason: {e}")
                self.reducer = None

        try:
            from sklearn.cluster import KMeans
            print("Using sklearn KMeans")
            self.cluster_model = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
            print("KMeans starting")
            labels = self.cluster_model.fit_predict(X_for_cluster).astype(np.int32)
            print("KMeans finished")
        except Exception as e:
            print(f"sklearn KMeans unavailable, using SimpleKMeans. Reason: {e}")
            self.cluster_model = SimpleKMeans(n_clusters=KMEANS_N_CLUSTERS, seed=RANDOM_SEED)
            print("KMeans starting")
            labels = self.cluster_model.fit_predict(X_for_cluster).astype(np.int32)
            print("KMeans finished")

        counts = Counter(labels.tolist())
        self.debug_info = {
            "embedding_backend": self.embedding_backend.backend_name,
            "used_umap": self.use_projection,
            "train_crop_count": len(crops_bgr),
            "cluster_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        }
        return labels

    def predict(self, crops_bgr: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not crops_bgr:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        X = self.embedding_backend.embed(crops_bgr)
        X_for_cluster = X
        if self.use_projection and self.reducer is not None:
            X_for_cluster = self.reducer.transform(X).astype(np.float32)

        labels = self.cluster_model.predict(X_for_cluster).astype(np.int32)

        # Confidence approximation from distance to cluster centers.
        confidence = np.ones(len(labels), dtype=np.float32)
        try:
            if hasattr(self.cluster_model, "transform"):
                dists = self.cluster_model.transform(X_for_cluster)
            elif hasattr(self.cluster_model, "distances"):
                dists = self.cluster_model.distances(X_for_cluster)
            else:
                dists = None

            if dists is not None and dists.shape[1] >= 2:
                sorted_d = np.sort(dists, axis=1)
                # Larger margin -> higher confidence.
                margin = sorted_d[:, 1] - sorted_d[:, 0]
                scale = np.maximum(sorted_d[:, 1], 1e-6)
                confidence = np.clip(margin / scale, 0.0, 1.0).astype(np.float32)
        except Exception:
            pass

        return labels, confidence


# =============================================================================
# Load tracking rows and extract crops
# =============================================================================
def load_tracks() -> Tuple[dict, List[dict]]:
    with open(TRACKS_PATH, "r") as f:
        data = json.load(f)
    return data, data["tracks"]


def group_tracks_by_frame(tracks: List[dict]) -> Dict[int, List[dict]]:
    by_frame = defaultdict(list)
    for row in tracks:
        by_frame[int(row["frame_index"])].append(row)
    return by_frame


def collect_training_crops(video_path: Path, tracks: List[dict]) -> List[CropRecord]:
    """
    Collect player crops for training the team classifier.

    Unlike the notebook, we do not rerun detection. We use existing tracked boxes.
    """
    print("Collecting training crops from tracking output")
    by_frame = group_tracks_by_frame(tracks)
    player_track_lengths = Counter(int(t["track_id"]) for t in tracks if int(t["label"]) in PLAYER_LABELS)

    candidate_frames = sorted(by_frame.keys())[::max(1, TRAIN_STRIDE_FRAMES)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    records: List[CropRecord] = []
    for frame_idx in candidate_frames:
        if len(records) >= MAX_TRAIN_CROPS:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        for row in by_frame.get(frame_idx, []):
            label = int(row["label"])
            tid = int(row["track_id"])
            if label not in PLAYER_LABELS:
                continue
            if player_track_lengths[tid] < MIN_PLAYER_TRACK_LEN_FOR_TRAIN:
                continue

            bbox = [float(v) for v in row["bbox_xyxy"]]
            bw, bh = bbox_wh(bbox)
            if bw < MIN_TRAIN_CROP_W or bh < MIN_TRAIN_CROP_H:
                continue

            crop = torso_crop_from_frame(frame, bbox)
            # crop = crop_from_frame(frame, bbox)
            if crop is None:
                continue

            records.append(CropRecord(
                crop_bgr=crop,
                track_id=tid,
                frame_index=frame_idx,
                label=label,
                bbox_xyxy=bbox,
            ))

            if len(records) >= MAX_TRAIN_CROPS:
                break

    cap.release()
    print(f"Collected {len(records)} training crops")
    return records


def collect_track_prediction_crops(video_path: Path, tracks: List[dict]) -> Tuple[Dict[int, List[CropRecord]], Dict[int, dict]]:
    """
    Quality-preserving optimized version.

    This keeps the original sampling policy exactly:
      - TEAM_ASSIGNMENT_LABELS only (players + goalkeepers)
      - up to MAX_CROPS_PER_TRACK_FOR_PREDICT crops per track
      - torso_crop_from_frame(...)
      - mean_color_lab from the sampled torso crops

    The only real optimization is I/O:
      Original: cap.set(...) separately for every sampled crop / every track.
      New: choose the same sampled rows first, group them by frame, then read the
      video sequentially and extract all crops needed for that frame.
    """
    by_tid_rows = defaultdict(list)
    for row in tracks:
        label = int(row.get("label", -1))
        if label not in TEAM_ASSIGNMENT_LABELS:
            continue
        by_tid_rows[int(row["track_id"])].append(row)

    sampled_by_frame = defaultdict(list)
    track_meta: Dict[int, dict] = {}

    for tid, rows in sorted(by_tid_rows.items()):
        rows = sorted(rows, key=lambda r: int(r["frame_index"]))
        label = int(rows[0]["label"])
        frame_indices = [int(r["frame_index"]) for r in rows]

        # Same policy as the original: MAX_CROPS_PER_TRACK_FOR_PREDICT is 30 by default.
        if len(rows) <= MAX_CROPS_PER_TRACK_FOR_PREDICT:
            sampled = rows
        else:
            idxs = np.linspace(0, len(rows) - 1, MAX_CROPS_PER_TRACK_FOR_PREDICT).round().astype(int)
            sampled = [rows[i] for i in idxs]

        for row in sampled:
            sampled_by_frame[int(row["frame_index"])].append(row)

        all_centers = [bbox_center([float(v) for v in r["bbox_xyxy"]]) for r in rows]
        all_bottoms = [bottom_center([float(v) for v in r["bbox_xyxy"]]) for r in rows]
        all_whs = [bbox_wh([float(v) for v in r["bbox_xyxy"]]) for r in rows]

        track_meta[tid] = {
            "label": label,
            "start_frame": min(frame_indices),
            "end_frame": max(frame_indices),
            "count": len(rows),
            "mean_center_xy": np.mean(np.array(all_centers, dtype=np.float32), axis=0),
            "mean_bottom_center_xy": np.mean(np.array(all_bottoms, dtype=np.float32), axis=0),
            "mean_bbox_wh": np.mean(np.array(all_whs, dtype=np.float32), axis=0),
            "mean_color_lab": np.array([0, 0, 0], dtype=np.float32),
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    track_crops: Dict[int, List[CropRecord]] = defaultdict(list)
    labs_by_tid: Dict[int, List[np.ndarray]] = defaultdict(list)

    sampled_frames = sorted(sampled_by_frame.keys())
    print(
        f"Collecting prediction crops sequentially: "
        f"{len(sampled_frames)} frames, {len(by_tid_rows)} track IDs, "
        f"max {MAX_CROPS_PER_TRACK_FOR_PREDICT} crops/track"
    )

    current_frame_idx = -1
    for k, frame_idx in enumerate(sampled_frames):
        if k % 250 == 0:
            print(f"  prediction crop frame {k + 1}/{len(sampled_frames)}")

        if frame_idx != current_frame_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        current_frame_idx = frame_idx
        if not ret:
            continue

        for row in sampled_by_frame[frame_idx]:
            tid = int(row["track_id"])
            label = int(row["label"])
            bbox = [float(v) for v in row["bbox_xyxy"]]

            crop = torso_crop_from_frame(frame, bbox)
            if crop is None:
                continue

            track_crops[tid].append(CropRecord(
                crop_bgr=crop,
                track_id=tid,
                frame_index=frame_idx,
                label=label,
                bbox_xyxy=bbox,
            ))
            labs_by_tid[tid].append(upper_body_lab_mean(crop))

    cap.release()

    for tid, labs in labs_by_tid.items():
        if labs:
            track_meta[tid]["mean_color_lab"] = np.mean(np.stack(labs, axis=0), axis=0).astype(np.float32)

    print(f"Collected prediction crops for {len(track_crops)} track IDs")
    return track_crops, track_meta


# =============================================================================
# Team assignment logic
# =============================================================================
def majority_vote(labels: np.ndarray, confidences: np.ndarray) -> Tuple[Optional[int], Optional[float], Dict[str, int]]:
    if len(labels) == 0:
        return None, None, {}

    counts = Counter(labels.tolist())
    best_label, best_count = counts.most_common(1)[0]
    vote_conf = best_count / len(labels)

    # Combine voting confidence with average model confidence for that label.
    model_conf = float(np.mean(confidences[labels == best_label])) if len(confidences) else 1.0
    final_conf = 0.7 * vote_conf + 0.3 * model_conf

    votes = {str(k): int(v) for k, v in sorted(counts.items())}
    return int(best_label), float(final_conf), votes


def compute_team_position_centroids(assignments: Dict[int, TrackAssignment]) -> Dict[int, np.ndarray]:
    positions_by_team = defaultdict(list)
    for a in assignments.values():
        if a.label in PLAYER_LABELS and a.team_id is not None and not a.referee_like:
            positions_by_team[int(a.team_id)].append(np.array(a.mean_bottom_center_xy, dtype=np.float32))

    centroids = {}
    for team_id, pts in positions_by_team.items():
        if pts:
            centroids[team_id] = np.stack(pts, axis=0).mean(axis=0)
    return centroids


def resolve_goalkeeper_team_by_position(
    goalkeeper_assignment: TrackAssignment,
    team_position_centroids: Dict[int, np.ndarray],
) -> Tuple[Optional[int], Optional[float]]:
    if len(team_position_centroids) < 2:
        return goalkeeper_assignment.team_id, goalkeeper_assignment.team_confidence

    gk_xy = np.array(goalkeeper_assignment.mean_bottom_center_xy, dtype=np.float32)
    dists = {}
    for tid, centroid in team_position_centroids.items():
        dists[tid] = float(np.linalg.norm(gk_xy - centroid))

    best_team = min(dists, key=dists.get)
    sorted_d = sorted(dists.values())
    if len(sorted_d) >= 2:
        confidence = float(np.clip((sorted_d[1] - sorted_d[0]) / max(sorted_d[1], 1e-6), 0.0, 1.0))
    else:
        confidence = None
    return int(best_team), confidence


def maybe_referee_like(label: int, count: int, confidence: Optional[float], mean_color_lab: np.ndarray) -> bool:
    """
    Conservative heuristic.
    We only flag, not delete.
    """
    if not ENABLE_REFEREE_LIKE_FLAG:
        return False
    if label not in PLAYER_LABELS:
        return False
    if count < REFEREE_MIN_TRACK_LEN:
        return False

    low_conf = confidence is not None and confidence < REFEREE_LOW_CONFIDENCE_THRESHOLD
    dark_crop = float(mean_color_lab[0]) < REFEREE_DARK_LAB_L_THRESHOLD
    return bool(low_conf and dark_crop)


def build_assignments(
    classifier: TeamClassifierV2,
    track_crops: Dict[int, List[CropRecord]],
    track_meta: Dict[int, dict],
) -> Dict[int, TrackAssignment]:
    assignments: Dict[int, TrackAssignment] = {}

    # First assign players from crop embeddings.
    for tid, meta in sorted(track_meta.items()):
        label = int(meta["label"])
        crops = [r.crop_bgr for r in track_crops.get(tid, [])]

        team_id: Optional[int] = None
        confidence: Optional[float] = None
        votes: Dict[str, int] = {}
        method = "unassigned"

        if label in PLAYER_LABELS and len(crops) >= MIN_CROPS_PER_TRACK_FOR_CONFIDENT_TEAM:
            pred, conf = classifier.predict(crops)
            team_id, confidence, votes = majority_vote(pred, conf)
            method = "embedding_vote"
        elif label in PLAYER_LABELS and len(crops) > 0:
            pred, conf = classifier.predict(crops)
            team_id, confidence, votes = majority_vote(pred, conf)
            method = "embedding_vote_low_crop_count"

        mean_color_lab = meta["mean_color_lab"].astype(np.float32)
        referee_like = maybe_referee_like(label, int(meta["count"]), confidence, mean_color_lab)

        assignments[tid] = TrackAssignment(
            track_id=tid,
            label=label,
            start_frame=int(meta["start_frame"]),
            end_frame=int(meta["end_frame"]),
            count=int(meta["count"]),
            team_id=team_id,
            team_confidence=confidence,
            referee_like=referee_like,
            mean_color_lab=[round(float(x), 3) for x in mean_color_lab.tolist()],
            mean_center_xy=[round(float(x), 3) for x in meta["mean_center_xy"].tolist()],
            mean_bottom_center_xy=[round(float(x), 3) for x in meta["mean_bottom_center_xy"].tolist()],
            mean_bbox_wh=[round(float(x), 3) for x in meta["mean_bbox_wh"].tolist()],
            votes=votes,
            method=method,
        )

    # Then resolve goalkeepers by position, like the notebook.
    if ASSIGN_GOALKEEPERS_BY_POSITION:
        team_centroids = compute_team_position_centroids(assignments)
        for a in assignments.values():
            if a.label in GOALKEEPER_LABELS:
                team_id, conf = resolve_goalkeeper_team_by_position(a, team_centroids)
                a.team_id = team_id
                a.team_confidence = conf
                a.method = "goalkeeper_position_centroid"
                a.referee_like = False

    return assignments


def build_ball_pass_through_assignments(all_tracks: List[dict]) -> Dict[int, TrackAssignment]:
    """
    Keep every tracked label=0 ball row visible in team_assignment_v2.json.
    Team assignment does not assign a team to the ball.
    """
    by_tid: Dict[int, List[dict]] = defaultdict(list)
    for row in all_tracks:
        if int(row.get("label", -1)) in BALL_LABELS:
            by_tid[int(row["track_id"])].append(row)

    assignments: Dict[int, TrackAssignment] = {}
    for tid, rows in sorted(by_tid.items()):
        frames = [int(r["frame_index"]) for r in rows]
        bboxes = [[float(v) for v in r["bbox_xyxy"]] for r in rows]

        widths, heights, cxs, cys, bcys = [], [], [], [], []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            widths.append(max(1.0, x2 - x1))
            heights.append(max(1.0, y2 - y1))
            cxs.append((x1 + x2) / 2.0)
            cys.append((y1 + y2) / 2.0)
            bcys.append(y2)

        assignments[tid] = TrackAssignment(
            track_id=tid,
            label=0,
            start_frame=min(frames),
            end_frame=max(frames),
            count=len(rows),
            team_id=None,
            team_confidence=None,
            referee_like=False,
            mean_color_lab=[],
            mean_center_xy=[round(float(np.mean(cxs)), 3), round(float(np.mean(cys)), 3)],
            mean_bottom_center_xy=[round(float(np.mean(cxs)), 3), round(float(np.mean(bcys)), 3)],
            mean_bbox_wh=[round(float(np.mean(widths)), 3), round(float(np.mean(heights)), 3)],
            votes={},
            method="ball_pass_through",
        )

    return assignments


# =============================================================================
# Saving outputs
# =============================================================================
def save_assignments(
    assignments: Dict[int, TrackAssignment],
    classifier_debug: Dict[str, Any],
    out_path: Path,
    ball_detections: Optional[List[dict]] = None,
) -> None:
    _backup_if_exists(out_path)

    rows = []
    for tid, a in sorted(assignments.items()):
        rows.append({
            "track_id": int(tid),
            "label": int(a.label),
            "start_frame": int(a.start_frame),
            "end_frame": int(a.end_frame),
            "count": int(a.count),
            "team_id": int(a.team_id) if a.team_id is not None else None,
            "team_confidence": round(float(a.team_confidence), 4) if a.team_confidence is not None else None,
            "referee_like": bool(a.referee_like),
            "mean_color_lab": a.mean_color_lab,
            "mean_center_xy": a.mean_center_xy,
            "mean_bottom_center_xy": a.mean_bottom_center_xy,
            "mean_bbox_wh": a.mean_bbox_wh,
            "votes": a.votes,
            "method": a.method,
        })

    payload = {
        "video": str(VIDEO_PATH),
        "tracks_json": str(TRACKS_PATH),
        "classifier_debug": classifier_debug,
        "ball_detections": ball_detections or [],
        "tracks": rows,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved team assignment V2 JSON: {out_path}")


def save_debug(training_records: List[CropRecord], train_labels: np.ndarray, classifier: TeamClassifierV2, out_path: Path) -> None:
    _backup_if_exists(out_path)
    payload = {
        "num_training_crops": len(training_records),
        "classifier_debug": classifier.debug_info,
        "training_cluster_counts": {str(k): int(v) for k, v in sorted(Counter(train_labels.tolist()).items())},
        "sample_training_records": [
            {
                "track_id": int(r.track_id),
                "frame_index": int(r.frame_index),
                "label": int(r.label),
                "cluster": int(train_labels[i]),
                "bbox_xyxy": [round(float(x), 2) for x in r.bbox_xyxy],
            }
            for i, r in enumerate(training_records[:200])
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved debug JSON: {out_path}")


# =============================================================================
# Overlay video
# =============================================================================
def save_overlay_video(video_path: Path, all_tracks: List[dict], assignments: Dict[int, TrackAssignment], out_path: Path) -> None:
    _backup_if_exists(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    by_frame = group_tracks_by_frame(all_tracks)

    def style_for_track(tid: int) -> Tuple[Tuple[int, int, int], str]:
        a = assignments.get(tid)
        if a is None:
            return UNKNOWN_COLOR, "UNK"
        if a.label in BALL_LABELS:
            return (0, 255, 0), "BALL"
        if a.referee_like:
            return REFEREE_COLOR, "REF?"
        if a.team_id in TEAM_COLORS:
            if a.label in GOALKEEPER_LABELS:
                return TEAM_COLORS[a.team_id], f"GK-T{a.team_id}"
            return TEAM_COLORS[a.team_id], f"T{a.team_id}"
        return UNKNOWN_COLOR, "UNK"

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        for row in by_frame.get(frame_idx, []):
            tid = int(row["track_id"])
            label = int(row["label"])
            if label not in TEAM_ASSIGNMENT_LABELS and label not in PASS_THROUGH_LABELS:
                continue

            bbox = [float(v) for v in row["bbox_xyxy"]]
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            color, team_txt = style_for_track(tid)
            text = f"TID:{tid} {team_txt}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            y_top = max(0, y1 - th - 6)
            cv2.rectangle(frame, (x1, y_top), (min(w - 1, x1 + tw + 4), y1), color, -1)
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cv2.putText(frame, text, (x1 + 2, y1 - 4), font, font_scale, text_color, thickness)

        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved overlay video: {out_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("=" * 72)
    print("TEAM ASSIGNMENT V2 STAGE")
    print("=" * 72)
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", type=str, default=None, help="Optional override video path")
    args, _ = parser.parse_known_args()

    # Allow overriding the global VIDEO_PATH from CLI while keeping default otherwise.
    if args.video:
        global VIDEO_PATH
        VIDEO_PATH = Path(args.video)

    _ensure_paths_exist()

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"Using device: {device}")

    raw_meta, all_tracks = load_tracks()

    training_records = collect_training_crops(VIDEO_PATH, all_tracks)
    if len(training_records) < 10:
        raise RuntimeError("Not enough training crops for team assignment V2")

    classifier = TeamClassifierV2(device=device)
    train_crops = [r.crop_bgr for r in training_records]
    train_labels = classifier.fit(train_crops)

    track_crops, track_meta = collect_track_prediction_crops(VIDEO_PATH, all_tracks)
    print("Prediction crops collected")
    assignments = build_assignments(classifier, track_crops, track_meta)
    ball_detections = raw_meta.get("ball_detections", [])

    assignment_path = OUT_DIR / "team_assignment_v2.json"
    debug_path = OUT_DIR / "team_assignment_v2_debug.json"
    save_assignments(assignments, classifier.debug_info, assignment_path, ball_detections)
    save_debug(training_records, train_labels, classifier, debug_path)

    if SAVE_OVERLAY_VIDEO:
        overlay_path = OUT_DIR / OVERLAY_VIDEO_NAME
        save_overlay_video(VIDEO_PATH, all_tracks, assignments, overlay_path)

    print("=" * 72)
    print("TEAM ASSIGNMENT V2 STAGE COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()
