"""
evaluate_pipeline.py

One-file evaluator for the football analytics pipeline:

    Object Detection -> Tracking -> Team Assignment -> ReID

This script is designed to work with your existing project outputs without
modifying the pipeline itself.

It supports two evaluation modes:

1. No Ground Truth mode
   --------------------
   Produces useful diagnostic metrics from the pipeline outputs only:
   - number of detections
   - detection confidence statistics
   - number of tracks
   - track fragmentation indicators
   - short/noisy tracks
   - team assignment confidence and vote consistency
   - ReID merge summary
   - blocked merge reasons

2. Ground Truth mode
   -----------------
   If you manually create GT JSON files, the script computes stronger metrics:
   - Detection Precision / Recall / F1 using IoU matching
   - Tracking ID switch / fragmentation approximations using GT identities
   - Team assignment accuracy
   - ReID pairwise merge precision/recall/F1

Recommended project location:
-----------------------------
    /home/shaikar/sn_pipe_trial/pipeline/evaluate_pipeline.py

Run:
----
    python pipeline/evaluate_pipeline.py

Optional custom paths:
----------------------
    python pipeline/evaluate_pipeline.py \
        --detections outputs/detections/predictions.json \
        --tracks outputs/tracks/tracks.json \
        --team outputs/team_assignment_v2/team_assignment_v2.json \
        --reid-report outputs/reid_v2/reid_report.json \
        --reid-map outputs/reid_v2/trackid_to_globalid.json

Optional GT files:
------------------
    python pipeline/evaluate_pipeline.py \
        --gt-detections data/gt/gt_detections.json \
        --gt-tracks data/gt/gt_tracks.json \
        --gt-teams data/gt/gt_teams.json \
        --gt-reid data/gt/gt_reid.json

Expected GT schemas are documented in the comments below.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np


# =============================================================================
# Default paths
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DETECTIONS_PATH = ROOT / "outputs" / "detections" / "predictions.json"
DEFAULT_TRACKS_PATH = ROOT / "outputs" / "tracks" / "tracks.json"
DEFAULT_TEAM_PATH = ROOT / "outputs" / "team_assignment_v2" / "team_assignment_v2.json"
DEFAULT_REID_REPORT_PATH = ROOT / "outputs" / "reid_v2" / "reid_report.json"
DEFAULT_REID_MAP_PATH = ROOT / "outputs" / "reid_v2" / "trackid_to_globalid.json"
DEFAULT_OUT_PATH = ROOT / "outputs" / "evaluation" / "evaluation_report.json"


# =============================================================================
# Configuration
# =============================================================================
IOU_THRESHOLD = 0.5
PLAYER_LABELS = {1}
GOALKEEPER_LABELS = {3}
VALID_PERSON_LABELS = PLAYER_LABELS | GOALKEEPER_LABELS

# Tracks shorter than this are treated as noisy/fragmented indicators.
SHORT_TRACK_MAX_LEN = 5

# Low team confidence threshold. Tune after inspecting several videos.
LOW_TEAM_CONFIDENCE = 0.65


# =============================================================================
# IO / utility helpers
# =============================================================================
def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved evaluation report: {path}")


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def median_or_none(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def pct_or_none(values: List[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def greedy_match_by_iou(
    pred_boxes: List[dict],
    gt_boxes: List[dict],
    iou_thr: float = IOU_THRESHOLD,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy one-to-one matching by highest IoU.

    pred_boxes and gt_boxes should contain:
        {"bbox_xyxy": [x1,y1,x2,y2], "label": int, ...}
    """
    candidates = []
    for pi, p in enumerate(pred_boxes):
        for gi, g in enumerate(gt_boxes):
            if int(p.get("label", -1)) != int(g.get("label", -2)):
                continue
            iou = bbox_iou(p["bbox_xyxy"], g["bbox_xyxy"])
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))

    candidates.sort(reverse=True)
    used_p = set()
    used_g = set()
    matches = []

    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, float(iou)))

    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_p]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_g]
    return matches, unmatched_pred, unmatched_gt


# =============================================================================
# Parsers for your project outputs
# =============================================================================
def parse_detection_frames(detections_json: Optional[dict]) -> Dict[int, List[dict]]:
    """
    Tries to parse your detection predictions.json flexibly.

    Expected common structure:
        {
          "frames": [
             {"frame_index": 0, "detections": [...]},
             ...
          ]
        }

    Each detection should have bbox_xyxy / bbox and label/class_id and score/confidence.
    """
    if not detections_json:
        return {}

    frames_obj = detections_json.get("frames", [])
    out: Dict[int, List[dict]] = defaultdict(list)

    if isinstance(frames_obj, list):
        for frame_row in frames_obj:
            frame_idx = int(frame_row.get("frame_index", frame_row.get("frame", 0)))
            dets = frame_row.get("detections", frame_row.get("objects", []))
            for d in dets:
                bbox = d.get("bbox_xyxy", d.get("xyxy", d.get("bbox")))
                if bbox is None:
                    continue
                label = d.get("label", d.get("class_id", d.get("class", -1)))
                score = d.get("score", d.get("confidence", d.get("conf", None)))
                out[frame_idx].append({
                    "frame_index": frame_idx,
                    "bbox_xyxy": [float(x) for x in bbox],
                    "label": int(label),
                    "score": safe_float(score),
                })

    return dict(out)


def parse_tracks(tracks_json: Optional[dict]) -> List[dict]:
    if not tracks_json:
        return []
    rows = tracks_json.get("tracks", [])
    parsed = []
    for r in rows:
        if "bbox_xyxy" not in r:
            continue
        parsed.append({
            "frame_index": int(r["frame_index"]),
            "track_id": int(r["track_id"]),
            "bbox_xyxy": [float(x) for x in r["bbox_xyxy"]],
            "label": int(r.get("label", -1)),
            "score": safe_float(r.get("score", None)),
        })
    return parsed


def parse_team_assignments(team_json: Optional[dict]) -> Dict[int, dict]:
    if not team_json:
        return {}
    out = {}
    for r in team_json.get("tracks", []):
        tid = int(r["track_id"])
        out[tid] = r
    return out


def parse_reid_map(reid_map_json: Optional[dict]) -> Dict[int, int]:
    if not reid_map_json:
        return {}
    raw = reid_map_json.get("trackid_to_globalid", {})
    return {int(k): int(v) for k, v in raw.items()}


# =============================================================================
# Ground Truth parsers / schemas
# =============================================================================
# 1. GT detections schema:
# {
#   "frames": [
#     {
#       "frame_index": 0,
#       "objects": [
#         {"gt_id": "p1", "label": 1, "bbox_xyxy": [x1,y1,x2,y2]},
#         ...
#       ]
#     }
#   ]
# }
#
# 2. GT tracks schema:
# Same as GT detections, but gt_id should be consistent across frames.
#
# 3. GT teams schema:
# {
#   "track_to_team": {"6": 1, "26": 1, "14": 0, ...}
# }
# or:
# {
#   "gt_identity_to_team": {"p1": 0, "p2": 1, ...}
# }
#
# 4. GT ReID schema:
# {
#   "same_identity_pairs": [[6,26], [14,29], [22,23], [22,24]],
#   "different_identity_pairs": [[10,38], [10,29]]
# }


def parse_gt_frames(gt_json: Optional[dict]) -> Dict[int, List[dict]]:
    if not gt_json:
        return {}
    out = defaultdict(list)
    for frame_row in gt_json.get("frames", []):
        frame_idx = int(frame_row.get("frame_index", frame_row.get("frame", 0)))
        objects = frame_row.get("objects", frame_row.get("detections", []))
        for obj in objects:
            bbox = obj.get("bbox_xyxy", obj.get("bbox"))
            if bbox is None:
                continue
            out[frame_idx].append({
                "frame_index": frame_idx,
                "gt_id": str(obj.get("gt_id", obj.get("identity", ""))),
                "bbox_xyxy": [float(x) for x in bbox],
                "label": int(obj.get("label", obj.get("class_id", -1))),
            })
    return dict(out)


def parse_gt_reid(gt_reid_json: Optional[dict]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    if not gt_reid_json:
        return set(), set()

    def norm_pair(p: List[int]) -> Tuple[int, int]:
        a, b = int(p[0]), int(p[1])
        return (min(a, b), max(a, b))

    same = {norm_pair(p) for p in gt_reid_json.get("same_identity_pairs", [])}
    diff = {norm_pair(p) for p in gt_reid_json.get("different_identity_pairs", [])}
    return same, diff


# =============================================================================
# 1. Object Detection Evaluation
# =============================================================================
def evaluate_object_detection(
    detections_by_frame: Dict[int, List[dict]],
    gt_by_frame: Optional[Dict[int, List[dict]]] = None,
) -> Dict[str, Any]:
    """
    Evaluates object detection.

    Without GT:
    - counts detections
    - confidence distribution
    - label distribution

    With GT:
    - IoU matching
    - precision / recall / F1 overall and by label
    - mean matched IoU
    """
    all_dets = [d for dets in detections_by_frame.values() for d in dets]
    scores = [d["score"] for d in all_dets if d.get("score") is not None]
    labels = [int(d.get("label", -1)) for d in all_dets]

    report: Dict[str, Any] = {
        "mode": "no_ground_truth" if not gt_by_frame else "with_ground_truth",
        "num_frames_with_detections": len(detections_by_frame),
        "num_detections": len(all_dets),
        "label_distribution": {str(k): int(v) for k, v in sorted(Counter(labels).items())},
        "score_stats": {
            "mean": mean_or_none(scores),
            "median": median_or_none(scores),
            "p10": pct_or_none(scores, 10),
            "p90": pct_or_none(scores, 90),
        },
    }

    if not gt_by_frame:
        report["notes"] = [
            "No GT detections provided, so this is a diagnostic summary only.",
            "For true detection evaluation, provide GT boxes and compute IoU-based Precision/Recall/F1.",
        ]
        return report

    tp = fp = fn = 0
    matched_ious = []
    by_label_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ious": []})

    # Evaluate only frames that were manually annotated in GT.
    # Otherwise, detections in unlabeled frames are wrongly counted as false positives.
    all_frames = sorted(gt_by_frame.keys())
    
    for frame_idx in all_frames:
        preds = detections_by_frame.get(frame_idx, [])
        gts = gt_by_frame.get(frame_idx, [])
        matches, unmatched_pred, unmatched_gt = greedy_match_by_iou(preds, gts, IOU_THRESHOLD)

        tp += len(matches)
        fp += len(unmatched_pred)
        fn += len(unmatched_gt)

        for pi, gi, iou in matches:
            label = int(preds[pi]["label"])
            by_label_counts[label]["tp"] += 1
            by_label_counts[label]["ious"].append(float(iou))
            matched_ious.append(float(iou))
        for pi in unmatched_pred:
            label = int(preds[pi]["label"])
            by_label_counts[label]["fp"] += 1
        for gi in unmatched_gt:
            label = int(gts[gi]["label"])
            by_label_counts[label]["fn"] += 1

    report["overall"] = precision_recall_f1(tp, fp, fn)
    report["mean_matched_iou"] = mean_or_none(matched_ious)
    report["by_label"] = {}
    for label, c in sorted(by_label_counts.items()):
        row = precision_recall_f1(c["tp"], c["fp"], c["fn"])
        row["mean_iou"] = mean_or_none(c["ious"])
        report["by_label"][str(label)] = row

    return report


# =============================================================================
# 2. Tracking Evaluation
# =============================================================================
def evaluate_tracking(
    tracks: List[dict],
    gt_tracks_by_frame: Optional[Dict[int, List[dict]]] = None,
) -> Dict[str, Any]:
    """
    Evaluates tracking quality.

    Without GT:
    - number of unique tracks
    - average / median track length
    - short tracks count
    - track length distribution

    With GT:
    - matches each predicted track observation to GT object by IoU
    - estimates ID switches and fragmentation per GT identity
    """
    by_tid = defaultdict(list)
    by_frame = defaultdict(list)
    for r in tracks:
        by_tid[int(r["track_id"])].append(r)
        by_frame[int(r["frame_index"])].append(r)

    lengths = {tid: len(rows) for tid, rows in by_tid.items()}
    labels_by_tid = {tid: Counter(int(r["label"]) for r in rows).most_common(1)[0][0] for tid, rows in by_tid.items()}
    short_tracks = [tid for tid, n in lengths.items() if n <= SHORT_TRACK_MAX_LEN]

    report: Dict[str, Any] = {
        "mode": "no_ground_truth" if not gt_tracks_by_frame else "with_ground_truth",
        "num_track_observations": len(tracks),
        "num_unique_track_ids": len(by_tid),
        "track_length_stats": {
            "mean": mean_or_none(list(lengths.values())),
            "median": median_or_none(list(lengths.values())),
            "min": int(min(lengths.values())) if lengths else 0,
            "max": int(max(lengths.values())) if lengths else 0,
        },
        "num_short_tracks": len(short_tracks),
        "short_track_ids": sorted(short_tracks),
        "track_label_distribution": {
            str(k): int(v) for k, v in sorted(Counter(labels_by_tid.values()).items())
        },
    }

    if not gt_tracks_by_frame:
        report["notes"] = [
            "No GT tracks provided, so this is a diagnostic tracking summary only.",
            "High number of short tracks indicates fragmentation or noisy detections.",
            "For true tracking evaluation, provide GT identities per frame.",
        ]
        return report

    # Approximate GT identity -> sequence of matched predicted track IDs.
    gt_to_pred_seq = defaultdict(list)
    pred_to_gt_seq = defaultdict(list)
    matched_ious = []

    all_frames = sorted(set(by_frame.keys()) | set(gt_tracks_by_frame.keys()))
    for frame_idx in all_frames:
        preds = by_frame.get(frame_idx, [])
        gts = gt_tracks_by_frame.get(frame_idx, [])
        matches, _, _ = greedy_match_by_iou(preds, gts, IOU_THRESHOLD)
        for pi, gi, iou in matches:
            pred_tid = int(preds[pi]["track_id"])
            gt_id = str(gts[gi]["gt_id"])
            gt_to_pred_seq[gt_id].append((frame_idx, pred_tid))
            pred_to_gt_seq[pred_tid].append((frame_idx, gt_id))
            matched_ious.append(float(iou))

    # ID switches: for each GT identity, count changes in matched predicted track_id over time.
    id_switches = 0
    fragments = 0
    gt_identity_rows = {}
    for gt_id, seq in gt_to_pred_seq.items():
        seq = sorted(seq)
        pred_ids = [tid for _, tid in seq]
        changes = sum(1 for i in range(1, len(pred_ids)) if pred_ids[i] != pred_ids[i - 1])
        unique_pred_ids = len(set(pred_ids))
        id_switches += changes
        fragments += max(0, unique_pred_ids - 1)
        gt_identity_rows[gt_id] = {
            "matched_frames": len(seq),
            "unique_pred_track_ids": sorted(set(pred_ids)),
            "num_unique_pred_track_ids": unique_pred_ids,
            "id_switches": changes,
        }

    report["gt_identity_tracking"] = gt_identity_rows
    report["approx_id_switches"] = int(id_switches)
    report["approx_fragments"] = int(fragments)
    report["mean_matched_iou"] = mean_or_none(matched_ious)

    return report


# =============================================================================
# 3. Team Assignment Evaluation
# =============================================================================
def evaluate_team_assignment(
    team_assignments: Dict[int, dict],
    gt_team_json: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Evaluates team assignment.

    Important:
    ----------
    KMeans cluster labels are arbitrary.
    Therefore, team_id=0 and team_id=1 may be flipped between runs.

    With GT, we calculate both:
    - normal accuracy: pred == gt
    - flipped accuracy: (1 - pred) == gt

    Final team_accuracy is max(normal_accuracy, flipped_accuracy).
    """
    rows = list(team_assignments.values())
    team_ids = [r.get("team_id") for r in rows if r.get("team_id") is not None]
    confidences = [float(r["team_confidence"]) for r in rows if r.get("team_confidence") is not None]

    low_conf = []
    for r in rows:
        conf = r.get("team_confidence")
        if conf is not None and float(conf) < LOW_TEAM_CONFIDENCE:
            low_conf.append(int(r["track_id"]))

    referee_like = [int(r["track_id"]) for r in rows if bool(r.get("referee_like", False))]

    # Vote consistency = majority votes / total votes.
    vote_consistency = {}
    for r in rows:
        votes = r.get("votes", {}) or {}
        total_votes = sum(int(v) for v in votes.values())
        if total_votes > 0:
            best = max(int(v) for v in votes.values())
            vote_consistency[int(r["track_id"])] = best / total_votes

    report: Dict[str, Any] = {
        "mode": "no_ground_truth" if not gt_team_json else "with_ground_truth",
        "num_assigned_tracks": len(rows),
        "team_distribution": {str(k): int(v) for k, v in sorted(Counter(team_ids).items())},
        "confidence_stats": {
            "mean": mean_or_none(confidences),
            "median": median_or_none(confidences),
            "p10": pct_or_none(confidences, 10),
            "p90": pct_or_none(confidences, 90),
        },
        "low_confidence_threshold": LOW_TEAM_CONFIDENCE,
        "low_confidence_track_ids": sorted(low_conf),
        "referee_like_track_ids": sorted(referee_like),
        "vote_consistency_stats": {
            "mean": mean_or_none(list(vote_consistency.values())),
            "median": median_or_none(list(vote_consistency.values())),
        },
    }

    if not gt_team_json:
        report["notes"] = [
            "No GT team labels provided, so this is a diagnostic team-assignment summary only.",
            "Low confidence or low vote consistency tracks should be manually inspected.",
        ]
        return report

    gt_track_to_team = gt_team_json.get("track_to_team", {})
    gt_track_to_team = {int(k): int(v) for k, v in gt_track_to_team.items()}

    normal_correct = 0
    flipped_correct = 0
    total = 0

    normal_mistakes = []
    flipped_mistakes = []

    for tid, gt_team in gt_track_to_team.items():
        pred = team_assignments.get(tid)
        if pred is None:
            continue

        pred_team = pred.get("team_id")
        if pred_team is None:
            continue

        pred_team = int(pred_team)
        gt_team = int(gt_team)

        # Supports binary team assignment only: 0/1
        if pred_team not in [0, 1] or gt_team not in [0, 1]:
            continue

        total += 1

        # Normal mapping
        if pred_team == gt_team:
            normal_correct += 1
        else:
            normal_mistakes.append({
                "track_id": tid,
                "gt_team": gt_team,
                "pred_team": pred_team,
                "confidence": pred.get("team_confidence"),
                "votes": pred.get("votes", {}),
            })

        # Flipped mapping: 0 <-> 1
        flipped_pred_team = 1 - pred_team

        if flipped_pred_team == gt_team:
            flipped_correct += 1
        else:
            flipped_mistakes.append({
                "track_id": tid,
                "gt_team": gt_team,
                "pred_team": flipped_pred_team,
                "original_pred_team": pred_team,
                "confidence": pred.get("team_confidence"),
                "votes": pred.get("votes", {}),
            })

    normal_accuracy = normal_correct / total if total else 0.0
    flipped_accuracy = flipped_correct / total if total else 0.0

    use_flipped_mapping = flipped_accuracy > normal_accuracy

    if use_flipped_mapping:
        final_accuracy = flipped_accuracy
        final_correct = flipped_correct
        final_mistakes = flipped_mistakes
        mapping_used = "flipped_0_1"
    else:
        final_accuracy = normal_accuracy
        final_correct = normal_correct
        final_mistakes = normal_mistakes
        mapping_used = "normal"

    report["team_accuracy"] = round(final_accuracy, 4) if total else None
    report["normal_accuracy"] = round(normal_accuracy, 4) if total else None
    report["flipped_accuracy"] = round(flipped_accuracy, 4) if total else None
    report["team_label_mapping_used"] = mapping_used
    report["num_evaluated_tracks"] = total
    report["num_correct"] = final_correct
    report["num_wrong"] = len(final_mistakes)
    report["mistakes"] = final_mistakes

    return report


# =============================================================================
# 4. ReID Evaluation
# =============================================================================
def pairs_from_reid_map(trackid_to_globalid: Dict[int, int]) -> Set[Tuple[int, int]]:
    tids = sorted(trackid_to_globalid.keys())
    pairs = set()
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tids[i], tids[j]
            if trackid_to_globalid[a] == trackid_to_globalid[b]:
                pairs.add((a, b))
    return pairs


def evaluate_reid(
    reid_report: Optional[dict],
    trackid_to_globalid: Dict[int, int],
    gt_reid_json: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Evaluates ReID.

    Without GT:
    - number of local/global IDs
    - number of merges
    - groups
    - blocked reasons summary

    With GT same/different pairs:
    - pairwise merge precision/recall/F1
    - false merge pairs
    - missed merge pairs
    """
    groups = defaultdict(list)
    for tid, gid in trackid_to_globalid.items():
        groups[int(gid)].append(int(tid))

    merged_groups = {gid: sorted(tids) for gid, tids in groups.items() if len(tids) > 1}
    pred_same_pairs = pairs_from_reid_map(trackid_to_globalid)

    report: Dict[str, Any] = {
        "mode": "no_ground_truth" if not gt_reid_json else "with_ground_truth",
        "num_local_track_ids": len(trackid_to_globalid),
        "num_global_ids": len(set(trackid_to_globalid.values())),
        "num_merges": len(trackid_to_globalid) - len(set(trackid_to_globalid.values())),
        "merged_groups": {str(k): v for k, v in sorted(merged_groups.items())},
        "num_predicted_same_pairs": len(pred_same_pairs),
    }

    if reid_report:
        candidates = reid_report.get("merge_candidates", [])
        blocked = Counter(c.get("blocked_reason") for c in candidates if c.get("blocked_reason") is not None)
        merged_candidates = [c for c in candidates if bool(c.get("merged"))]
        report["blocked_reason_counts"] = {str(k): int(v) for k, v in blocked.most_common()}
        report["num_merge_candidates_in_report"] = len(candidates)
        report["num_merged_candidates_in_report"] = len(merged_candidates)
        report["top_merged_candidates"] = merged_candidates[:20]

        if "reid_summary" in reid_report:
            report["reported_summary"] = reid_report["reid_summary"]

    if not gt_reid_json:
        report["notes"] = [
            "No GT ReID pairs provided, so this is a diagnostic ReID summary only.",
            "For true ReID evaluation, provide same_identity_pairs and different_identity_pairs.",
        ]
        return report

    gt_same, gt_diff = parse_gt_reid(gt_reid_json)

    tp_pairs = pred_same_pairs & gt_same
    fp_pairs = pred_same_pairs & gt_diff
    # If a predicted pair is not in gt_same or gt_diff, we do not count it as FP,
    # because it may simply be unlabeled.
    fn_pairs = gt_same - pred_same_pairs

    metrics = precision_recall_f1(len(tp_pairs), len(fp_pairs), len(fn_pairs))
    report["pairwise_metrics"] = metrics
    report["true_positive_pairs"] = sorted([list(p) for p in tp_pairs])
    report["false_merge_pairs"] = sorted([list(p) for p in fp_pairs])
    report["missed_merge_pairs"] = sorted([list(p) for p in fn_pairs])
    report["num_gt_same_pairs"] = len(gt_same)
    report["num_gt_different_pairs"] = len(gt_diff)

    return report


# =============================================================================
# Full pipeline consistency checks
# =============================================================================
def evaluate_pipeline_consistency(
    tracks: List[dict],
    team_assignments: Dict[int, dict],
    trackid_to_globalid: Dict[int, int],
) -> Dict[str, Any]:
    """
    Cross-module checks that do not require GT.

    These are very useful for presentation/evaluation planning:
    - global IDs that contain tracks from different teams
    - global IDs with temporal conflicts
    - merged groups with low-confidence team assignments
    """
    by_tid_rows = defaultdict(list)
    for r in tracks:
        by_tid_rows[int(r["track_id"])].append(r)

    tid_frame_ranges = {}
    for tid, rows in by_tid_rows.items():
        frames = [int(r["frame_index"]) for r in rows]
        tid_frame_ranges[tid] = (min(frames), max(frames))

    gid_to_tids = defaultdict(list)
    for tid, gid in trackid_to_globalid.items():
        gid_to_tids[int(gid)].append(int(tid))

    mixed_team_groups = []
    temporal_conflict_groups = []
    low_conf_merged_groups = []

    for gid, tids in sorted(gid_to_tids.items()):
        if len(tids) <= 1:
            continue

        teams = []
        low_conf_tracks = []
        for tid in tids:
            ta = team_assignments.get(tid, {})
            if ta.get("team_id") is not None:
                teams.append(int(ta["team_id"]))
            conf = ta.get("team_confidence")
            if conf is not None and float(conf) < LOW_TEAM_CONFIDENCE:
                low_conf_tracks.append(tid)

        if len(set(teams)) > 1:
            mixed_team_groups.append({
                "global_id": gid,
                "track_ids": sorted(tids),
                "team_ids": sorted(set(teams)),
            })

        # Temporal conflict = two tracks in the same global ID overlap in time.
        conflicts = []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                if a not in tid_frame_ranges or b not in tid_frame_ranges:
                    continue
                a0, a1 = tid_frame_ranges[a]
                b0, b1 = tid_frame_ranges[b]
                overlap = not (a1 < b0 or b1 < a0)
                if overlap:
                    conflicts.append([a, b])
        if conflicts:
            temporal_conflict_groups.append({
                "global_id": gid,
                "track_ids": sorted(tids),
                "conflicting_pairs": conflicts,
            })

        if low_conf_tracks:
            low_conf_merged_groups.append({
                "global_id": gid,
                "track_ids": sorted(tids),
                "low_confidence_track_ids": sorted(low_conf_tracks),
            })

    return {
        "mixed_team_global_id_groups": mixed_team_groups,
        "temporal_conflict_global_id_groups": temporal_conflict_groups,
        "low_confidence_merged_groups": low_conf_merged_groups,
        "num_mixed_team_groups": len(mixed_team_groups),
        "num_temporal_conflict_groups": len(temporal_conflict_groups),
        "num_low_confidence_merged_groups": len(low_conf_merged_groups),
    }


# =============================================================================
# Main orchestration
# =============================================================================
def build_evaluation_report(args: argparse.Namespace) -> Dict[str, Any]:
    detections_json = load_json(args.detections)
    tracks_json = load_json(args.tracks)
    team_json = load_json(args.team)
    reid_report_json = load_json(args.reid_report)
    reid_map_json = load_json(args.reid_map)

    gt_detections_json = load_json(args.gt_detections)
    gt_tracks_json = load_json(args.gt_tracks)
    gt_teams_json = load_json(args.gt_teams)
    gt_reid_json = load_json(args.gt_reid)

    detections_by_frame = parse_detection_frames(detections_json)
    tracks = parse_tracks(tracks_json)
    team_assignments = parse_team_assignments(team_json)
    reid_map = parse_reid_map(reid_map_json)

    gt_detections_by_frame = parse_gt_frames(gt_detections_json)
    gt_tracks_by_frame = parse_gt_frames(gt_tracks_json)

    report = {
        "paths": {
            "detections": str(args.detections),
            "tracks": str(args.tracks),
            "team": str(args.team),
            "reid_report": str(args.reid_report),
            "reid_map": str(args.reid_map),
            "gt_detections": str(args.gt_detections) if args.gt_detections else None,
            "gt_tracks": str(args.gt_tracks) if args.gt_tracks else None,
            "gt_teams": str(args.gt_teams) if args.gt_teams else None,
            "gt_reid": str(args.gt_reid) if args.gt_reid else None,
        },
        "object_detection": evaluate_object_detection(
            detections_by_frame=detections_by_frame,
            gt_by_frame=gt_detections_by_frame if gt_detections_by_frame else None,
        ),
        "tracking": evaluate_tracking(
            tracks=tracks,
            gt_tracks_by_frame=gt_tracks_by_frame if gt_tracks_by_frame else None,
        ),
        "team_assignment": evaluate_team_assignment(
            team_assignments=team_assignments,
            gt_team_json=gt_teams_json,
        ),
        "reid": evaluate_reid(
            reid_report=reid_report_json,
            trackid_to_globalid=reid_map,
            gt_reid_json=gt_reid_json,
        ),
        "pipeline_consistency": evaluate_pipeline_consistency(
            tracks=tracks,
            team_assignments=team_assignments,
            trackid_to_globalid=reid_map,
        ),
    }

    return report


def print_short_summary(report: Dict[str, Any]) -> None:
    print("=" * 80)
    print("PIPELINE EVALUATION SUMMARY")
    print("=" * 80)

    od = report["object_detection"]
    print("\n[Object Detection]")
    print(f"  Mode: {od['mode']}")
    print(f"  Frames with detections: {od.get('num_frames_with_detections')}")
    print(f"  Num detections: {od.get('num_detections')}")
    if "overall" in od:
        print(f"  Precision/Recall/F1: {od['overall']['precision']} / {od['overall']['recall']} / {od['overall']['f1']}")
        print(f"  Mean IoU: {od.get('mean_matched_iou')}")

    tr = report["tracking"]
    print("\n[Tracking]")
    print(f"  Mode: {tr['mode']}")
    print(f"  Unique track IDs: {tr.get('num_unique_track_ids')}")
    print(f"  Short tracks: {tr.get('num_short_tracks')}")
    if "approx_id_switches" in tr:
        print(f"  Approx ID switches: {tr['approx_id_switches']}")
        print(f"  Approx fragments: {tr['approx_fragments']}")

    ta = report["team_assignment"]
    print("\n[Team Assignment]")
    print(f"  Mode: {ta['mode']}")
    print(f"  Team distribution: {ta.get('team_distribution')}")
    print(f"  Low-confidence tracks: {ta.get('low_confidence_track_ids')}")
    if "team_accuracy" in ta:
        print(f"  Team accuracy: {ta['team_accuracy']}")
        # KMeans team labels are arbitrary, so we also show whether the evaluator
        # used the normal mapping or flipped 0<->1 mapping.
        if "normal_accuracy" in ta and "flipped_accuracy" in ta:
            print(f"  Normal / flipped accuracy: {ta['normal_accuracy']} / {ta['flipped_accuracy']}")
            print(f"  Mapping used: {ta.get('team_label_mapping_used')}")

    reid = report["reid"]
    print("\n[ReID]")
    print(f"  Mode: {reid['mode']}")
    print(f"  Local IDs -> Global IDs: {reid.get('num_local_track_ids')} -> {reid.get('num_global_ids')}")
    print(f"  Num merges: {reid.get('num_merges')}")
    print(f"  Merged groups: {reid.get('merged_groups')}")
    if "pairwise_metrics" in reid:
        m = reid["pairwise_metrics"]
        print(f"  Pairwise Precision/Recall/F1: {m['precision']} / {m['recall']} / {m['f1']}")
        print(f"  False merge pairs: {len(reid.get('false_merge_pairs', []))}")
        print(f"  Missed merge pairs: {len(reid.get('missed_merge_pairs', []))}")

        # Print a small preview only; the full lists are stored in evaluation_report.json.
        false_preview = reid.get('false_merge_pairs', [])[:10]
        missed_preview = reid.get('missed_merge_pairs', [])[:10]
        if false_preview:
            print(f"  False merge preview: {false_preview}")
        if missed_preview:
            print(f"  Missed merge preview: {missed_preview}")

    pc = report["pipeline_consistency"]
    print("\n[Pipeline Consistency]")
    print(f"  Mixed-team global groups: {pc['num_mixed_team_groups']}")
    print(f"  Temporal-conflict global groups: {pc['num_temporal_conflict_groups']}")
    print(f"  Low-confidence merged groups: {pc['num_low_confidence_merged_groups']}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Object Detection, Tracking, Team Assignment, and ReID outputs.")

    p.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS_PATH)
    p.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS_PATH)
    p.add_argument("--team", type=Path, default=DEFAULT_TEAM_PATH)
    p.add_argument("--reid-report", type=Path, default=DEFAULT_REID_REPORT_PATH)
    p.add_argument("--reid-map", type=Path, default=DEFAULT_REID_MAP_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)

    p.add_argument("--gt-detections", type=Path, default=None)
    p.add_argument("--gt-tracks", type=Path, default=None)
    p.add_argument("--gt-teams", type=Path, default=None)
    p.add_argument("--gt-reid", type=Path, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = build_evaluation_report(args)
    save_json(report, args.out)
    print_short_summary(report)


if __name__ == "__main__":
    main()
