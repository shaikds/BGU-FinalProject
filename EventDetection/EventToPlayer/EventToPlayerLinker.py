"""
Event-to-Player Linker (Method 2: Temporal Window + Proximity Voting)

Links T-DEED ball action events to player IDs using tracking+ReID data.
For each event, looks at a temporal window around the event frame,
finds the nearest player to the ball in each frame, and assigns
the event to the player with the most "votes" (weighted by distance).

INPUT FORMAT:
=============
1) Tracking + ReID data (JSON or list of dicts):
   Each entry: {"x": float, "y": float, "type": int, "frame": int, "id": int}
   - type: 0 = ball, 1 = player, 2 = referee, 3 = goalkeeper
   - id: unique ReID identity for players/keepers, ignored for ball
   - x, y: position coordinates (pixels or pitch coordinates)
   - frame: frame number

2) T-DEED events (JSON or list of dicts):
   Each entry: {"frame": int, "type": str, "confidence": float}
   - frame: exact frame number where the event was detected
   - type: event type string (e.g., "PASS", "SHOT", "HEADER", etc.)
   - confidence: detection confidence [0.0 - 1.0]

OUTPUT FORMAT:
==============
List of dicts, one per event:
{
    "event_frame": int,
    "event_type": str,
    "event_confidence": float,
    "assigned_player_id": int or None,
    "assignment_confidence": float,   # how confident we are in the assignment [0-1]
    "player_distance": float,         # avg distance of assigned player to ball
    "votes": dict,                    # {player_id: weighted_vote_count}
    "ball_found_in_window": bool      # whether ball was detected in the window
}

USAGE:
======
    # From code:
    from event_player_linker import link_events_to_players
    results = link_events_to_players(tracking_data, tdeed_events)

    # From command line:
    python event_player_linker.py --tracking tracking.json --events events.json --output results.json

    # With custom parameters:
    python event_player_linker.py --tracking tracking.json --events events.json --output results.json \
        --window 5 --min-confidence 0.3 --include-keepers
"""

import json
import argparse
import math
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


# =============================================================================
# Type constants (RF-DETR SoccerNet class IDs)
# https://huggingface.co/julianzu9612/RFDETR-Soccernet
# =============================================================================
TYPE_BALL = 0
TYPE_PLAYER = 1
TYPE_REFEREE = 2
TYPE_KEEPER = 3


def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _build_frame_index(tracking_data: List[Dict]) -> Dict[int, List[Dict]]:
    """Index tracking data by frame number for fast lookup."""
    frame_index = defaultdict(list)
    for entry in tracking_data:
        frame_index[entry["frame"]].append(entry)
    return frame_index


def _get_ball_position(frame_entries: List[Dict]) -> Optional[Tuple[float, float]]:
    """Extract ball position from a frame's tracking entries."""
    for entry in frame_entries:
        if entry["type"] == TYPE_BALL:
            return (entry["x"], entry["y"])
    return None


def _get_players_in_frame(
    frame_entries: List[Dict], include_keepers: bool = True
) -> List[Dict]:
    """Extract player (and optionally keeper) entries from a frame."""
    valid_types = {TYPE_PLAYER}
    if include_keepers:
        valid_types.add(TYPE_KEEPER)
    return [e for e in frame_entries if e["type"] in valid_types]


def link_events_to_players(
    tracking_data: List[Dict],
    tdeed_events: List[Dict],
    window_size: int = 5,
    min_confidence: float = 0.0,
    include_keepers: bool = True,
    distance_weighting: bool = True,
) -> List[Dict]:
    """
    Link T-DEED events to player IDs using temporal window proximity voting.

    Parameters
    ----------
    tracking_data : list of dict
        Tracking+ReID entries with keys: x, y, type, frame, id
    tdeed_events : list of dict
        T-DEED event entries with keys: frame, type, confidence
    window_size : int
        Number of frames to look before AND after the event frame.
        Total window = 2 * window_size + 1 frames. Default: 5.
    min_confidence : float
        Minimum T-DEED confidence to process an event. Default: 0.0 (all events).
    include_keepers : bool
        Whether goalkeepers (type=3) can be assigned events. Default: True.
    distance_weighting : bool
        If True, votes are weighted by 1/distance (closer = stronger vote).
        If False, each frame gives 1 unweighted vote to the nearest player.

    Returns
    -------
    list of dict
        One result per event with assignment details (see module docstring).
    """
    # Build frame index for O(1) lookup
    frame_index = _build_frame_index(tracking_data)

    results = []

    for event in tdeed_events:
        event_frame = event["frame"]
        event_type = event["type"]
        event_conf = event["confidence"]

        # Skip low-confidence events
        if event_conf < min_confidence:
            continue

        # Collect votes across the temporal window
        votes = defaultdict(float)          # player_id -> weighted vote
        distances_sum = defaultdict(float)  # player_id -> sum of distances
        distances_count = defaultdict(int)  # player_id -> count
        ball_found = False

        for offset in range(-window_size, window_size + 1):
            f = event_frame + offset
            if f not in frame_index:
                continue

            frame_entries = frame_index[f]
            ball_pos = _get_ball_position(frame_entries)

            if ball_pos is None:
                continue

            ball_found = True
            players = _get_players_in_frame(frame_entries, include_keepers)

            if not players:
                continue

            # Find nearest player to ball in this frame
            best_player = None
            best_dist = float("inf")

            for p in players:
                dist = _euclidean_distance(p["x"], p["y"], ball_pos[0], ball_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_player = p["id"]

            if best_player is not None:
                if distance_weighting and best_dist > 0:
                    votes[best_player] += 1.0 / best_dist
                else:
                    votes[best_player] += 1.0

                distances_sum[best_player] += best_dist
                distances_count[best_player] += 1

        # Determine winner
        assigned_id = None
        assignment_conf = 0.0
        avg_distance = 0.0

        if votes:
            # Winner = player with highest vote total
            assigned_id = max(votes, key=votes.get)
            total_votes = sum(votes.values())
            assignment_conf = votes[assigned_id] / total_votes if total_votes > 0 else 0

            if distances_count[assigned_id] > 0:
                avg_distance = distances_sum[assigned_id] / distances_count[assigned_id]

        results.append(
            {
                "event_frame": event_frame,
                "event_type": event_type,
                "event_confidence": event_conf,
                "assigned_player_id": assigned_id,
                "assignment_confidence": round(assignment_conf, 4),
                "player_distance": round(avg_distance, 2),
                "votes": dict(votes),
                "ball_found_in_window": ball_found,
            }
        )

    return results


def print_summary(results: List[Dict]) -> None:
    """Print a human-readable summary of the linking results."""
    total = len(results)
    assigned = sum(1 for r in results if r["assigned_player_id"] is not None)
    no_ball = sum(1 for r in results if not r["ball_found_in_window"])

    print(f"\n{'='*60}")
    print(f"  Event-to-Player Linking Summary")
    print(f"{'='*60}")
    print(f"  Total events processed : {total}")
    print(f"  Successfully assigned  : {assigned}/{total}")
    print(f"  No ball in window      : {no_ball}")
    print(f"{'='*60}\n")

    for r in results:
        pid = r["assigned_player_id"]
        pid_str = f"Player {pid}" if pid is not None else "UNASSIGNED"
        conf_str = f"{r['assignment_confidence']:.1%}"
        print(
            f"  Frame {r['event_frame']:>6d} | {r['event_type']:<12s} | "
            f"conf={r['event_confidence']:.2f} | -> {pid_str:<12s} | "
            f"match={conf_str} | dist={r['player_distance']:.1f}"
        )

    print()


# =============================================================================
# CLI entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Link T-DEED events to player IDs via temporal proximity voting."
    )
    parser.add_argument(
        "--tracking", required=True, help="Path to tracking+ReID JSON file"
    )
    parser.add_argument(
        "--events", required=True, help="Path to T-DEED events JSON file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write results JSON"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Temporal window size (frames before/after). Default: 5",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum T-DEED confidence threshold. Default: 0.0",
    )
    parser.add_argument(
        "--include-keepers",
        action="store_true",
        default=True,
        help="Allow keepers (type=3) to be assigned events. Default: True",
    )
    parser.add_argument(
        "--no-distance-weighting",
        action="store_true",
        default=False,
        help="Disable distance weighting (use simple vote count instead)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Print human-readable summary to console",
    )

    args = parser.parse_args()

    # Load inputs
    with open(args.tracking, "r") as f:
        tracking_data = json.load(f)

    with open(args.events, "r") as f:
        tdeed_events = json.load(f)

    print(f"Loaded {len(tracking_data)} tracking entries")
    print(f"Loaded {len(tdeed_events)} T-DEED events")

    # Run linking
    results = link_events_to_players(
        tracking_data=tracking_data,
        tdeed_events=tdeed_events,
        window_size=args.window,
        min_confidence=args.min_confidence,
        include_keepers=args.include_keepers,
        distance_weighting=not args.no_distance_weighting,
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} results to {args.output}")

    if args.summary:
        print_summary(results)


if __name__ == "__main__":
    main()