# save as: ~/T-DEED-2/sequence_filter.py

import json
from collections import defaultdict

VIDEO_FPS = 30

# Load raw predictions (run with --inference_threshold 0.05)
with open('inference_output/results_inference.json', 'r') as f:
    data = json.load(f)

predictions = data['predictions']
print(f"Raw predictions: {len(predictions)}")

# ============================================
# CONFIG
# ============================================

# Classes to IGNORE (too noisy, not interesting)
IGNORE_CLASSES = {"DRIVE"}  # DRIVE is mostly noise

# High-value events (keep even at lower confidence)
HIGH_VALUE_EVENTS = {"GOAL", "SHOT", "FREE KICK", "CROSS", "HEADER"}

# Expected sequences (if we see A, look for B nearby)
EVENT_SEQUENCES = {
    "SHOT": ["GOAL", "BALL PLAYER BLOCK", "OUT"],
    "CROSS": ["HEADER", "SHOT", "GOAL"],
    "FREE KICK": ["SHOT", "GOAL", "CROSS"],
}

# ============================================
# STEP 1: Remove DRIVE noise
# ============================================

filtered = [p for p in predictions if p['label'] not in IGNORE_CLASSES]
print(f"After removing DRIVE: {len(filtered)}")

# ============================================
# STEP 2: Temporal clustering - find "event windows"
# ============================================

def find_event_clusters(preds, window_sec=2.0):
    """
    Group predictions into time windows.
    Real events have multiple predictions clustered together.
    """
    window_frames = int(window_sec * VIDEO_FPS)
    
    # Sort by frame
    preds_sorted = sorted(preds, key=lambda x: x['frame'])
    
    clusters = []
    current_cluster = []
    
    for pred in preds_sorted:
        if not current_cluster:
            current_cluster = [pred]
        elif pred['frame'] - current_cluster[-1]['frame'] <= window_frames:
            current_cluster.append(pred)
        else:
            if len(current_cluster) >= 2:  # At least 2 predictions
                clusters.append(current_cluster)
            current_cluster = [pred]
    
    # Don't forget last cluster
    if len(current_cluster) >= 2:
        clusters.append(current_cluster)
    
    return clusters

clusters = find_event_clusters(filtered, window_sec=1.5)
print(f"Found {len(clusters)} event clusters")

# ============================================
# STEP 3: Score each cluster
# ============================================

def score_cluster(cluster):
    """
    Score a cluster based on:
    - Number of predictions
    - Max confidence
    - Presence of high-value events
    - Event sequences (CROSS → SHOT → GOAL)
    """
    labels = set(p['label'] for p in cluster)
    max_conf = max(p['confidence'] for p in cluster)
    
    score = 0
    
    # Base score from confidence
    score += max_conf * 2
    
    # Bonus for multiple predictions
    score += min(len(cluster) * 0.1, 0.5)
    
    # Bonus for high-value events
    high_value_count = sum(1 for p in cluster if p['label'] in HIGH_VALUE_EVENTS)
    score += high_value_count * 0.2
    
    # Bonus for sequences (SHOT + GOAL = very likely real)
    if "GOAL" in labels and "SHOT" in labels:
        score += 0.5
    if "CROSS" in labels and ("HEADER" in labels or "SHOT" in labels):
        score += 0.3
    
    return score

# ============================================
# STEP 4: Extract best event from each cluster
# ============================================

def extract_events(clusters, min_score=0.5):
    """
    From each cluster, extract the most important events.
    """
    results = []
    
    for cluster in clusters:
        cluster_score = score_cluster(cluster)
        
        if cluster_score < min_score:
            continue
        
        # Get time range
        start_frame = min(p['frame'] for p in cluster)
        end_frame = max(p['frame'] for p in cluster)
        center_frame = (start_frame + end_frame) // 2
        
        # Find best prediction per class in cluster
        best_per_class = {}
        for p in cluster:
            label = p['label']
            if label not in best_per_class or p['confidence'] > best_per_class[label]['confidence']:
                best_per_class[label] = p
        
        # Prioritize high-value events
        for label in HIGH_VALUE_EVENTS:
            if label in best_per_class:
                event = best_per_class[label].copy()
                event['cluster_score'] = cluster_score
                event['cluster_size'] = len(cluster)
                results.append(event)
        
        # Add PASS/other only if no high-value event
        if not any(l in best_per_class for l in HIGH_VALUE_EVENTS):
            best = max(best_per_class.values(), key=lambda x: x['confidence'])
            if best['confidence'] >= 0.3:  # Higher threshold for non-important
                event = best.copy()
                event['cluster_score'] = cluster_score
                event['cluster_size'] = len(cluster)
                results.append(event)
    
    return results

events = extract_events(clusters, min_score=0.4)
print(f"Extracted {len(events)} events")

# ============================================
# STEP 5: Add isolated high-confidence events
# ============================================

# Some real events might not have clusters
clustered_frames = set()
for cluster in clusters:
    for p in cluster:
        clustered_frames.add(p['frame'])

isolated_high_conf = []
for p in filtered:
    if p['frame'] not in clustered_frames:
        if p['confidence'] >= 0.5 or (p['label'] in HIGH_VALUE_EVENTS and p['confidence'] >= 0.35):
            isolated_high_conf.append(p)

events.extend(isolated_high_conf)
print(f"After adding isolated high-conf: {len(events)}")

# ============================================
# STEP 6: Final NMS
# ============================================

def final_nms(events, window_sec=1.5):
    """Remove duplicates, keep best per window."""
    window_frames = int(window_sec * VIDEO_FPS)
    
    events_sorted = sorted(events, key=lambda x: -x['confidence'])
    
    kept = []
    for event in events_sorted:
        # Check if too close to already kept event of same class
        too_close = any(
            event['label'] == k['label'] and abs(event['frame'] - k['frame']) <= window_frames
            for k in kept
        )
        if not too_close:
            kept.append(event)
    
    kept.sort(key=lambda x: x['frame'])
    return kept

final_events = final_nms(events)
print(f"Final events: {len(final_events)}")

# ============================================
# OUTPUT
# ============================================

# Add timestamps
for e in final_events:
    time_sec = e['frame'] / VIDEO_FPS
    e['timestamp'] = f"{int(time_sec//60):02d}:{time_sec%60:05.2f}"

# Save
output = {'predictions': final_events}
with open('inference_output/results_sequence.json', 'w') as f:
    json.dump(output, f, indent=2)

# Print
print(f"\n{'='*70}")
print(f"{'Time':<10} {'Label':<25} {'Conf':>6} {'Cluster':>8}")
print(f"{'='*70}")

for e in final_events:
    cluster_info = e.get('cluster_size', '-')
    print(f"{e['timestamp']:<10} {e['label']:<25} {e['confidence']:>6.2f} {str(cluster_info):>8}")

# Summary by class
print(f"\n--- Summary ---")
class_counts = defaultdict(int)
for e in final_events:
    class_counts[e['label']] += 1

for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"{label}: {count}")