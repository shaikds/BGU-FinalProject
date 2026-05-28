# ensemble_inference.py

import json
import numpy as np
from collections import defaultdict

def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)

def ensemble_predictions(result_files, weights=None):
    """
    Ensemble multiple inference results.
    """
    if weights is None:
        weights = [1.0] * len(result_files)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Load all results
    all_results = [load_results(f) for f in result_files]
    
    # Group predictions by (frame, label)
    combined = defaultdict(list)
    for results, weight in zip(all_results, weights):
        for pred in results['predictions']:
            key = (pred['frame'], pred['label'])
            combined[key].append(pred['confidence'] * weight)
    
    # Average predictions
    ensemble_preds = []
    for (frame, label), confs in combined.items():
        avg_conf = sum(confs)  # Weighted sum (weights already applied)
        ensemble_preds.append({
            'frame': frame,
            'label': label,
            'confidence': avg_conf,
            'num_models': len(confs)  # Agreement count
        })
    
    # Sort by frame
    ensemble_preds.sort(key=lambda x: x['frame'])
    
    return {'predictions': ensemble_preds}

# Usage:
# Run inference with both models first:
# python inference.py --model SoccerNetBall_challenge1 --video_path video.mp4 --inference_threshold 0.01
# mv inference_output/results_inference.json inference_output/results_model1.json
# python inference.py --model SoccerNetBall_challenge2 --video_path video.mp4 --inference_threshold 0.01
# mv inference_output/results_inference.json inference_output/results_model2.json

results = ensemble_predictions([
    'inference_output/results_model1.json',
    'inference_output/results_model2.json'
], weights=[0.6, 0.4])  # Weight better model higher

# Filter by agreement - keep if 2+ models agree
high_agreement = [p for p in results['predictions'] if p['num_models'] >= 2]

with open('inference_output/results_ensemble.json', 'w') as f:
    json.dump({'predictions': high_agreement}, f, indent=2)