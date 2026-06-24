#!/usr/bin/env python3
import json
import os
import sys

infile = 'inference_output/results_snball.json'
if not os.path.exists(infile):
    print(f"File not found: {infile}", file=sys.stderr)
    sys.exit(1)

with open(infile, 'r') as f:
    data = json.load(f)

preds = []
for p in data.get('predictions', []):
    label = p.get('label', '')
    if not isinstance(label, str):
        preds.append(p)
        continue
    # remove DRIVE entries
    if label.strip().upper() == 'DRIVE':
        continue
    # replace any tackle/tackles occurrences with BALL PLAYER BLOCK
    if 'TACKLE' in label.upper():
        p['label'] = 'BALL PLAYER BLOCK'
    preds.append(p)

data['predictions'] = preds

with open(infile, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Wrote {len(preds)} predictions to {infile}")
