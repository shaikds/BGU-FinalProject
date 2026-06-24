#!/usr/bin/env python3
"""
Filter events in a JSON `predictions` file by per-label confidence thresholds.

Features:
- Accepts any JSON with a top-level `predictions` list of objects with `label` and `confidence`.
- Thresholds passed as repeated `--threshold "LABEL=VALUE"` arguments.
- Optional mapping: any label containing "TACKLE" -> "BALL PLAYER BLOCK" with `--map-tackle`.
- Optional removal of `DRIVE` labels with `--remove-drive`.
- Default threshold for unspecified labels via `--default-threshold`.

Example:
  python3 scripts/filter_by_confidence.py \
    --json inference_output/results_snball.json \
    --output inference_output/results_snball_filtered.json \
    --threshold PASS=0.61 \
    --threshold "HIGH PASS=0.49" \
    --threshold "BALL PLAYER BLOCK=0.41" \
    --map-tackle

"""
import argparse
import json
import os
import sys
from collections import Counter


def parse_threshold_entry(s):
    if '=' not in s:
        raise argparse.ArgumentTypeError('Threshold must be in LABEL=VALUE format')
    label, val = s.split('=', 1)
    label = label.strip()
    try:
        thr = float(val.strip())
    except Exception:
        raise argparse.ArgumentTypeError(f'Invalid threshold value: {s}')
    return label.upper(), thr


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def normalize_label(label, map_tackle=False):
    if not isinstance(label, str):
        return label
    lab = label.strip()
    if map_tackle and 'TACKLE' in lab.upper():
        return 'BALL PLAYER BLOCK'
    return lab


def filter_predictions(preds, thresholds_map, default_threshold=0.0, map_tackle=False, remove_drive=False):
    out = []
    removed = 0
    for p in preds:
        lab = p.get('label', '')
        lab_norm = normalize_label(lab, map_tackle=map_tackle)
        if not isinstance(lab_norm, str):
            continue
        if remove_drive and lab_norm.upper() == 'DRIVE':
            removed += 1
            continue
        conf = p.get('confidence', None)
        try:
            conf_val = float(conf) if conf is not None else 0.0
        except Exception:
            conf_val = 0.0

        thr = thresholds_map.get(lab_norm.upper(), default_threshold)
        if conf_val >= thr:
            # write normalized label back
            p['label'] = lab_norm
            out.append(p)
        else:
            removed += 1
    return out, removed


def main(argv=None):
    p = argparse.ArgumentParser(description='Filter events by per-label confidence thresholds')
    p.add_argument('--json', '-j', required=True, help='Input JSON file (must contain top-level "predictions" list)')
    p.add_argument('--output', '-o', required=True, help='Output JSON file')
    p.add_argument('--threshold', '-t', action='append', default=[], help='Threshold entry in LABEL=VALUE (repeatable)')
    p.add_argument('--default-threshold', '-d', type=float, default=0.0, help='Default threshold for labels without explicit entries')
    p.add_argument('--map-tackle', action='store_true', help='Map any label containing "TACKLE" to "BALL PLAYER BLOCK"')
    p.add_argument('--remove-drive', action='store_true', help='Remove labels equal to DRIVE')
    p.add_argument('--inplace', action='store_true', help='Overwrite input file (writes to --json)')
    args = p.parse_args(argv)

    if not os.path.exists(args.json):
        print(f'Input JSON not found: {args.json}', file=sys.stderr)
        sys.exit(2)

    # parse thresholds
    thr_map = {}
    for entry in args.threshold:
        label, thr = parse_threshold_entry(entry)
        thr_map[label.upper()] = thr

    data = load_json(args.json)
    preds = data.get('predictions', [])
    orig_count = len(preds)

    filtered, removed = filter_predictions(preds, thr_map, default_threshold=args.default_threshold, map_tackle=args.map_tackle, remove_drive=args.remove_drive)

    data['predictions'] = filtered

    out_path = args.json if args.inplace else args.output
    write_json(out_path, data)

    counts = Counter([p.get('label', '') for p in filtered])

    print(f'Input file: {args.json}')
    print(f'Output file: {out_path}')
    print(f'Original predictions: {orig_count}')
    print(f'Remaining predictions: {len(filtered)}')
    print(f'Removed predictions: {removed}')
    print('Counts by label:')
    for k, v in counts.most_common():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
