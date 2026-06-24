#!/usr/bin/env python3
"""
Filter events in a JSON file by per-label confidence thresholds.

Examples:
  python3 scripts/filter_events.py \
    --input inference_output/results_snball.json \
    --output inference_output/results_snball_filtered.json \
    --threshold "PASS=0.61" --threshold "HIGH PASS=0.49" --threshold "BALL PLAYER BLOCK=0.41" \
    --case-insensitive

Behavior:
- By default, labels not present in thresholds are kept (no filtering).
- Use `--default-threshold` to apply a fallback threshold to all unspecified labels.
- Supports mapping any label containing "TACKLE" -> "BALL PLAYER BLOCK" via `--map-tackle`.
- Supports removing `DRIVE` with `--remove-drive`.
- You can specify the JSON list key (default `predictions`) and the label/confidence keys.
"""

import argparse
import json
import os
import sys
from collections import Counter


def parse_threshold_arg(s):
    if '=' not in s:
        raise argparse.ArgumentTypeError('threshold must be LABEL=VALUE')
    label, val = s.split('=', 1)
    try:
        v = float(val)
    except Exception:
        raise argparse.ArgumentTypeError(f'Invalid threshold value: {val}')
    return label.strip(), v


def load_thresholds(th_args, thresholds_json, case_insensitive):
    thr = {}
    if thresholds_json:
        with open(thresholds_json, 'r') as f:
            j = json.load(f)
        if not isinstance(j, dict):
            raise ValueError('thresholds JSON must be an object mapping labels to numbers')
        for k, v in j.items():
            key = k.strip().lower() if case_insensitive else k.strip()
            thr[key] = float(v)
    for a in (th_args or []):
        k, v = parse_threshold_arg(a)
        key = k.lower() if case_insensitive else k
        thr[key] = float(v)
    return thr


def filter_predictions(preds, label_key, conf_key, thresholds, default_thr, case_insensitive, map_tackle, remove_drive, drop_missing_conf):
    filtered = []
    before = Counter()
    after = Counter()

    for ev in preds:
        label = ev.get(label_key, '')
        if not isinstance(label, str):
            label = str(label)
        label_norm = label.strip()

        # map tackles to BALL PLAYER BLOCK if requested
        if map_tackle and 'tackle' in label_norm.lower():
            label_norm = 'BALL PLAYER BLOCK'
            ev[label_key] = label_norm

        # remove DRIVE if requested
        if remove_drive and label_norm.strip().upper() == 'DRIVE':
            before['DRIVE'] += 1
            continue

        key = label_norm.lower() if case_insensitive else label_norm
        before[label_norm] += 1

        thr = thresholds.get(key) if thresholds else None
        if thr is None:
            thr = default_thr

        conf = ev.get(conf_key)
        if conf is None:
            if thr is None:
                # no threshold -> keep
                filtered.append(ev)
                after[label_norm] += 1
                continue
            else:
                if drop_missing_conf:
                    # drop events missing confidence when a threshold applies
                    continue
                conf_val = 0.0
        else:
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = 0.0

        if thr is not None and conf_val < thr:
            # drop
            continue

        filtered.append(ev)
        after[label_norm] += 1

    return filtered, before, after


def main():
    p = argparse.ArgumentParser(description='Filter events in a JSON list by per-label confidence')
    p.add_argument('--input', '-i', required=True, help='Input JSON file')
    p.add_argument('--output', '-o', required=True, help='Output JSON file (filtered)')
    p.add_argument('--list-key', default='predictions', help='Key in JSON containing the list (default: predictions)')
    p.add_argument('--label-key', default='label', help='Field name for label in each entry')
    p.add_argument('--confidence-key', default='confidence', help='Field name for confidence in each entry')
    p.add_argument('--threshold', '-t', action='append', help='Per-label threshold as "LABEL=VALUE" (can repeat)')
    p.add_argument('--thresholds-json', help='JSON file mapping LABEL->threshold')
    p.add_argument('--default-threshold', type=float, default=None, help='Default threshold for labels not specified (if set)')
    p.add_argument('--case-insensitive', action='store_true', help='Match labels case-insensitively')
    p.add_argument('--map-tackle', action='store_true', help='Map any label containing "TACKLE" to "BALL PLAYER BLOCK"')
    p.add_argument('--remove-drive', action='store_true', help='Remove events whose label is DRIVE')
    p.add_argument('--drop-missing-confidence', action='store_true', help='Drop events missing a confidence when a threshold applies')
    p.add_argument('--pretty', action='store_true', help='Write pretty-printed JSON')

    args = p.parse_args()

    if not os.path.exists(args.input):
        print('Input file not found:', args.input, file=sys.stderr)
        sys.exit(2)

    with open(args.input, 'r') as f:
        data = json.load(f)

    if args.list_key not in data or not isinstance(data[args.list_key], list):
        print(f'Expected a list at key "{args.list_key}" in the JSON', file=sys.stderr)
        sys.exit(3)

    thresholds = load_thresholds(args.threshold, args.thresholds_json, args.case_insensitive)

    preds = data[args.list_key]
    filtered, before_counts, after_counts = filter_predictions(preds,
                                                               args.label_key,
                                                               args.confidence_key,
                                                               thresholds,
                                                               args.default_threshold,
                                                               args.case_insensitive,
                                                               args.map_tackle,
                                                               args.remove_drive,
                                                               args.drop_missing_confidence)

    data_out = dict(data)
    data_out[args.list_key] = filtered

    with open(args.output, 'w') as f:
        if args.pretty:
            json.dump(data_out, f, indent=4)
        else:
            json.dump(data_out, f)

    print(f'Input events: {len(preds)}; Output events: {len(filtered)}')
    print('Counts before:')
    for k, v in sorted(before_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f'  {k}: {v}')
    print('Counts after:')
    for k, v in sorted(after_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
