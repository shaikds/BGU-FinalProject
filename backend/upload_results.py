#!/usr/bin/env python3
"""
upload_results.py  –  Push event_player_linker.py output to the backend.

Usage (minimal):
    python upload_results.py results.json

Usage (with full game metadata):
    python upload_results.py results.json \
        --home-team "מכבי חיפה" \
        --away-team "הפועל באר שבע" \
        --home-score 2 \
        --away-score 1 \
        --date "15.05.2025" \
        --time "20:00" \
        --url http://localhost:8000
"""
import argparse
import json
import sys
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file",    help="Path to linker output JSON")
    parser.add_argument("--home-team",  default="קבוצה בית")
    parser.add_argument("--away-team",  default="קבוצה אורחת")
    parser.add_argument("--home-score", type=int, default=0)
    parser.add_argument("--away-score", type=int, default=0)
    parser.add_argument("--date",       default="")
    parser.add_argument("--time",       default="")
    parser.add_argument("--url",        default="http://localhost:8000")
    args = parser.parse_args()

    with open(args.json_file) as f:
        payload = json.load(f)

    # Inject game metadata
    payload["game"] = {
        "home_team":  args.home_team,
        "away_team":  args.away_team,
        "home_score": args.home_score,
        "away_score": args.away_score,
        "date":       args.date,
        "time":       args.time,
    }

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{args.url.rstrip('/')}/sessions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"✅  Uploaded!")
            print(f"    session_id    : {result['session_id']}")
            print(f"    events stored : {result['events_stored']}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"❌  HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
