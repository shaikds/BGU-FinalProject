import json

files = {
    "pred": "/home/shaikar/sn_pipe_trial/outputs/detections/predictions.json",
    "tracks": "/home/shaikar/sn_pipe_trial/outputs/tracks/tracks.json",
    "team": "/home/shaikar/sn_pipe_trial/outputs/team_assignment_v2/team_assignment_v2.json",
    "reid": "/home/shaikar/sn_pipe_trial/outputs/reid_v2/reid_observations.json",
}

for name, path in files.items():
    data = json.load(open(path))
    if name == "pred":
        rows = [d for f in data["frames"] for d in f["detections"]]
    elif name == "team":
        rows = data["tracks"]
    elif name == "reid":
        rows = data["observations"]
    else:
        rows = data["tracks"]

    print(name, sum(int(r["label"]) == 0 for r in rows))
