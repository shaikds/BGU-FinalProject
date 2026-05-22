# Soccer Event-Player Linker – Backend

FastAPI + MongoDB backend that stores and serves the output of `event_player_linker.py`.

---

## Setup

```bash
# 1. Create & activate a conda env (or any venv)
conda create -n soccer_backend python=3.11
conda activate soccer_backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure database
cp .env.example .env
# edit .env with your MongoDB URI

# 4. Start MongoDB locally (if not using Atlas)
mongod --dbpath /data/db   # or via Docker: docker run -p 27017:27017 mongo

# 5. Run the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## Uploading results

```bash
# From command line
python upload_results.py /path/to/linker_output.json

# Or with curl
curl -X POST http://localhost:8000/sessions \
     -H "Content-Type: application/json" \
     -d @/path/to/linker_output.json
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Upload a full linker JSON |
| `GET`  | `/sessions` | List all sessions |
| `GET`  | `/sessions/{id}` | Get session metadata |
| `DELETE` | `/sessions/{id}` | Delete session + its events |
| `GET`  | `/sessions/{id}/events` | List events (filterable) |
| `GET`  | `/sessions/{id}/players/{player_id}/events` | Events for one player |
| `GET`  | `/sessions/{id}/stats` | Aggregated stats for the dashboard |

### Event filters (query params on `/events`)

| Param | Type | Description |
|-------|------|-------------|
| `event_type` | string | e.g. `PASS`, `SHOT`, `GOAL` |
| `player_id` | int | filter to one player |
| `assigned_only` | bool | exclude unassigned events |
| `min_confidence` | float 0–1 | minimum event confidence |
| `skip` / `limit` | int | pagination |

---

## Data model stored per event

```
event_frame            int      – frame number in the video
event_time_sec         float    – derived: frame / fps
event_type             str      – PASS / DRIVE / SHOT / GOAL / …
event_confidence       float
event_source           str      – SoccerNetBall | SoccerNet
assigned_player_id     int|null
assigned_track_id      int|null
assignment_confidence  float
player_distance        float    – pixels between ball and player bbox
num_ball_frames_in_window int
ball_found_in_window   bool
votes                  dict     – {player_id: weighted_vote_score}
session_id             str      – parent session ObjectId
```

---

## Frontend integration (Next.js example)

```ts
// Fetch events for the table
const res = await fetch(
  `http://localhost:8000/sessions/${sessionId}/events?limit=200&assigned_only=true`
);
const events = await res.json();

// Fetch stats for the summary cards
const stats = await fetch(
  `http://localhost:8000/sessions/${sessionId}/stats`
).then(r => r.json());
```
