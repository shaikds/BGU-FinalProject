"""
Soccer Event-Player Linker Backend
FastAPI + MongoDB  —  Python 3.8 compatible
Accepts both old and new event_player_linker.py JSON schemas.

Install:
    pip install fastapi uvicorn motor pydantic python-dotenv

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, model_validator

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB  = os.getenv("MONGODB_DB",  "soccer_analysis")

client: Optional[AsyncIOMotorClient] = None


async def get_db():
    return client[MONGODB_DB]


# ── Pydantic models ───────────────────────────────────────────────────────────

class AnalysisParameters(BaseModel):
    """
    Flexible — accepts both old schema (distance_weighting) and new schema
    (contact_margin_factor, overlap_bonus, contact_bonus, method).
    Unknown extra fields are silently accepted.
    """
    window_size:            int
    min_confidence:         float
    include_keepers:        bool
    temporal_sigma:         float
    # Old schema field (optional for new JSONs)
    distance_weighting:     Optional[bool]  = None
    # New schema fields (optional for old JSONs)
    contact_margin_factor:  Optional[float] = None
    overlap_bonus:          Optional[float] = None
    contact_bonus:          Optional[float] = None
    method:                 Optional[str]   = None

    model_config = {"extra": "allow"}   # silently ignore any other new fields


class EventResult(BaseModel):
    """
    Accepts both:
      - Old schema: player_distance  (float)
      - New schema: player_feet_distance (float) + bbox_overlap_frames (int)
    If player_distance is absent, it is copied from player_feet_distance.
    """
    event_frame:               int
    event_type:                str
    event_confidence:          float
    event_source:              str
    assigned_player_id:        Optional[int]   = None
    assigned_track_id:         Optional[int]   = None
    assignment_confidence:     float
    ball_found_in_window:      bool
    num_ball_frames_in_window: int
    votes:                     Dict[str, float]
    event_time_sec:            Optional[float] = None

    # Distance field — one or both may be present depending on schema version
    player_distance:           Optional[float] = None   # old schema
    player_feet_distance:      Optional[float] = None   # new schema

    # New schema only
    bbox_overlap_frames:       Optional[int]   = None

    model_config = {"extra": "allow"}   # accept any future fields

    @model_validator(mode="after")
    def normalise_distance(self) -> "EventResult":
        """Ensure player_distance is always set regardless of schema version."""
        if self.player_distance is None and self.player_feet_distance is not None:
            self.player_distance = self.player_feet_distance
        if self.player_distance is None:
            self.player_distance = 0.0
        return self


class GameMetadata(BaseModel):
    home_team:  str = "קבוצה בית"
    away_team:  str = "קבוצה אורחת"
    home_score: int = 0
    away_score: int = 0
    date:       str = ""
    time:       str = ""


class AnalysisSessionIn(BaseModel):
    video:        str
    fps:          float
    num_events:   int
    num_assigned: int
    parameters:   AnalysisParameters
    results:      List[EventResult]
    game:         GameMetadata = GameMetadata()

    @model_validator(mode="after")
    def compute_timestamps(self) -> "AnalysisSessionIn":
        for ev in self.results:
            if ev.event_time_sec is None:
                ev.event_time_sec = round(ev.event_frame / self.fps, 3)
        return self


class GameOut(BaseModel):
    id:           str
    homeTeam:     str
    awayTeam:     str
    homeScore:    int
    awayScore:    int
    date:         str
    time:         str
    players:      int
    totalActions: int


class SessionDetailOut(GameOut):
    video:        str
    fps:          float
    num_events:   int
    num_assigned: int
    coverage_pct: float
    uploaded_at:  datetime
    parameters:   AnalysisParameters


# ── Helpers ───────────────────────────────────────────────────────────────────

def oid(raw: str) -> ObjectId:
    try:
        return ObjectId(raw)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid id: {raw}")


async def _count_unique_players(db, session_id: str) -> int:
    pipeline = [
        {"$match": {"session_id": session_id, "assigned_player_id": {"$ne": None}}},
        {"$group": {"_id": "$assigned_player_id"}},
        {"$count": "total"},
    ]
    r = await db.events.aggregate(pipeline).to_list(1)
    return r[0]["total"] if r else 0


async def _count_total_assigned(db, session_id: str) -> int:
    return await db.events.count_documents(
        {"session_id": session_id, "assigned_player_id": {"$ne": None}}
    )


def _doc_to_game_out(doc: dict, players: int, total_actions: int) -> GameOut:
    gm: dict       = doc.get("game", {})
    uploaded: datetime = doc.get("uploaded_at", datetime.now(timezone.utc))
    return GameOut(
        id           = str(doc["_id"]),
        homeTeam     = gm.get("home_team", "קבוצה בית"),
        awayTeam     = gm.get("away_team", "קבוצה אורחת"),
        homeScore    = gm.get("home_score", 0),
        awayScore    = gm.get("away_score", 0),
        date         = gm.get("date") or uploaded.strftime("%d.%m.%Y"),
        time         = gm.get("time") or uploaded.strftime("%H:%M"),
        players      = players,
        totalActions = total_actions,
    )


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Soccer Event-Player Linker API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global client
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[MONGODB_DB]
    await db.sessions.create_index("uploaded_at")
    await db.events.create_index("session_id")
    await db.events.create_index([("session_id", 1), ("event_frame", 1)])
    await db.events.create_index([("session_id", 1), ("assigned_player_id", 1)])
    await db.events.create_index([("session_id", 1), ("event_type", 1)])
    await db.events.create_index(
        [("session_id", 1), ("assigned_player_id", 1), ("event_type", 1)]
    )


@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions", status_code=status.HTTP_201_CREATED)
async def upload_session(payload: AnalysisSessionIn):
    db = await get_db()
    session_doc = {
        "video":        payload.video,
        "fps":          payload.fps,
        "num_events":   payload.num_events,
        "num_assigned": payload.num_assigned,
        "parameters":   payload.parameters.model_dump(),
        "game":         payload.game.model_dump(),
        "uploaded_at":  datetime.now(timezone.utc),
    }
    res        = await db.sessions.insert_one(session_doc)
    session_id = str(res.inserted_id)

    if payload.results:
        event_docs = []
        for ev in payload.results:
            d = ev.model_dump()
            d["session_id"] = session_id
            event_docs.append(d)
        await db.events.insert_many(event_docs)

    return {"session_id": session_id, "events_stored": len(payload.results)}


@app.get("/sessions", response_model=List[GameOut])
async def list_sessions(
    skip:  int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    db   = await get_db()
    docs = (
        await db.sessions.find()
        .sort("uploaded_at", -1)
        .skip(skip)
        .limit(limit)
        .to_list(limit)
    )
    result = []
    for doc in docs:
        sid     = str(doc["_id"])
        players = await _count_unique_players(db, sid)
        total   = await _count_total_assigned(db, sid)
        result.append(_doc_to_game_out(doc, players, total))
    return result


@app.get("/sessions/{session_id}", response_model=SessionDetailOut)
async def get_session(session_id: str):
    db  = await get_db()
    doc = await db.sessions.find_one({"_id": oid(session_id)})
    if not doc:
        raise HTTPException(404, "Session not found")

    sid     = str(doc["_id"])
    players = await _count_unique_players(db, sid)
    total   = await _count_total_assigned(db, sid)
    base    = _doc_to_game_out(doc, players, total)

    return SessionDetailOut(
        **base.model_dump(),
        video        = doc["video"],
        fps          = doc["fps"],
        num_events   = doc["num_events"],
        num_assigned = doc["num_assigned"],
        coverage_pct = round(doc["num_assigned"] / max(doc["num_events"], 1) * 100, 1),
        uploaded_at  = doc["uploaded_at"],
        parameters   = AnalysisParameters(**doc["parameters"]),
    )


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    db = await get_db()
    r  = await db.sessions.delete_one({"_id": oid(session_id)})
    if r.deleted_count == 0:
        raise HTTPException(404, "Session not found")
    await db.events.delete_many({"session_id": session_id})


@app.get("/sessions/{session_id}/player-stats")
async def player_stats(session_id: str):
    """
    Returns per-player action counts:
      [{ "id": 18, "PASS": 3, "DRIVE": 7, "GOAL": 0, ... }, ...]
    All 12 action types always present (0 if none).
    """
    db = await get_db()
    if not await db.sessions.find_one({"_id": oid(session_id)}, {"_id": 1}):
        raise HTTPException(404, "Session not found")

    ALL_ACTIONS = [
        "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS",
        "THROW IN", "SHOT", "BALL PLAYER BLOCK",
        "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL",
    ]

    pipeline = [
        {"$match": {"session_id": session_id, "assigned_player_id": {"$ne": None}}},
        {"$group": {
            "_id":   {"player_id": "$assigned_player_id", "event_type": "$event_type"},
            "count": {"$sum": 1},
        }},
    ]
    rows = await db.events.aggregate(pipeline).to_list(50000)

    player_map: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {a: 0 for a in ALL_ACTIONS}
    )
    for row in rows:
        pid   = row["_id"]["player_id"]
        etype = row["_id"]["event_type"]
        count = row["count"]
        if etype in player_map[pid]:
            player_map[pid][etype] = count

    return [{"id": pid, **counts} for pid, counts in sorted(player_map.items())]


@app.get("/sessions/{session_id}/events")
async def list_events(
    session_id:     str,
    event_type:     Optional[str]   = None,
    player_id:      Optional[int]   = None,
    assigned_only:  bool            = False,
    min_confidence: float           = Query(0.0, ge=0.0, le=1.0),
    skip:           int             = Query(0, ge=0),
    limit:          int             = Query(200, ge=1, le=1000),
):
    db = await get_db()
    if not await db.sessions.find_one({"_id": oid(session_id)}, {"_id": 1}):
        raise HTTPException(404, "Session not found")

    filt: Dict[str, Any] = {
        "session_id":       session_id,
        "event_confidence": {"$gte": min_confidence},
    }
    if event_type:
        filt["event_type"] = event_type.upper()
    if player_id is not None:
        filt["assigned_player_id"] = player_id
    if assigned_only:
        filt["assigned_player_id"] = {"$ne": None}

    docs = (
        await db.events.find(filt, {"_id": 0})
        .sort("event_frame", 1)
        .skip(skip)
        .limit(limit)
        .to_list(limit)
    )
    return docs


@app.get("/sessions/{session_id}/stats")
async def session_stats(session_id: str):
    """Summary card numbers for the game detail page."""
    db = await get_db()
    if not await db.sessions.find_one({"_id": oid(session_id)}, {"_id": 1}):
        raise HTTPException(404, "Session not found")

    pipeline = [
        {"$match": {"session_id": session_id, "assigned_player_id": {"$ne": None}}},
        {"$group": {
            "_id":          None,
            "totalActions": {"$sum": 1},
            "totalGoals":   {"$sum": {"$cond": [{"$eq": ["$event_type", "GOAL"]}, 1, 0]}},
            "totalPasses":  {"$sum": {"$cond": [{"$eq": ["$event_type", "PASS"]}, 1, 0]}},
        }},
    ]
    r = await db.events.aggregate(pipeline).to_list(1)
    if not r:
        return {"totalActions": 0, "totalGoals": 0, "totalPasses": 0}
    doc = r[0]
    doc.pop("_id", None)
    return doc
