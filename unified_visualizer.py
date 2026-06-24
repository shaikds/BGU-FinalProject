#!/usr/bin/env python3
"""
Unified visualizer:  PLAYERS (ReID ellipses)  +  EVENTS (broadcast panel)  +
optional "WHO DID THE EVENT" actor highlight.

This merges the two existing scripts WITHOUT changing their visual style:

  * Players / ball  ->  static open ellipse at the feet + "TID:x GID:y" label
                        and a green ellipse for the ball   (from the ReID overlay)

  * Events          ->  bar-chart-race panel (top-right), toast cards (top-left),
                        big-event flash, bottom timeline   (from the event overlay)

  * Who did it      ->  OPTIONAL.  When --show-actor is set, the player assigned to
                        each event is highlighted for a short window with a pulsing
                        ring (in the event's colour) and an event badge above them.
                        This is drawn in the same PIL overlay so it matches the look.

------------------------------------------------------------------------------
INPUTS
------------------------------------------------------------------------------
Player boxes (one of, auto-detected):
  reid_observations.json : {"observations":[{frame_index,track_id,global_id,
                                              bbox_xyxy,label}], "ball_detections":[...]}
  tracks.json            : {"tracks":[{frame_index,track_id,bbox_xyxy,label}]}
                           (+ optional --mapping trackid_to_globalid.json)

Events (one of, auto-detected):
  assignment JSON  : {"results":[{event_frame,event_type,assigned_track_id,
                                   assigned_player_id,event_confidence,...}]}  (actor available)
  predictions JSON : {"predictions":[{frame,label}]}                          (no actor)

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
  python3 unified_visualizer.py \
      --video   input_video.mp4 \
      --players outputs/reid_v2/reid_observations.json \
      --events  event_player_assignment.json \
      --output  out.mp4 \
      --show-actor                # <- omit to NOT draw the actor highlight

      REAL EXAMPLE:
      (soccernet_gs_env) [shaikar@ise-4090-21 ~]$ conda activate tdeed_inference2
(tdeed_inference2) (soccernet_gs_env) [shaikar@ise-4090-21 ~]$ python unified_visualizer.py -v /home/shaikar/PIPELINE-PROJECT-VID1.mp4 -p outputs/reid_v2/reid_observations.json -e linked.json -o out.mp4
Players not found: outputs/reid_v2/reid_observations.json
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


# =========================================================================== #
#  Small helpers (shared)
# =========================================================================== #
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_cubic(t):
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def ease_out_back(t):
    t = clamp(t, 0.0, 1.0)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def format_time(frame_idx, fps):
    seconds = frame_idx / float(fps)
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m:02d}:{s:05.2f}"
    return f"{seconds:.2f}s"


# A vibrant, harmonious palette (RGB), assigned deterministically to event types.
PALETTE = [
    (56, 189, 248),    # sky
    (52, 211, 153),    # emerald
    (251, 146, 60),    # orange
    (244, 114, 182),   # pink
    (167, 139, 250),   # violet
    (248, 113, 113),   # red
    (45, 212, 191),    # teal
    (129, 140, 248),   # indigo
    (163, 230, 53),    # lime
    (232, 121, 249),   # fuchsia
    (96, 165, 250),    # blue
    (250, 204, 21),    # amber
]
COLOR_OVERRIDES = {
    'GOAL': (250, 204, 21),
    'SHOT': (248, 113, 113),
}
BIG_EVENTS = {'GOAL'}             # gets the full-screen celebratory flash

# Player / ball ellipse geometry (from the ReID overlay script)
PLAYER_ELLIPSE_W_RATIO = 0.50      # semi-axis x as a fraction of bbox width
PLAYER_ELLIPSE_H_TO_W = 0.31       # semi-axis y as a fraction of semi-axis x
PLAYER_ELLIPSE_W_MIN = 18
PLAYER_ELLIPSE_W_MAX = 90
PLAYER_ELLIPSE_H_MIN = 6
PLAYER_ELLIPSE_H_MAX = 28

BALL_ELLIPSE_W_RATIO = 0.70
BALL_ELLIPSE_H_TO_W = 0.33
BALL_ELLIPSE_W_MIN = 10
BALL_ELLIPSE_W_MAX = 50
BALL_ELLIPSE_H_MIN = 4
BALL_ELLIPSE_H_MAX = 16

ELLIPSE_THICKNESS_DIVISOR = 16
ELLIPSE_THICKNESS_MIN = 2
ELLIPSE_THICKNESS_MAX = 6

BALL_LABELS = {0}
BALL_COLOR = (0, 255, 0)          # BGR green


def ellipse_dims_from_bbox(bbox_w, w_ratio, h_to_w, w_min, w_max, h_min, h_max):
    ew = int(np.clip(bbox_w * w_ratio, w_min, w_max))
    eh = int(np.clip(ew * h_to_w, h_min, h_max))
    eth = int(np.clip(round(ew / ELLIPSE_THICKNESS_DIVISOR),
                       ELLIPSE_THICKNESS_MIN, ELLIPSE_THICKNESS_MAX))
    return ew, eh, eth


# =========================================================================== #
#  Loading: player boxes
# =========================================================================== #
def load_players(players_path, mapping_path=None):
    """
    Returns:
      frame_to_obs : dict[int] -> list of {track_id, global_id, bbox, label}
      gid_set      : set of all global ids (for stable colour assignment)
    Handles both reid_observations.json and tracks.json formats.
    """
    with open(players_path, "r") as f:
        data = json.load(f)

    trackid_to_globalid = {}
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            m = json.load(f)
        raw = m.get("trackid_to_globalid", m)
        trackid_to_globalid = {int(k): int(v) for k, v in raw.items()}

    frame_to_obs = defaultdict(list)
    gid_set = set()

    if "observations" in data:
        # ReID output: global_id already baked in.
        for o in data["observations"]:
            tid = int(o["track_id"])
            gid = o.get("global_id", None)
            gid = int(gid) if gid is not None else None
            lab = int(o.get("label", -1))
            frame_to_obs[int(o["frame_index"])].append({
                "track_id": tid, "global_id": gid,
                "bbox": [float(v) for v in o["bbox_xyxy"]], "label": lab,
            })
            if gid is not None:
                gid_set.add(gid)
        # Ball detections live in a side-car list (label 0).
        for b in data.get("ball_detections", []) or []:
            frame_to_obs[int(b["frame_index"])].append({
                "track_id": int(b.get("track_id", -1)), "global_id": None,
                "bbox": [float(v) for v in b["bbox_xyxy"]], "label": 0,
            })

    elif "tracks" in data:
        # Raw tracking output: derive global id from mapping (fallback = track id).
        for t in data["tracks"]:
            tid = int(t["track_id"])
            lab = int(t.get("label", -1))
            if lab in BALL_LABELS:
                gid = None
            else:
                gid = trackid_to_globalid.get(tid, tid)
                gid_set.add(gid)
            frame_to_obs[int(t["frame_index"])].append({
                "track_id": tid, "global_id": gid,
                "bbox": [float(v) for v in t["bbox_xyxy"]], "label": lab,
            })
    else:
        raise ValueError("Players JSON must contain 'observations' or 'tracks'.")

    print(f"Loaded player observations across {len(frame_to_obs)} frames "
          f"| {len(gid_set)} global ids")
    return frame_to_obs, gid_set


# =========================================================================== #
#  Loading: events (+ optional actor)
# =========================================================================== #
def load_events(events_path, remove_drive=True, map_tackle=True, min_conf=0.0):
    """
    Returns:
      frame_events : dict[int] -> list of cleaned label strings (drives the panel)
      counts       : Counter of all cleaned labels
      events_list  : list of {frame, label, track_id, global_id} for actor highlight
    Handles both the assignment 'results' format and the legacy 'predictions' format.
    """
    with open(events_path, "r") as f:
        data = json.load(f)

    frame_events = defaultdict(list)
    counts = Counter()
    events_list = []

    if "results" in data:
        rows = data["results"]
        for r in rows:
            label = str(r.get("event_type", "")).strip()
            if not label:
                continue
            if float(r.get("event_confidence", 1.0)) < min_conf:
                continue
            up = label.upper()
            if map_tackle and "TACKLE" in up:
                label, up = "BALL PLAYER BLOCK", "BALL PLAYER BLOCK"
            if remove_drive and up == "DRIVE":
                continue
            frame = int(r.get("event_frame", 0))
            tid = r.get("assigned_track_id", None)
            gid = r.get("assigned_player_id", None)
            frame_events[frame].append(label)
            counts[label] += 1
            events_list.append({
                "frame": frame, "label": label,
                "track_id": int(tid) if tid is not None else None,
                "global_id": int(gid) if gid is not None else None,
            })

    elif "predictions" in data:
        for p in data["predictions"]:
            label = p.get("label", "")
            if not isinstance(label, str):
                continue
            label = label.strip()
            up = label.upper()
            if map_tackle and "TACKLE" in up:
                label, up = "BALL PLAYER BLOCK", "BALL PLAYER BLOCK"
            if remove_drive and up == "DRIVE":
                continue
            frame = int(p.get("frame", 0))
            frame_events[frame].append(label)
            counts[label] += 1
            events_list.append({"frame": frame, "label": label,
                                "track_id": None, "global_id": None})
    else:
        raise ValueError("Events JSON must contain 'results' or 'predictions'.")

    n_actor = sum(1 for e in events_list if e["track_id"] is not None
                  or e["global_id"] is not None)
    print(f"Loaded {sum(counts.values())} events ({len(counts)} types) "
          f"| {n_actor} with an assigned actor")
    return frame_events, counts, events_list


# =========================================================================== #
#  Player / ball drawing  (faithful port of the ReID overlay style)
# =========================================================================== #
def draw_players(frame, observations, gid_colors, w, h, label_mode="gid"):
    for o in observations:
        tid = o["track_id"]
        gid = o["global_id"]
        label = o["label"]

        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        bbox_w = max(1, x2 - x1)
        if label in BALL_LABELS:
            color = BALL_COLOR
            label_txt = "BALL"
            ew, eh, eth = ellipse_dims_from_bbox(
                bbox_w,
                BALL_ELLIPSE_W_RATIO,
                BALL_ELLIPSE_H_TO_W,
                BALL_ELLIPSE_W_MIN,
                BALL_ELLIPSE_W_MAX,
                BALL_ELLIPSE_H_MIN,
                BALL_ELLIPSE_H_MAX,
            )
        else:
            if gid is None:
                continue
            color = gid_colors.get(gid, (200, 200, 200))
            if label_mode == "gid":
                label_txt = f"GID:{gid}"
            elif label_mode == "tid":
                label_txt = f"TID:{tid}"
            elif label_mode == "none":
                label_txt = ""
            else:
                label_txt = f"TID:{tid} GID:{gid}"
            ew, eh, eth = ellipse_dims_from_bbox(
                bbox_w,
                PLAYER_ELLIPSE_W_RATIO,
                PLAYER_ELLIPSE_H_TO_W,
                PLAYER_ELLIPSE_W_MIN,
                PLAYER_ELLIPSE_W_MAX,
                PLAYER_ELLIPSE_H_MIN,
                PLAYER_ELLIPSE_H_MAX,
            )

        cx = int((x1 + x2) / 2)
        cy = int(y2)

        cv2.ellipse(frame, (cx, cy), (ew, eh), 0, 15, 345,
                    color, eth, cv2.LINE_AA)

        if not label_txt:
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_txt, font, font_scale, thickness)

        text_x = max(0, min(w - tw - 6, cx - tw // 2))
        text_y = min(h - 4, cy + eh + th + 10)
        box_y1 = max(0, text_y - th - 4)
        box_y2 = min(h, text_y + 4)

        cv2.rectangle(frame, (text_x, box_y1), (text_x + tw + 4, box_y2), color, -1)
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(frame, label_txt, (text_x + 2, text_y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
    return frame


# =========================================================================== #
#  EventViz  (faithful port of the broadcast overlay, extended with actor layer)
# =========================================================================== #
class EventViz:
    PULSE_SEC = 0.65              # count-pop duration
    TOAST_SEC = 1.7              # toast lifetime
    FLASH_SEC = 1.0              # big-event flash lifetime
    ACTOR_HOLD_SEC = 1.4         # how long the "who did it" highlight stays up

    def __init__(self, width, height, fps, counts_total, total_frames,
                 max_event_frame, title="MATCH EVENTS", use_pil=True):
        self.w, self.h, self.fps = width, height, fps
        self.title = title
        self.total_frames = total_frames if total_frames and total_frames > 0 \
            else int(max_event_frame * 1.05) + 1

        self.labels = sorted(counts_total.keys(),
                             key=lambda l: (-counts_total[l], l))
        self.totals = dict(counts_total)
        self.colors = {}
        for i, lab in enumerate(self.labels):
            self.colors[lab] = COLOR_OVERRIDES.get(lab.upper(),
                                                    PALETTE[i % len(PALETTE)])

        self.running = {l: 0 for l in self.labels}
        self.pulse_at = {l: -1e9 for l in self.labels}
        self.bar_cur = {l: 0.0 for l in self.labels}
        self.toasts = []
        self.flashes = []

        # ---- geometry: small panel, anchored TOP-RIGHT ------------------- #
        s = clamp(height / 720.0, 0.55, 1.6)
        self.s = s
        self.margin = int(16 * s)
        self.pad = int(8 * s)
        self.title_h = int(28 * s)
        self.row_h = int(22 * s)
        self.radius = int(8 * s)
        self.panel_w = int(clamp(width * 0.24, 180, int(340 * s)))
        self.panel_x = self.w - self.panel_w - self.margin
        self.panel_y = self.margin
        n = max(1, len(self.labels))
        self.panel_h = self.title_h + n * self.row_h + self.pad

        order = self._order()
        self.y_cur = {}
        content_top = self.panel_y + self.title_h + int(3 * s)
        for idx, lab in enumerate(order):
            self.y_cur[lab] = content_top + idx * self.row_h

        # fonts
        self.use_pil = bool(HAVE_PIL and use_pil)
        self._font_cache = {}
        self._bold_path = self._find_font(
            ["Poppins-Bold.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf",
             "arialbd.ttf"])
        self._med_path = self._find_font(
            ["Poppins-Medium.ttf", "Poppins-Regular.ttf", "DejaVuSans.ttf",
             "Arial.ttf", "arial.ttf"])
        if self.use_pil and (self._bold_path is None or self._med_path is None):
            self.use_pil = False

    # ---- font handling ---------------------------------------------------- #
    _FONT_DIRS = [
        "/usr/share/fonts/truetype/google-fonts",
        "/usr/share/fonts/truetype/dejavu",
        "/Library/Fonts", "/System/Library/Fonts/Supplemental",
        "C:/Windows/Fonts",
    ]

    def _find_font(self, names):
        for name in names:
            if os.path.isabs(name) and os.path.exists(name):
                return name
            for d in self._FONT_DIRS:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return p
        return None

    def _font(self, size, bold=True):
        size = max(7, int(size))
        key = (bold, size)
        if key not in self._font_cache:
            path = self._bold_path if bold else self._med_path
            self._font_cache[key] = ImageFont.truetype(path, size)
        return self._font_cache[key]

    # ---- ordering --------------------------------------------------------- #
    def _order(self):
        return sorted(
            self.labels,
            key=lambda l: (-self.running[l], -self.totals.get(l, 0), l))

    # ---- per-frame state update ------------------------------------------ #
    def update(self, frame_idx, labels_now):
        for lab in labels_now:
            if lab in self.running:
                self.running[lab] += 1
                self.pulse_at[lab] = frame_idx
                self.toasts.append({'label': lab, 'start': frame_idx})
                if lab.upper() in BIG_EVENTS:
                    self.flashes.append({'label': lab, 'start': frame_idx})

        self.toasts = [t for t in self.toasts
                       if (frame_idx - t['start']) / self.fps <= self.TOAST_SEC]
        self.flashes = [f for f in self.flashes
                        if (frame_idx - f['start']) / self.fps <= self.FLASH_SEC]

        order = self._order()
        content_top = self.panel_y + self.title_h + int(3 * self.s)
        for idx, lab in enumerate(order):
            target = content_top + idx * self.row_h
            self.y_cur[lab] = lerp(self.y_cur[lab], target, 0.22)

        max_count = max(1, max(self.running.values()))
        cnt_col_w = int(28 * self.s)
        bar_max = self.panel_w - 2 * self.pad - cnt_col_w
        for lab in self.labels:
            target = (self.running[lab] / max_count) * bar_max
            self.bar_cur[lab] = lerp(self.bar_cur[lab], target, 0.25)

    # ===================================================================== #
    #  Dispatch
    # ===================================================================== #
    def render(self, frame_bgr, frame_idx, actor_highlights=None):
        if self.use_pil:
            return self._render_pil(frame_bgr, frame_idx, actor_highlights)
        return self._render_cv2(frame_bgr, frame_idx, actor_highlights)

    # ===================================================================== #
    #  PIL rendering (the pretty path)
    # ===================================================================== #
    @staticmethod
    def _rrect(draw, box, r, fill):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        r = int(max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2)))
        if hasattr(draw, "rounded_rectangle") and r > 0:
            draw.rounded_rectangle(box, radius=r, fill=fill)
        else:
            draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
            draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
            if r > 0:
                draw.pieslice([x0, y0, x0 + 2 * r, y0 + 2 * r], 180, 270, fill=fill)
                draw.pieslice([x1 - 2 * r, y0, x1, y0 + 2 * r], 270, 360, fill=fill)
                draw.pieslice([x0, y1 - 2 * r, x0 + 2 * r, y1], 90, 180, fill=fill)
                draw.pieslice([x1 - 2 * r, y1 - 2 * r, x1, y1], 0, 90, fill=fill)

    @staticmethod
    def _tsize(font, s):
        try:
            l, t, r, b = font.getbbox(s)
            return r - l, b - t
        except Exception:
            return font.getsize(s)

    def _text(self, draw, xy, s, font, fill, anchor_x='l', shadow=True):
        tw, th = self._tsize(font, s)
        x, y = xy
        if anchor_x == 'r':
            x -= tw
        elif anchor_x == 'c':
            x -= tw // 2
        if shadow:
            draw.text((x + max(1, int(1 * self.s)), y + max(1, int(1 * self.s))),
                      s, font=font, fill=(0, 0, 0, 120))
        draw.text((x, y), s, font=font, fill=fill)
        return tw, th

    def _render_pil(self, frame_bgr, frame_idx, actor_highlights=None):
        base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        s = self.s

        # ---- actor highlight first (sits on the field, under the panel) -- #
        self._draw_actor_highlights_pil(d, frame_idx, actor_highlights)

        px, py, pw, ph = self.panel_x, self.panel_y, self.panel_w, self.panel_h

        # ---- panel shadow + body ----------------------------------------- #
        self._rrect(d, (px + int(3 * s), py + int(4 * s),
                        px + pw + int(3 * s), py + ph + int(4 * s)),
                    self.radius, (0, 0, 0, 60))
        self._rrect(d, (px, py, px + pw, py + ph), self.radius, (16, 18, 26, 165))

        # ---- title bar ---------------------------------------------------- #
        self._rrect(d, (px, py, px + pw, py + self.title_h),
                    self.radius, (30, 34, 48, 195))
        d.rectangle([px, py + self.title_h - self.radius,
                     px + pw, py + self.title_h], fill=(30, 34, 48, 195))

        title_font = self._font(14 * s, bold=True)
        self._text(d,
                   (px + self.pad,
                    py + self.title_h // 2 - self._tsize(title_font, self.title)[1] // 2 - int(1 * s)),
                   self.title, title_font, (236, 240, 248, 255))

        clk_font = self._font(10 * s, bold=True)
        clk = format_time(frame_idx, self.fps)
        blink = 0.55 + 0.45 * math.sin(frame_idx / self.fps * 5.0)
        dot_r = int(4 * s)
        cx = px + pw - self.pad - int(3 * s)
        cy = py + self.title_h // 2
        ctw, cth = self._tsize(clk_font, clk)
        self._text(d, (cx, cy - cth // 2 - int(1 * s)), clk, clk_font,
                   (200, 206, 218, 255), anchor_x='r')
        d.ellipse([cx - ctw - int(12 * s) - dot_r, cy - dot_r,
                   cx - ctw - int(12 * s) + dot_r, cy + dot_r],
                  fill=(248, 80, 80, int(120 + 135 * blink)))

        # ---- rows (bar-chart race) ---------------------------------------- #
        lab_font = self._font(11 * s, bold=True)
        bar_x0 = px + self.pad
        cnt_col_w = int(28 * s)
        bar_max = pw - 2 * self.pad - cnt_col_w
        bar_h = int(self.row_h * 0.52)

        for lab in self.labels:
            y = int(self.y_cur[lab])
            cy = y + (self.row_h - bar_h) // 2
            col = self.colors[lab]
            cnt = self.running[lab]

            age_p = (frame_idx - self.pulse_at[lab]) / self.fps
            p = 0.0
            if 0 <= age_p <= self.PULSE_SEC:
                p = 1.0 - ease_out_cubic(age_p / self.PULSE_SEC)

            self._rrect(d, (bar_x0, cy, bar_x0 + bar_max, cy + bar_h),
                        bar_h // 2, (255, 255, 255, 22))

            if p > 0.01:
                g = int(100 * p)
                self._rrect(d, (bar_x0 - int(2 * s), cy - int(2 * s),
                                bar_x0 + max(bar_h, self.bar_cur[lab]) + int(2 * s),
                                cy + bar_h + int(2 * s)),
                            bar_h // 2 + int(2 * s), col + (g,))

            fill_w = max(bar_h if cnt > 0 else 0, self.bar_cur[lab])
            if fill_w > 0:
                a = 235 if cnt > 0 else 60
                self._rrect(d, (bar_x0, cy, bar_x0 + fill_w, cy + bar_h),
                            bar_h // 2, col + (a,))

            txt_col = (255, 255, 255, 255) if cnt > 0 else (150, 156, 170, 255)
            self._text(d,
                       (bar_x0 + int(6 * s),
                        cy + bar_h // 2 - self._tsize(lab_font, lab)[1] // 2 - int(1 * s)),
                       lab, lab_font, txt_col)

            cnt_size = (11.5 * s) * (1.0 + 0.35 * p)
            cnt_font = self._font(cnt_size, bold=True)
            cnt_col_clr = tuple(int(clamp(c + 60 * p, 0, 255)) for c in col) + (255,) \
                if cnt > 0 else (150, 156, 170, 255)
            self._text(d,
                       (px + pw - self.pad,
                        cy + bar_h // 2 - self._tsize(cnt_font, str(cnt))[1] // 2 - int(1 * s)),
                       str(cnt), cnt_font, cnt_col_clr, anchor_x='r')

        self._draw_toasts_pil(d, frame_idx)
        self._draw_flash_pil(d, frame_idx)
        self._draw_timeline_pil(d, frame_idx)

        out = Image.alpha_composite(base, ov).convert("RGB")
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    # ---- NEW: actor highlight (PIL) -------------------------------------- #
    def _draw_actor_highlights_pil(self, d, frame_idx, highlights):
        if not highlights:
            return
        s = self.s
        for hgl in highlights:
            label = hgl['label']
            start = hgl['start']
            age = (frame_idx - start) / self.fps
            if age < 0 or age > self.ACTOR_HOLD_SEC:
                continue
            env = min(clamp(age / 0.12, 0, 1),
                      clamp((self.ACTOR_HOLD_SEC - age) / 0.4, 0, 1))
            if env <= 0.01:
                continue
            col = self.colors.get(label, (250, 204, 21))
            x1, y1, x2, y2 = hgl['bbox']
            cx = (x1 + x2) / 2.0
            cy = float(y2)                      # feet
            ew = max(30 * s, (x2 - x1) * 0.72)
            eh = max(12 * s, ew * 0.34)

            # expanding pulse ring
            cyc = (age % 0.9) / 0.9
            grow = 1.0 + 0.7 * cyc
            pa = int(150 * env * (1.0 - cyc))
            if pa > 2:
                d.ellipse([cx - ew * grow, cy - eh * grow,
                           cx + ew * grow, cy + eh * grow],
                          outline=col + (pa,), width=max(2, int(3 * s)))

            # solid base ring (closed -> clearly emphasised vs the player ellipse)
            d.ellipse([cx - ew, cy - eh, cx + ew, cy + eh],
                      outline=col + (int(235 * env),), width=max(3, int(5 * s)))

            # event badge above the player's head
            bf = self._font(13 * s, bold=True)
            tw, th = self._tsize(bf, label)
            pad = int(7 * s)
            bw = tw + 2 * pad + int(5 * s)
            bh = th + 2 * pad
            bx0 = clamp(cx - bw / 2, 2, self.w - bw - 2)
            by1 = y1 - int(8 * s)
            by0 = by1 - bh
            if by0 < 2:                          # not enough room above -> below feet
                by0 = y2 + int(8 * s)
                by1 = by0 + bh
            self._rrect(d, (bx0, by0, bx0 + bw, by1), int(8 * s),
                        (12, 14, 22, int(225 * env)))
            self._rrect(d, (bx0, by0, bx0 + int(5 * s), by1), int(3 * s),
                        col + (int(235 * env),))
            self._text(d, (bx0 + int(5 * s) + pad, by0 + pad - int(1 * s)),
                       label, bf, (240, 244, 252, int(255 * env)), shadow=False)

    def _draw_toasts_pil(self, d, frame_idx):
        s = self.s
        card_w = int(clamp(self.w * 0.22, 160, 280))
        card_h = int(38 * s)
        gap = int(8 * s)
        left = self.margin
        top = self.margin
        active = sorted(self.toasts, key=lambda t: -t['start'])
        for i, t in enumerate(active):
            age = (frame_idx - t['start']) / self.fps
            fin = clamp(age / 0.18, 0, 1)
            fout = clamp((self.TOAST_SEC - age) / 0.45, 0, 1)
            a = min(fin, fout)
            if a <= 0.01:
                continue
            slide = int((1 - ease_out_cubic(fin)) * 55 * s)
            x0 = left - slide
            x1 = x0 + card_w
            y0 = top + i * (card_h + gap)
            y1 = y0 + card_h
            col = self.colors.get(t['label'], (200, 200, 200))
            A = int(235 * a)
            self._rrect(d, (x0, y0, x1, y1), int(8 * s), (18, 20, 30, int(215 * a)))
            self._rrect(d, (x0, y0, x0 + int(5 * s), y1), int(3 * s), col + (A,))
            dot_r = int(5 * s)
            dcx, dcy = x0 + int(18 * s), (y0 + y1) // 2
            d.ellipse([dcx - dot_r, dcy - dot_r, dcx + dot_r, dcy + dot_r],
                      fill=col + (A,))
            lf = self._font(13 * s, bold=True)
            tf = self._font(10 * s, bold=False)
            lh = self._tsize(lf, t['label'])[1]
            self._text(d, (x0 + int(30 * s), dcy - lh + int(2 * s)),
                       t['label'], lf, (240, 244, 252, A))
            self._text(d, (x0 + int(30 * s), dcy + int(2 * s)),
                       format_time(t['start'], self.fps), tf, (160, 166, 180, A))

    def _draw_flash_pil(self, d, frame_idx):
        if not self.flashes:
            return
        s = self.s
        f = self.flashes[-1]
        age = (frame_idx - f['start']) / self.fps
        prog = clamp(age / self.FLASH_SEC, 0, 1)
        a = (ease_out_cubic(clamp(age / 0.12, 0, 1)) *
             clamp((self.FLASH_SEC - age) / 0.4, 0, 1))
        if a <= 0.01:
            return
        col = self.colors.get(f['label'], (250, 204, 21))
        text = (f['label'].upper() + "!") if f['label'].upper() in BIG_EVENTS else f['label'].upper()
        size = (90 * s) * (0.7 + 0.35 * ease_out_back(clamp(age / 0.35, 0, 1)))
        font = self._font(size, bold=True)
        tw, th = self._tsize(font, text)
        cx, cy = self.w // 2, int(self.h * 0.42 - prog * 40 * s)
        for k, ga in ((int(26 * s), 50), (int(14 * s), 70)):
            d.rounded_rectangle(
                [cx - tw // 2 - k, cy - th // 2 - k, cx + tw // 2 + k, cy + th // 2 + k],
                radius=int(20 * s), fill=(col[0], col[1], col[2], int(ga * a)))
        d.text((cx - tw // 2 + int(3 * s), cy - th // 2 + int(3 * s)),
               text, font=font, fill=(0, 0, 0, int(160 * a)))
        d.text((cx - tw // 2, cy - th // 2), text, font=font,
               fill=col + (int(255 * a),))

    def _draw_timeline_pil(self, d, frame_idx):
        s = self.s
        x0 = self.margin
        x1 = self.w - self.margin
        y = self.h - self.margin
        h = int(5 * s)
        self._rrect(d, (x0, y - h // 2, x1, y + h // 2), h // 2, (255, 255, 255, 40))
        frac = clamp(frame_idx / max(1, self.total_frames), 0, 1)
        px = int(x0 + frac * (x1 - x0))
        self._rrect(d, (x0, y - h // 2, px, y + h // 2), h // 2, (120, 200, 255, 220))
        ph = int(7 * s)
        d.ellipse([px - ph, y - ph, px + ph, y + ph], fill=(235, 242, 252, 255))
        d.ellipse([px - ph // 2, y - ph // 2, px + ph // 2, y + ph // 2],
                  fill=(90, 170, 250, 255))

    # ===================================================================== #
    #  OpenCV fallback (no Pillow / no fonts)
    # ===================================================================== #
    def _render_cv2(self, frame, frame_idx, actor_highlights=None):
        s = self.s
        font = cv2.FONT_HERSHEY_SIMPLEX

        self._draw_actor_highlights_cv2(frame, frame_idx, actor_highlights)

        px, py, pw, ph = self.panel_x, self.panel_y, self.panel_w, self.panel_h
        ov = frame.copy()
        cv2.rectangle(ov, (px, py), (px + pw, py + ph), (26, 18, 16), -1)
        cv2.rectangle(ov, (px, py), (px + pw, py + self.title_h), (48, 34, 30), -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

        cv2.putText(frame, self.title,
                    (px + self.pad, py + int(self.title_h * 0.70)),
                    font, 0.5 * s, (248, 240, 236), max(1, int(1.5 * s)), cv2.LINE_AA)
        clk = format_time(frame_idx, self.fps)
        (cw, _), _ = cv2.getTextSize(clk, font, 0.42 * s, 1)
        cv2.putText(frame, clk,
                    (px + pw - self.pad - cw, py + int(self.title_h * 0.70)),
                    font, 0.42 * s, (218, 206, 200), 1, cv2.LINE_AA)

        bar_x0 = px + self.pad
        cnt_col_w = int(28 * s)
        bar_max = pw - 2 * self.pad - cnt_col_w
        bar_h = int(self.row_h * 0.52)
        for lab in self.labels:
            y = int(self.y_cur[lab])
            cy = y + (self.row_h - bar_h) // 2
            r, g, b = self.colors[lab]
            bgr = (b, g, r)
            cnt = self.running[lab]
            cv2.rectangle(frame, (bar_x0, cy), (bar_x0 + bar_max, cy + bar_h),
                          (60, 60, 60), -1)
            fw = int(max(bar_h if cnt > 0 else 0, self.bar_cur[lab]))
            if fw > 0:
                cv2.rectangle(frame, (bar_x0, cy), (bar_x0 + fw, cy + bar_h), bgr, -1)
            tcol = (255, 255, 255) if cnt > 0 else (150, 156, 170)
            cv2.putText(frame, lab,
                        (bar_x0 + int(6 * s), cy + int(bar_h * 0.72)),
                        font, 0.38 * s, tcol, max(1, int(1.2 * s)), cv2.LINE_AA)
            ctxt = str(cnt)
            (ctw, _), _ = cv2.getTextSize(ctxt, font, 0.46 * s, 2)
            cv2.putText(frame, ctxt,
                        (px + pw - self.pad - ctw, cy + int(bar_h * 0.74)),
                        font, 0.46 * s, (255, 255, 255),
                        max(1, int(1.5 * s)), cv2.LINE_AA)

        card_w = int(clamp(self.w * 0.22, 155, 260))
        card_h = int(34 * s)
        for i, t in enumerate(sorted(self.toasts, key=lambda t: -t['start'])):
            age = (frame_idx - t['start']) / self.fps
            if age > self.TOAST_SEC:
                continue
            x0 = self.margin
            x1 = x0 + card_w
            y0 = self.margin + i * (card_h + int(7 * s))
            r, g, b = self.colors.get(t['label'], (200, 200, 200))
            o = frame.copy()
            cv2.rectangle(o, (x0, y0), (x1, y0 + card_h), (28, 20, 18), -1)
            cv2.rectangle(o, (x0, y0), (x0 + int(5 * s), y0 + card_h), (b, g, r), -1)
            cv2.addWeighted(o, 0.82, frame, 0.18, 0, frame)
            cv2.putText(frame, t['label'],
                        (x0 + int(14 * s), y0 + int(card_h * 0.65)),
                        font, 0.38 * s, (240, 244, 252),
                        max(1, int(0.9 * s)), cv2.LINE_AA)

        x0, x1 = self.margin, self.w - self.margin
        yb = self.h - self.margin
        cv2.line(frame, (x0, yb), (x1, yb), (90, 90, 90), max(1, int(4 * s)))
        frac = clamp(frame_idx / max(1, self.total_frames), 0, 1)
        pxh = int(x0 + frac * (x1 - x0))
        cv2.line(frame, (x0, yb), (pxh, yb), (255, 200, 120), max(1, int(4 * s)))
        cv2.circle(frame, (pxh, yb), int(6 * s), (252, 242, 235), -1)
        return frame

    # ---- NEW: actor highlight (OpenCV fallback) -------------------------- #
    def _draw_actor_highlights_cv2(self, frame, frame_idx, highlights):
        if not highlights:
            return
        s = self.s
        font = cv2.FONT_HERSHEY_SIMPLEX
        for hgl in highlights:
            age = (frame_idx - hgl['start']) / self.fps
            if age < 0 or age > self.ACTOR_HOLD_SEC:
                continue
            env = min(clamp(age / 0.12, 0, 1),
                      clamp((self.ACTOR_HOLD_SEC - age) / 0.4, 0, 1))
            if env <= 0.01:
                continue
            r, g, b = self.colors.get(hgl['label'], (250, 204, 21))
            bgr = (int(b), int(g), int(r))
            x1, y1, x2, y2 = [int(v) for v in hgl['bbox']]
            cx, cy = (x1 + x2) // 2, y2
            ew = int(max(30 * s, (x2 - x1) * 0.72))
            eh = int(max(12 * s, ew * 0.34))
            cv2.ellipse(frame, (cx, cy), (ew, eh), 0, 0, 360, bgr,
                        max(2, int(4 * s)), cv2.LINE_AA)
            txt = hgl['label']
            (tw, th), _ = cv2.getTextSize(txt, font, 0.5 * s, max(1, int(1 * s)))
            bx = int(clamp(cx - tw // 2 - 6, 2, self.w - tw - 12))
            by = y1 - int(10 * s)
            if by - th - 8 < 2:
                by = y2 + th + int(16 * s)
            cv2.rectangle(frame, (bx, by - th - 8), (bx + tw + 10, by + 2),
                          (20, 16, 12), -1)
            cv2.rectangle(frame, (bx, by - th - 8), (bx + int(4 * s), by + 2),
                          bgr, -1)
            cv2.putText(frame, txt, (bx + int(7 * s), by - 4),
                        font, 0.5 * s, (240, 244, 252),
                        max(1, int(1 * s)), cv2.LINE_AA)


# =========================================================================== #
#  Actor resolution: which on-screen player performed each event, per frame
# =========================================================================== #
def resolve_actor_highlights(frame_idx, events_list, frame_to_obs, hold_frames):
    """
    For the current frame, return a list of {label, bbox, start} for every event
    whose hold window covers frame_idx and whose assigned player is visible now.
    Matches by local track_id first, then by global_id.
    """
    out = []
    obs = frame_to_obs.get(frame_idx, [])
    if not obs:
        return out
    by_tid = {o["track_id"]: o for o in obs}
    by_gid = {o["global_id"]: o for o in obs if o["global_id"] is not None}

    for e in events_list:
        start = e["frame"]
        if not (start <= frame_idx <= start + hold_frames):
            continue
        o = None
        if e["track_id"] is not None and e["track_id"] in by_tid:
            o = by_tid[e["track_id"]]
        elif e["global_id"] is not None and e["global_id"] in by_gid:
            o = by_gid[e["global_id"]]
        if o is None:
            continue
        out.append({"label": e["label"], "bbox": o["bbox"], "start": start})
    return out


# =========================================================================== #
#  Main loop
# =========================================================================== #
def visualize(video_path, players_path, events_path, out_path,
              mapping_path=None, show_actor=False,
              remove_drive=True, map_tackle=True, min_conf=0.0,
              fps_override=None, title="MATCH EVENTS", use_pil=True,
              player_label="tid_gid", max_frames=None, preview=False):

    frame_to_obs, gid_set = load_players(players_path, mapping_path)
    frame_events, counts, events_list = load_events(
        events_path, remove_drive=remove_drive, map_tackle=map_tackle, min_conf=min_conf)
    max_event_frame = max(frame_events.keys()) if frame_events else 0

    # Stable colour per global id (same seed as the ReID overlay -> same colours).
    np.random.seed(42)
    gid_colors = {gid: tuple(int(c) for c in np.random.randint(50, 255, size=3))
                  for gid in sorted(gid_set)}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    fps = float(fps_override) if fps_override else cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    viz = EventViz(width, height, fps, counts, total_frames, max_event_frame,
                   title=title, use_pil=use_pil)
    hold_frames = int(viz.ACTOR_HOLD_SEC * fps)

    mode = "Pillow (high quality)" if viz.use_pil else "OpenCV fallback"
    print(f"Writing visualization to {out_path}")
    print(f"  fps={fps:.2f}  size={width}x{height}  frames={total_frames}  "
          f"events={sum(counts.values())}  actor={'ON' if show_actor else 'OFF'}  "
          f"renderer={mode}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        # 1) players + ball (ReID overlay style)
        draw_players(frame, frame_to_obs.get(frame_idx, []),
                     gid_colors, width, height, label_mode=player_label)

        # 2) update event panel state
        viz.update(frame_idx, frame_events.get(frame_idx, []))

        # 3) who-did-it actor highlights (optional)
        actor_highlights = None
        if show_actor:
            actor_highlights = resolve_actor_highlights(
                frame_idx, events_list, frame_to_obs, hold_frames)

        # 4) composite the broadcast overlay (panel + toasts + flash + timeline
        #    + actor highlights), then write
        frame = viz.render(frame, frame_idx, actor_highlights)
        writer.write(frame)

        if preview:
            cv2.imshow('vis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total_frames}", flush=True)
        frame_idx += 1

    cap.release()
    writer.release()
    if preview:
        cv2.destroyAllWindows()
    print(f"Done. Wrote {frame_idx} frames to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Unified visualizer: players + events + optional actor highlight')
    p.add_argument('--video', '-v', required=True, help='Input video path')
    p.add_argument('--players', '-p', required=True,
                   help='reid_observations.json or tracks.json')
    p.add_argument('--events', '-e', required=True,
                   help="Events JSON ('results' assignment format or 'predictions')")
    p.add_argument('--output', '-o', default='unified_visualization.mp4',
                   help='Output video path')
    p.add_argument('--mapping', default=None,
                   help='Optional trackid_to_globalid.json (only for tracks.json input)')

    # The "who did the event" layer — OPTIONAL, off by default.
    p.add_argument('--show-actor', dest='show_actor', action='store_true',
                   help='Highlight the player assigned to each event')

    p.add_argument('--player-label', default='gid',
                   choices=['tid_gid', 'gid', 'tid', 'none'],
                   help='Player label content (default: GID:y)')
    p.add_argument('--title', default='MATCH EVENTS', help='Panel title text')
    p.add_argument('--no-remove-drive', dest='remove_drive', action='store_false',
                   help='Do not remove DRIVE labels')
    p.add_argument('--no-map-tackle', dest='map_tackle', action='store_false',
                   help='Do not map TACKLE -> BALL PLAYER BLOCK')
    p.add_argument('--min-conf', type=float, default=0.0,
                   help='Drop events below this event_confidence (results format)')
    p.add_argument('--no-pil', dest='use_pil', action='store_false',
                   help='Force the OpenCV fallback renderer')
    p.add_argument('--fps', type=float, default=None, help='Override FPS')
    p.add_argument('--max-frames', type=int, default=None,
                   help='Process only the first N frames (debugging)')
    p.add_argument('--preview', action='store_true',
                   help='Show a preview window while processing')
    args = p.parse_args()

    for label, path in (("Video", args.video), ("Players", args.players),
                        ("Events", args.events)):
        if not os.path.exists(path):
            print(f"{label} not found: {path}")
            sys.exit(1)

    visualize(args.video, args.players, args.events, args.output,
              mapping_path=args.mapping,
              show_actor=args.show_actor,
              remove_drive=args.remove_drive,
              map_tackle=args.map_tackle,
              min_conf=args.min_conf,
              fps_override=args.fps,
              title=args.title,
              use_pil=args.use_pil,
              player_label=args.player_label,
              max_frames=args.max_frames,
              preview=args.preview)