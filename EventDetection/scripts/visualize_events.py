#!/usr/bin/env python3
"""
Visualize events from a JSON file on a video -- now with a beautiful, dynamic
broadcast-style overlay.

Same behaviour as before:
  - Reads `predictions` from a JSON file, each having `frame` and `label`.
  - By default removes `DRIVE` labels and maps any label containing `TACKLE`
    to `BALL PLAYER BLOCK`.
  - Writes an annotated mp4 to --output.

Changes vs previous version:
  - Panel moved to TOP-RIGHT (was top-left).
  - Panel is significantly smaller and less opaque (less intrusive).
  - Row height, bar height, fonts, and padding all reduced.
  - Toast cards moved to TOP-LEFT so they don't collide with the panel.
  - All other behaviour (bar-chart race, smooth reorder, big-event flash,
    timeline, PIL/OpenCV fallback) unchanged.

Usage:
    python3 visualize_events.py \
        --video input_video.mp4 \
        --json inference_output/results_snball.json \
        --output inference_output/visualized_snball.mp4
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


# --------------------------------------------------------------------------- #
#  Data loading / cleaning  (unchanged behaviour)
# --------------------------------------------------------------------------- #
def load_and_clean(json_path, remove_drive=True, map_tackle=True):
    with open(json_path, 'r') as f:
        data = json.load(f)

    preds = data.get('predictions', [])
    frame_events = defaultdict(list)

    for p in preds:
        label = p.get('label', '')
        if not isinstance(label, str):
            continue
        label_norm = label.strip()
        label_upper = label_norm.upper()
        if map_tackle and 'TACKLE' in label_upper:
            label_norm = 'BALL PLAYER BLOCK'
            label_upper = label_norm.upper()
        if remove_drive and label_upper == 'DRIVE':
            continue
        frame = int(p.get('frame', 0))
        frame_events[frame].append(label_norm)

    counts = Counter()
    for v in frame_events.values():
        counts.update(v)

    return frame_events, counts


def format_time(frame_idx, fps):
    seconds = frame_idx / float(fps)
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m:02d}:{s:05.2f}"
    return f"{seconds:.2f}s"


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
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


# A vibrant, harmonious palette (RGB). Colours are assigned deterministically.
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
# A few sport-specific overrides for instant readability.
COLOR_OVERRIDES = {
    'GOAL': (250, 204, 21),
    'SHOT': (248, 113, 113),
}
BIG_EVENTS = {'GOAL'}            # gets the full-screen celebratory flash


# --------------------------------------------------------------------------- #
#  Renderer
# --------------------------------------------------------------------------- #
class EventViz:
    PULSE_SEC = 0.65             # count-pop duration
    TOAST_SEC = 1.7              # toast lifetime
    FLASH_SEC = 1.0              # big-event flash lifetime

    def __init__(self, width, height, fps, counts_total, total_frames,
                 max_event_frame, title="MATCH EVENTS", use_pil=True):
        self.w, self.h, self.fps = width, height, fps
        self.title = title
        self.total_frames = total_frames if total_frames and total_frames > 0 \
            else int(max_event_frame * 1.05) + 1

        # Stable label set + colour assignment (sorted by final total desc).
        self.labels = sorted(counts_total.keys(),
                             key=lambda l: (-counts_total[l], l))
        self.totals = dict(counts_total)
        self.colors = {}
        for i, lab in enumerate(self.labels):
            self.colors[lab] = COLOR_OVERRIDES.get(lab.upper(),
                                                    PALETTE[i % len(PALETTE)])

        # Live animation state
        self.running = {l: 0 for l in self.labels}
        self.pulse_at = {l: -1e9 for l in self.labels}
        self.bar_cur = {l: 0.0 for l in self.labels}
        self.toasts = []          # {label, start}
        self.flashes = []         # {label, start}

        # ------------------------------------------------------------------ #
        #  Geometry — SMALLER panel, anchored TOP-RIGHT
        # ------------------------------------------------------------------ #
        # Scale factor (tuned at 720p), capped tighter so panel stays small.
        s = clamp(height / 720.0, 0.55, 1.6)
        self.s = s

        self.margin  = int(16 * s)
        self.pad     = int(8 * s)
        self.title_h = int(28 * s)      # was 44*s
        self.row_h   = int(22 * s)      # was 38*s  — tighter rows
        self.radius  = int(8 * s)       # was 12*s

        # Narrower panel: ~24% of frame width, hard-capped smaller than before
        self.panel_w = int(clamp(width * 0.24, 180, int(340 * s)))

        # Anchor to TOP-RIGHT (was top-left)
        self.panel_x = self.w - self.panel_w - self.margin
        self.panel_y = self.margin

        n = max(1, len(self.labels))
        self.panel_h = self.title_h + n * self.row_h + self.pad

        # Initial row order / positions (so the panel doesn't slide on frame 0).
        order = self._order()
        self.y_cur = {}
        content_top = self.panel_y + self.title_h + int(3 * s)
        for idx, lab in enumerate(order):
            self.y_cur[lab] = content_top + idx * self.row_h

        # Fonts
        self.use_pil = bool(HAVE_PIL and use_pil)
        self._font_cache = {}
        self._bold_path = self._find_font(
            ["Poppins-Bold.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf",
             "arialbd.ttf"])
        self._med_path = self._find_font(
            ["Poppins-Medium.ttf", "Poppins-Regular.ttf", "DejaVuSans.ttf",
             "Arial.ttf", "arial.ttf"])
        if self.use_pil and (self._bold_path is None or self._med_path is None):
            self.use_pil = False  # no usable TTF -> fall back to OpenCV

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

        # expire toasts / flashes
        self.toasts = [t for t in self.toasts
                       if (frame_idx - t['start']) / self.fps <= self.TOAST_SEC]
        self.flashes = [f for f in self.flashes
                        if (frame_idx - f['start']) / self.fps <= self.FLASH_SEC]

        # ease row positions toward target order
        order = self._order()
        content_top = self.panel_y + self.title_h + int(3 * self.s)
        for idx, lab in enumerate(order):
            target = content_top + idx * self.row_h
            self.y_cur[lab] = lerp(self.y_cur[lab], target, 0.22)

        # ease bar widths toward target (normalised to current max).
        # A right-hand column is reserved for the count so it stays readable.
        max_count = max(1, max(self.running.values()))
        cnt_col_w = int(28 * self.s)           # was 46*s
        bar_max = self.panel_w - 2 * self.pad - cnt_col_w
        for lab in self.labels:
            target = (self.running[lab] / max_count) * bar_max
            self.bar_cur[lab] = lerp(self.bar_cur[lab], target, 0.25)

    # ===================================================================== #
    #  PIL rendering (the pretty path)
    # ===================================================================== #
    def render(self, frame_bgr, frame_idx):
        if self.use_pil:
            return self._render_pil(frame_bgr, frame_idx)
        return self._render_cv2(frame_bgr, frame_idx)

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

    def _render_pil(self, frame_bgr, frame_idx):
        base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        s = self.s

        px, py, pw, ph = self.panel_x, self.panel_y, self.panel_w, self.panel_h

        # ---- panel shadow + body ----------------------------------------- #
        # More transparent than original (165 vs 205) to be less intrusive
        self._rrect(d, (px + int(3 * s), py + int(4 * s),
                        px + pw + int(3 * s), py + ph + int(4 * s)),
                    self.radius, (0, 0, 0, 60))
        self._rrect(d, (px, py, px + pw, py + ph), self.radius, (16, 18, 26, 165))

        # ---- title bar ---------------------------------------------------- #
        self._rrect(d, (px, py, px + pw, py + self.title_h),
                    self.radius, (30, 34, 48, 195))
        d.rectangle([px, py + self.title_h - self.radius,
                     px + pw, py + self.title_h], fill=(30, 34, 48, 195))

        # Title text — smaller font (14*s was 20*s)
        title_font = self._font(14 * s, bold=True)
        self._text(d,
                   (px + self.pad,
                    py + self.title_h // 2 - self._tsize(title_font, self.title)[1] // 2 - int(1 * s)),
                   self.title, title_font, (236, 240, 248, 255))

        # Live clock on the right side of title bar — smaller (10*s was 15*s)
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
        # Smaller fonts: label 11*s (was 16*s), count ~11.5*s (was 16*s)
        lab_font = self._font(11 * s, bold=True)
        bar_x0 = px + self.pad
        cnt_col_w = int(28 * s)         # was 46*s
        bar_max = pw - 2 * self.pad - cnt_col_w
        bar_h = int(self.row_h * 0.52)  # slightly thinner bar (was 0.62)

        for lab in self.labels:
            y = int(self.y_cur[lab])
            cy = y + (self.row_h - bar_h) // 2
            col = self.colors[lab]
            cnt = self.running[lab]

            # pulse factor (1 right after increment -> 0)
            age_p = (frame_idx - self.pulse_at[lab]) / self.fps
            p = 0.0
            if 0 <= age_p <= self.PULSE_SEC:
                p = 1.0 - ease_out_cubic(age_p / self.PULSE_SEC)

            # track background
            self._rrect(d, (bar_x0, cy, bar_x0 + bar_max, cy + bar_h),
                        bar_h // 2, (255, 255, 255, 22))

            # glow on pulse
            if p > 0.01:
                g = int(100 * p)
                self._rrect(d, (bar_x0 - int(2 * s), cy - int(2 * s),
                                bar_x0 + max(bar_h, self.bar_cur[lab]) + int(2 * s),
                                cy + bar_h + int(2 * s)),
                            bar_h // 2 + int(2 * s), col + (g,))

            # fill bar
            fill_w = max(bar_h if cnt > 0 else 0, self.bar_cur[lab])
            if fill_w > 0:
                a = 235 if cnt > 0 else 60
                self._rrect(d, (bar_x0, cy, bar_x0 + fill_w, cy + bar_h),
                            bar_h // 2, col + (a,))

            # label text (on/above bar)
            txt_col = (255, 255, 255, 255) if cnt > 0 else (150, 156, 170, 255)
            self._text(d,
                       (bar_x0 + int(6 * s),
                        cy + bar_h // 2 - self._tsize(lab_font, lab)[1] // 2 - int(1 * s)),
                       lab, lab_font, txt_col)

            # count value (right column, pops on pulse)
            # Pulse scaling reduced slightly (0.35 was 0.5)
            cnt_size = (11.5 * s) * (1.0 + 0.35 * p)
            cnt_font = self._font(cnt_size, bold=True)
            cnt_col_clr = tuple(int(clamp(c + 60 * p, 0, 255)) for c in col) + (255,) \
                if cnt > 0 else (150, 156, 170, 255)
            self._text(d,
                       (px + pw - self.pad,
                        cy + bar_h // 2 - self._tsize(cnt_font, str(cnt))[1] // 2 - int(1 * s)),
                       str(cnt), cnt_font, cnt_col_clr, anchor_x='r')

        # ---- toasts (TOP-LEFT — moved away from panel) ------------------- #
        self._draw_toasts_pil(d, frame_idx)

        # ---- big-event flash (centre) ------------------------------------ #
        self._draw_flash_pil(d, frame_idx)

        # ---- timeline (bottom) ------------------------------------------ #
        self._draw_timeline_pil(d, frame_idx)

        out = Image.alpha_composite(base, ov).convert("RGB")
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    def _draw_toasts_pil(self, d, frame_idx):
        """
        Toast cards now slide in from the LEFT (top-left corner)
        to avoid colliding with the panel that is now on the right.
        """
        s = self.s
        card_w = int(clamp(self.w * 0.22, 160, 280))
        card_h = int(38 * s)
        gap = int(8 * s)
        left = self.margin          # left edge anchor (was right edge)
        top = self.margin
        # newest first, stacked downward
        active = sorted(self.toasts, key=lambda t: -t['start'])
        for i, t in enumerate(active):
            age = (frame_idx - t['start']) / self.fps
            fin = clamp(age / 0.18, 0, 1)
            fout = clamp((self.TOAST_SEC - age) / 0.45, 0, 1)
            a = min(fin, fout)
            if a <= 0.01:
                continue
            # Slide in from the left (negative x offset fades to 0)
            slide = int((1 - ease_out_cubic(fin)) * 55 * s)
            x0 = left - slide
            x1 = x0 + card_w
            y0 = top + i * (card_h + gap)
            y1 = y0 + card_h
            col = self.colors.get(t['label'], (200, 200, 200))
            A = int(235 * a)
            self._rrect(d, (x0, y0, x1, y1), int(8 * s), (18, 20, 30, int(215 * a)))
            # colour accent bar on the LEFT side
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
        h = int(5 * s)      # was 6*s
        self._rrect(d, (x0, y - h // 2, x1, y + h // 2), h // 2, (255, 255, 255, 40))
        frac = clamp(frame_idx / max(1, self.total_frames), 0, 1)
        px = int(x0 + frac * (x1 - x0))
        self._rrect(d, (x0, y - h // 2, px, y + h // 2), h // 2, (120, 200, 255, 220))
        ph = int(7 * s)     # was 8*s
        d.ellipse([px - ph, y - ph, px + ph, y + ph], fill=(235, 242, 252, 255))
        d.ellipse([px - ph // 2, y - ph // 2, px + ph // 2, y + ph // 2],
                  fill=(90, 170, 250, 255))

    # ===================================================================== #
    #  OpenCV fallback (no Pillow / no fonts)
    # ===================================================================== #
    def _render_cv2(self, frame, frame_idx):
        s = self.s
        font = cv2.FONT_HERSHEY_SIMPLEX
        px, py, pw, ph = self.panel_x, self.panel_y, self.panel_w, self.panel_h

        ov = frame.copy()
        cv2.rectangle(ov, (px, py), (px + pw, py + ph), (26, 18, 16), -1)
        cv2.rectangle(ov, (px, py), (px + pw, py + self.title_h), (48, 34, 30), -1)
        # Use a lower alpha (0.70 blend) to keep it less intrusive
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

        # toasts — TOP-LEFT
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

        # timeline
        x0, x1 = self.margin, self.w - self.margin
        yb = self.h - self.margin
        cv2.line(frame, (x0, yb), (x1, yb), (90, 90, 90), max(1, int(4 * s)))
        frac = clamp(frame_idx / max(1, self.total_frames), 0, 1)
        pxh = int(x0 + frac * (x1 - x0))
        cv2.line(frame, (x0, yb), (pxh, yb), (255, 200, 120), max(1, int(4 * s)))
        cv2.circle(frame, (pxh, yb), int(6 * s), (252, 242, 235), -1)
        return frame


# --------------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------------- #
def visualize(video_path, json_path, out_path, remove_drive=True, map_tackle=True,
              fps_override=None, preview=False, title="MATCH EVENTS", use_pil=True):
    frame_events, counts = load_and_clean(json_path, remove_drive=remove_drive,
                                          map_tackle=map_tackle)
    max_event_frame = max(frame_events.keys()) if frame_events else 0

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
    mode = "Pillow (high quality)" if viz.use_pil else "OpenCV fallback"
    print(f"Writing visualization to {out_path}")
    print(f"  fps={fps:.2f}  size={width}x{height}  frames={total_frames}  "
          f"events={sum(counts.values())}  renderer={mode}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        viz.update(frame_idx, frame_events.get(frame_idx, []))
        frame = viz.render(frame, frame_idx)
        writer.write(frame)

        if preview:
            cv2.imshow('vis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1

    cap.release()
    writer.release()
    if preview:
        cv2.destroyAllWindows()
    print(f"Done. Wrote {frame_idx} frames to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Visualize events JSON on video frames (pretty + dynamic)')
    p.add_argument('--video', '-v', required=True, help='Input video path')
    p.add_argument('--json', '-j',
                   default='inference_output/results_snball.json',
                   help='Events JSON file')
    p.add_argument('--output', '-o',
                   default='inference_output/visualized_snball.mp4',
                   help='Output video path')
    p.add_argument('--title', default='MATCH EVENTS', help='Panel title text')
    p.add_argument('--no-remove-drive', dest='remove_drive',
                   action='store_false', help='Do not remove DRIVE labels')
    p.add_argument('--no-map-tackle', dest='map_tackle',
                   action='store_false',
                   help='Do not map TACKLE -> BALL PLAYER BLOCK')
    p.add_argument('--no-pil', dest='use_pil', action='store_false',
                   help='Force the OpenCV fallback renderer')
    p.add_argument('--fps', type=float, default=None,
                   help='Override FPS (seconds computed from this)')
    p.add_argument('--preview', action='store_true',
                   help='Show a preview window while processing')
    args = p.parse_args()

    if not os.path.exists(args.json):
        print(f"JSON not found: {args.json}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)

    visualize(args.video, args.json, args.output,
              remove_drive=args.remove_drive,
              map_tackle=args.map_tackle,
              fps_override=args.fps,
              preview=args.preview,
              title=args.title,
              use_pil=args.use_pil)