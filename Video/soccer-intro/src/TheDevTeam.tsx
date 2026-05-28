import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { F, SVG_HALF } from "./TheFrustration";
import { SMOOTH } from "./SoccerIntro";

// ── Helpers ────────────────────────────────────────────────────────────────
const CLAMP = {
  extrapolateLeft: "clamp" as const,
  extrapolateRight: "clamp" as const,
};

// ── Hex grid (inline copy from ThePipeline) ─────────────────────────────────
function hexPoints(cx: number, cy: number, R: number): string {
  return Array.from({ length: 6 }, (_, k) => {
    const a = (Math.PI / 3) * k - Math.PI / 6;
    return `${(cx + R * Math.cos(a)).toFixed(2)},${(cy + R * Math.sin(a)).toFixed(2)}`;
  }).join(" ");
}
const HEX_R    = 32;
const HEX_CX   = HEX_R * Math.sqrt(3);
const HEX_CY   = 2 * HEX_R;
const HEX_COLS = Math.ceil(1920 / HEX_CX) + 2;
const HEX_ROWS = Math.ceil(1080 / HEX_CY) + 2;
const HEXAGONS = Array.from({ length: HEX_COLS }, (_, col) => {
  const cx = col * HEX_CX - HEX_CX;
  return Array.from({ length: HEX_ROWS }, (_, row) => {
    const offset = col % 2 === 1 ? HEX_R : 0;
    const cy = row * HEX_CY - HEX_CY + offset;
    return { cx, cy, pts: hexPoints(cx, cy, HEX_R) };
  });
}).flat();

// ── Timing ─────────────────────────────────────────────────────────────────
const T = {
  dev1In:         10,   // Dev1 slides from left
  dev2In:         25,   // Dev2 slides from right
  bulbAppear:     70,   // Lightbulb springs in
  bulbOn:         88,   // Lightbulb flashes
  bubbleIn:      120,   // IdeaBubble opens
  // speech lines at 125 / 190 / 250
  bubbleOutStart: 295,
  bubbleOutEnd:   315,
  figFadeStart:   325,  // Figures + bulb fade out
  figFadeEnd:     352,
  cameraIn:       320,  // Camera slides up from below
  lensStart:      360,  // Lens circle begins expanding
  lensEnd:        440,  // Lens circle fills screen
};

// ── Speech lines ────────────────────────────────────────────────────────────
const IDEA_LINES = [
  { text: "There is a way!",                        appearAt: 125 },
  { text: "We'll point a camera at the match —",    appearAt: 190 },
  { text: "and our AI pipeline does the rest!",     appearAt: 250 },
];

// ── Camera + lens absolute screen coordinates ──────────────────────────────
// Video camera SVG is 420×240. Lens barrel centre at SVG (115, 115).
// We want the lens at screen (960, 490) — the exact screen centre — so the
// LensZoom explosion feels symmetrical.
//   div left = 960 − 115 = 845
//   div top  = 490 − 115 = 375
const CAM_DIV_L  = 845;
const CAM_DIV_T  = 375;
const LENS_SCR_X = 960;   // 845 + 115
const LENS_SCR_Y = 490;   // 375 + 115

// ── Background ──────────────────────────────────────────────────────────────
const Background: React.FC<{ frame: number; glowUp: boolean }> = ({ frame, glowUp }) => {
  const hexPulse = Math.sin(frame * 0.015) * 0.025 + 0.045;
  const glowOp = glowUp
    ? interpolate(frame, [T.bulbOn, T.bulbOn + 20], [0.35, 0.75], CLAMP)
    : 0.35;
  const glowColor = glowUp ? "80,60,10" : "0,30,90";

  return (
    <AbsoluteFill style={{ backgroundColor: "#03050A" }}>
      {/* Central glow — warm yellow tint when bulb is on */}
      <div style={{
        position: "absolute",
        inset: 0,
        background: `radial-gradient(ellipse 70% 55% at 26% 56%, rgba(${glowColor},${glowOp.toFixed(2)}) 0%, transparent 70%)`,
      }} />

      {/* Hex grid */}
      <svg style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }}>
        {HEXAGONS.map(({ pts }, i) => (
          <polygon
            key={i}
            points={pts}
            fill="none"
            stroke={`rgba(0,160,255,${hexPulse.toFixed(3)})`}
            strokeWidth={0.5}
          />
        ))}
      </svg>

      {/* Vignette */}
      <div style={{
        position: "absolute",
        inset: 0,
        background: "radial-gradient(ellipse 80% 80% at 50% 50%, transparent 60%, rgba(0,0,0,0.75) 100%)",
        pointerEvents: "none",
      }} />
    </AbsoluteFill>
  );
};

// ── DevFigure ──────────────────────────────────────────────────────────────
interface DevFigureProps {
  frame:      number;
  fps:        number;
  footX:      number;
  footY:      number;
  color:      string;
  hairStyle:  "curly" | "short";
  isTalking:  boolean;
  enterFrom:  "left" | "right";
  appearAt:   number;
}

const DevFigure: React.FC<DevFigureProps> = ({
  frame, fps, footX, footY, color, hairStyle, isTalking, enterFrom, appearAt,
}) => {
  if (frame < appearAt) return null;

  const localFrame = frame - appearAt;

  // Slide in horizontally from the edge
  const slideIn = spring({
    frame: localFrame,
    fps,
    config: { stiffness: 90, damping: 18 },
  });
  const enterX = (1 - slideIn) * (enterFrom === "left" ? -700 : 700);

  // Subtle body sway
  const swayAmp    = Math.min(localFrame / 60, 1) * 3;
  const bodyShiftX = Math.sin(frame * 0.6) * swayAmp;

  // Relaxed arms at sides with gentle pendulum sway
  const armSway = Math.sin(frame * 0.45) * 6;
  const rax = F.shoulderArmX + 40 + armSway;
  const ray = F.hipY - 20;           // hang below shoulder, above foot
  const lax = -(F.shoulderArmX + 40 + armSway * 0.75);
  const lay = F.hipY - 20;

  // Face fades in with entrance
  const faceOp = interpolate(localFrame, [5, 22], [0, 1], CLAMP);

  // Talking mouth opens/closes when isTalking and speech has started
  const talkStart = appearAt + 105;
  const mouthOpen = isTalking && frame >= talkStart
    ? Math.max(0, Math.sin(frame * 0.55)) * 14
    : 0;

  const divL = footX - SVG_HALF;
  const divT = footY - SVG_HALF;

  return (
    <div
      style={{
        position: "absolute",
        left: divL,
        top:  divT,
        width:  SVG_HALF * 2,
        height: SVG_HALF * 2,
        transform: `translateX(${enterX}px)`,
        pointerEvents: "none",
      }}
    >
      <svg
        width={SVG_HALF * 2}
        height={SVG_HALF * 2}
        style={{ overflow: "visible" }}
      >
        <g transform={`translate(${SVG_HALF}, ${SVG_HALF}) translate(${bodyShiftX}, 0)`}>

          {/* ── Legs ── */}
          <line
            x1={0} y1={F.hipY}
            x2={-55} y2={0}
            stroke={color} strokeWidth={F.strokeLimb} strokeLinecap="round"
          />
          <line
            x1={0} y1={F.hipY}
            x2={ 55} y2={0}
            stroke={color} strokeWidth={F.strokeLimb} strokeLinecap="round"
          />

          {/* ── Torso ── */}
          <line
            x1={0} y1={F.neckY}
            x2={0} y2={F.hipY}
            stroke={color} strokeWidth={F.strokeBody} strokeLinecap="round"
          />

          {/* ── Left arm (relaxed at side) ── */}
          <line
            x1={0} y1={F.neckY + 14}
            x2={lax} y2={lay}
            stroke={color} strokeWidth={F.strokeLimb} strokeLinecap="round"
          />

          {/* ── Right arm ── */}
          <line
            x1={0} y1={F.neckY + 14}
            x2={rax} y2={ray}
            stroke={color} strokeWidth={F.strokeLimb} strokeLinecap="round"
          />

          {/* ── Head ── */}
          <circle
            cx={0} cy={F.headY}
            r={F.headR}
            fill={color}
            stroke="rgba(0,0,0,0.5)"
            strokeWidth={5}
          />

          {/* ── Hair ── */}
          {hairStyle === "curly" ? (
            // Five arc bumps — curly hair
            <g>
              {([-60, -30, 0, 30, 60] as number[]).map((dx, i) => (
                <path
                  key={i}
                  d={`M ${dx - 14} ${F.headY - F.headR + 6} Q ${dx} ${F.headY - F.headR - 24} ${dx + 14} ${F.headY - F.headR + 6}`}
                  stroke={color}
                  strokeWidth={10}
                  fill="none"
                  strokeLinecap="round"
                />
              ))}
            </g>
          ) : (
            // Short compact rounded cap
            <path
              d={`M ${-F.headR + 10} ${F.headY - 18} Q 0 ${F.headY - F.headR - 28} ${F.headR - 10} ${F.headY - 18}`}
              stroke={color}
              strokeWidth={12}
              fill="none"
              strokeLinecap="round"
            />
          )}

          {/* ── Face (fades in with entrance) ── */}
          <g opacity={faceOp}>

            {/* Eyes — round, friendly */}
            <ellipse cx={-26} cy={F.headY - 12} rx={14} ry={11} fill="white" />
            <circle  cx={-24} cy={F.headY - 10} r={6}           fill="#111" />
            <circle  cx={-20} cy={F.headY - 14} r={3}           fill="rgba(255,255,255,0.7)" />

            <ellipse cx={ 26} cy={F.headY - 12} rx={14} ry={11} fill="white" />
            <circle  cx={ 28} cy={F.headY - 10} r={6}           fill="#111" />
            <circle  cx={ 32} cy={F.headY - 14} r={3}           fill="rgba(255,255,255,0.7)" />

            {/* Mouth — talking oval or neutral smile */}
            {mouthOpen > 2 ? (
              <ellipse
                cx={0} cy={F.headY + 32}
                rx={18} ry={Math.max(4, mouthOpen)}
                fill="rgba(0,0,0,0.85)"
              />
            ) : (
              <path
                d={`M -22 ${F.headY + 26} Q 0 ${F.headY + 48} 22 ${F.headY + 26}`}
                stroke="rgba(0,0,0,0.8)"
                strokeWidth={6}
                fill="none"
                strokeLinecap="round"
              />
            )}

            {/* Shirt V-neck detail (only for curly/Dev1) */}
            {hairStyle === "curly" && (
              <>
                <line
                  x1={-13} y1={F.neckY - 8}
                  x2={0}   y2={F.neckY + 22}
                  stroke="rgba(255,255,255,0.38)" strokeWidth={4} strokeLinecap="round"
                />
                <line
                  x1={ 13} y1={F.neckY - 8}
                  x2={0}   y2={F.neckY + 22}
                  stroke="rgba(255,255,255,0.38)" strokeWidth={4} strokeLinecap="round"
                />
              </>
            )}
          </g>
        </g>
      </svg>
    </div>
  );
};

// ── LightBulb ──────────────────────────────────────────────────────────────
// Centred above Dev1's head. Dev1 head screen: (500, 1020 − 420) = (500, 600).
// Bulb circle centred at screen (500, 460); SVG bulb centre at cy=60, so
// div top = 460 − 60 = 400.
const BULB_SCR_X = 500;
const BULB_SCR_Y = 460;

const LightBulb: React.FC<{ frame: number; fps: number }> = ({ frame, fps }) => {
  if (frame < T.bulbAppear - 5) return null;

  const sc = spring({
    frame: Math.max(0, frame - T.bulbAppear),
    fps,
    config: { stiffness: 220, damping: 14 },
  });

  // Flicker just before turn-on
  const isFlickering = frame >= T.bulbOn - 8 && frame < T.bulbOn;
  const flicker = isFlickering
    ? 0.35 + Math.abs(Math.sin(frame * 3.5)) * 0.65
    : 1;

  // Flash burst at T.bulbOn
  const flash = interpolate(frame, [T.bulbOn, T.bulbOn + 4, T.bulbOn + 10], [0, 1, 0], CLAMP);

  // Steady warm glow pulse after on
  const glowPulse = frame >= T.bulbOn
    ? 0.38 + Math.sin(frame * 0.08) * 0.09
    : 0;

  return (
    <div
      style={{
        position: "absolute",
        left: BULB_SCR_X - 60,
        top:  BULB_SCR_Y - 60,
        width:  120,
        height: 160,
        transform: `scale(${sc})`,
        transformOrigin: "50% 100%",   // spring grows upward from base
        opacity: flicker,
        pointerEvents: "none",
      }}
    >
      <svg width={120} height={160} style={{ overflow: "visible" }}>

        {/* Outer warm glow rings */}
        {glowPulse > 0 && (
          <>
            <circle
              cx={60} cy={60}
              r={55 + glowPulse * 18}
              fill="none"
              stroke={`rgba(255,220,50,${glowPulse.toFixed(2)})`}
              strokeWidth={12}
            />
            <circle
              cx={60} cy={60}
              r={78 + glowPulse * 28}
              fill="none"
              stroke={`rgba(255,200,30,${(glowPulse * 0.38).toFixed(2)})`}
              strokeWidth={8}
            />
          </>
        )}

        {/* Flash burst */}
        {flash > 0.01 && (
          <circle
            cx={60} cy={60}
            r={160}
            fill={`rgba(255,240,180,${(flash * 0.28).toFixed(2)})`}
          />
        )}

        {/* Bulb body */}
        <circle
          cx={60} cy={60} r={42}
          fill="#FFF9C4"
          stroke="#FFC107"
          strokeWidth={4}
        />

        {/* Filament zigzag */}
        <path
          d="M 50,54 L 54,40 L 60,54 L 66,40 L 70,54"
          stroke="#FFA000"
          strokeWidth={3}
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Bulb base — three stacked rects */}
        <rect x={44} y={96}  width={32} height={10} rx={4} fill="#FFC107" stroke="#FF8F00" strokeWidth={2} />
        <rect x={46} y={104} width={28} height={8}  rx={3} fill="#FFB300" stroke="#FF8F00" strokeWidth={2} />
        <rect x={48} y={110} width={24} height={8}  rx={3} fill="#FFA000" stroke="#FF8F00" strokeWidth={2} />

        {/* Bulb shine glare arc */}
        <path
          d="M 42,40 Q 36,54 38,64"
          stroke="rgba(255,255,255,0.65)"
          strokeWidth={5}
          fill="none"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};

// ── IdeaBubble ─────────────────────────────────────────────────────────────
// Same white/red-border style as SpeechBubble in TheFrustration.
// Tail points downward toward Dev1's head at screen (500, 600).
// Bubble: left=290, top=60, width=1340, height=290.
// Bubble bottom is at y=350; tail offset left ≈ (500−290)=210px from bubble left.
const IdeaBubble: React.FC<{ frame: number; fps: number }> = ({ frame, fps }) => {
  if (frame < T.bubbleIn - 5) return null;

  const sc = spring({
    frame: Math.max(0, frame - T.bubbleIn),
    fps,
    config: { stiffness: 190, damping: 18 },
  });

  const fadeOut = interpolate(frame, [T.bubbleOutStart, T.bubbleOutEnd], [1, 0], CLAMP);
  if (fadeOut <= 0) return null;

  return (
    <div
      style={{
        position: "absolute",
        left:   290,
        top:     60,
        width:  1340,
        height:  290,
        transform: `scaleX(${sc})`,
        transformOrigin: "center center",
        opacity: fadeOut,
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          width:    "100%",
          height:   "100%",
          background: "rgba(255,255,255,0.97)",
          border:   "3px solid #e53935",
          borderRadius: 20,
          display:  "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding:  "20px 48px",
          gap: 4,
          position: "relative",
        }}
      >
        {/* Tail — points downward toward Dev1 (screen x=500, bubble left=290 → offset 210) */}
        <div style={{ position: "absolute", left: 190, bottom: -22 }}>
          <svg width={44} height={25} style={{ display: "block" }}>
            <polygon points="0,0 44,0 22,24" fill="rgba(255,255,255,0.97)" />
            <line x1={0}  y1={0} x2={22} y2={24} stroke="#e53935" strokeWidth={3} />
            <line x1={44} y1={0} x2={22} y2={24} stroke="#e53935" strokeWidth={3} />
          </svg>
        </div>

        {IDEA_LINES.map(({ text, appearAt }, i) => {
          if (frame < appearAt - 1) return null;
          const opacity = interpolate(frame, [appearAt, appearAt + 20], [0, 1], CLAMP);
          const slideY  = interpolate(frame, [appearAt, appearAt + 20], [18, 0], {
            ...CLAMP,
            easing: SMOOTH,
          });
          return (
            <div key={i} style={{ opacity, transform: `translateY(${slideY}px)` }}>
              <span
                style={{
                  fontSize:   46,
                  fontWeight: 900,
                  color:      "#1a1a1a",
                  fontFamily: "Impact, 'Arial Black', sans-serif",
                  letterSpacing: 1,
                  lineHeight: 1.2,
                  display:    "block",
                }}
              >
                {text}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ── CameraGraphic ──────────────────────────────────────────────────────────
// Broadcast-style video camera. SVG 420×240.
// Lens barrel centre: SVG (115, 115) → screen (960, 490).
//
// Parts:
//  • Top carry handle
//  • Shotgun microphone on handle
//  • Main rectangular body
//  • Large lens assembly (hood → focus ring → barrel → glass layers)
//  • Viewfinder housing + eyepiece (back right)
//  • Hand-grip (bottom right)
//  • Recording indicator light (red LED)
//  • Small control panel and LCD detail
const CameraGraphic: React.FC<{ frame: number; fps: number }> = ({ frame, fps }) => {
  if (frame < T.cameraIn - 5) return null;

  const slideIn = spring({
    frame: Math.max(0, frame - T.cameraIn),
    fps,
    config: { stiffness: 100, damping: 16 },
  });
  const enterY = (1 - slideIn) * 320;

  // Lens glow pulses brighter as the zoom-in approaches
  const preGlow = interpolate(frame, [T.lensStart - 18, T.lensStart], [0, 1], CLAMP);
  const glowR   = 52 + preGlow * 30;
  const glowOp  = preGlow * 0.75;

  // REC light blinks every ~20 frames while camera is on screen
  const recOn = frame >= T.cameraIn + 15;
  const recBlink = recOn ? (Math.floor(frame / 20) % 2 === 0 ? 1 : 0.3) : 0;

  return (
    <div
      style={{
        position: "absolute",
        left: CAM_DIV_L,
        top:  CAM_DIV_T,
        width:  420,
        height: 240,
        transform: `translateY(${enterY}px)`,
        pointerEvents: "none",
      }}
    >
      <svg width={420} height={240} style={{ overflow: "visible" }}>
        <defs>
          {/* Radial gradient for lens depth */}
          <radialGradient id="lensGrad" cx="40%" cy="35%" r="65%">
            <stop offset="0%"   stopColor="#1a3a8a" stopOpacity="0.9" />
            <stop offset="60%"  stopColor="#04102e" stopOpacity="1" />
            <stop offset="100%" stopColor="#000510" stopOpacity="1" />
          </radialGradient>
        </defs>

        {/* ── Top carry handle ── */}
        <rect x={88} y={20} width={210} height={28} rx={6}
          fill="#1a1a2e" stroke="#42A5F5" strokeWidth={2} />
        {/* Handle grip texture lines */}
        {[108, 128, 148, 168, 188, 208, 228, 248, 268].map((x, i) => (
          <line key={i} x1={x} y1={24} x2={x} y2={44}
            stroke="rgba(66,165,245,0.3)" strokeWidth={1} />
        ))}

        {/* ── Shotgun microphone ── */}
        <rect x={132} y={5} width={118} height={15} rx={5}
          fill="#0d1117" stroke="#42A5F5" strokeWidth={1.5} />
        {/* Mic capsule dots */}
        {[148, 162, 176, 190, 204, 218].map((x, i) => (
          <circle key={i} cx={x} cy={12} r={2.5}
            fill="rgba(66,165,245,0.5)" />
        ))}
        {/* Mic mount / clip */}
        <rect x={178} y={18} width={26} height={6} rx={2}
          fill="#1a1a2e" stroke="#42A5F5" strokeWidth={1} />

        {/* ── Main body ── */}
        <rect x={68} y={46} width={272} height={128} rx={10}
          fill="#1a1a2e" stroke="#42A5F5" strokeWidth={3} />

        {/* Body top edge accent line */}
        <line x1={78} y1={50} x2={330} y2={50}
          stroke="rgba(66,165,245,0.3)" strokeWidth={1} />

        {/* ── Lens assembly ── */}
        {/* Lens hood (square-to-circle adapter at front) */}
        <rect x={44} y={82} width={34} height={66} rx={5}
          fill="#111827" stroke="#42A5F5" strokeWidth={2} />

        {/* Outer focus ring — textured band */}
        <circle cx={115} cy={115} r={62}
          fill="#111827" stroke="#42A5F5" strokeWidth={3} />
        {/* Focus ring grip markings */}
        {Array.from({ length: 16 }, (_, i) => {
          const a = (i / 16) * Math.PI * 2;
          const r1 = 55, r2 = 63;
          return (
            <line
              key={i}
              x1={115 + r1 * Math.cos(a)} y1={115 + r1 * Math.sin(a)}
              x2={115 + r2 * Math.cos(a)} y2={115 + r2 * Math.sin(a)}
              stroke="rgba(66,165,245,0.45)" strokeWidth={2}
            />
          );
        })}

        {/* Zoom ring — inner band */}
        <circle cx={115} cy={115} r={55}
          fill="#0c1320" stroke="rgba(100,160,255,0.4)" strokeWidth={2} />
        {/* Zoom markings */}
        {Array.from({ length: 8 }, (_, i) => {
          const a = (i / 8) * Math.PI * 2 - Math.PI / 8;
          return (
            <line
              key={i}
              x1={115 + 47 * Math.cos(a)} y1={115 + 47 * Math.sin(a)}
              x2={115 + 54 * Math.cos(a)} y2={115 + 54 * Math.sin(a)}
              stroke="rgba(66,165,245,0.6)" strokeWidth={1.5}
            />
          );
        })}

        {/* Lens barrel body */}
        <circle cx={115} cy={115} r={46}
          fill="#06091a" stroke="rgba(66,130,200,0.5)" strokeWidth={1} />

        {/* Lens glass — deep layered reflections */}
        <circle cx={115} cy={115} r={43} fill="url(#lensGrad)" />
        <circle cx={115} cy={115} r={34}
          fill="rgba(8,20,70,0.92)" stroke="rgba(80,160,255,0.28)" strokeWidth={1} />
        <circle cx={115} cy={115} r={22}
          fill="rgba(12,30,90,0.88)" stroke="rgba(80,160,255,0.2)" strokeWidth={1} />
        <circle cx={115} cy={115} r={11}
          fill="rgba(18,44,120,0.94)" />
        <circle cx={115} cy={115} r={4}
          fill="rgba(40,80,180,1)" />

        {/* Lens glare — off-centre arc */}
        <path d="M 96,96 Q 88,108 92,118"
          stroke="rgba(255,255,255,0.55)" strokeWidth={4.5}
          fill="none" strokeLinecap="round" />
        <circle cx={100} cy={94} r={4}
          fill="rgba(255,255,255,0.35)" />

        {/* Pre-zoom lens glow */}
        {glowOp > 0.01 && (
          <circle cx={115} cy={115} r={glowR}
            fill="none"
            stroke={`rgba(100,180,255,${glowOp.toFixed(2)})`}
            strokeWidth={10}
          />
        )}

        {/* ── Viewfinder housing (back right) ── */}
        <rect x={334} y={56} width={68} height={50} rx={7}
          fill="#1a1a2e" stroke="#42A5F5" strokeWidth={2} />
        {/* Viewfinder screen (tiny LCD) */}
        <rect x={340} y={62} width={42} height={32} rx={3}
          fill="#0a1a0a" stroke="rgba(66,165,245,0.5)" strokeWidth={1} />
        <rect x={343} y={65} width={36} height={26} rx={2}
          fill="rgba(0,80,30,0.5)" />
        {/* Eyepiece cup */}
        <ellipse cx={408} cy={81} rx={14} ry={20}
          fill="#111" stroke="#42A5F5" strokeWidth={2} />
        <ellipse cx={408} cy={81} rx={9} ry={14}
          fill="#050a05" />

        {/* ── Hand grip (bottom right) ── */}
        <rect x={308} y={134} width={52} height={82} rx={9}
          fill="#151520" stroke="#42A5F5" strokeWidth={2} />
        {/* Grip texture */}
        {[148, 162, 176, 190, 204].map((y, i) => (
          <line key={i} x1={312} y1={y} x2={356} y2={y}
            stroke="rgba(66,165,245,0.22)" strokeWidth={1} />
        ))}
        {/* Trigger / record button on grip */}
        <circle cx={334} cy={142} r={8}
          fill="#ff1744" stroke="#b71c1c" strokeWidth={2}
          opacity={recBlink} />

        {/* ── REC indicator LED (front of body) ── */}
        <circle cx={94} cy={64} r={7}
          fill="#ff1744" stroke="#b71c1c" strokeWidth={1.5}
          opacity={recBlink} />
        {/* LED glow halo */}
        {recBlink > 0.5 && (
          <circle cx={94} cy={64} r={13}
            fill="none"
            stroke="rgba(255,23,68,0.35)"
            strokeWidth={4} />
        )}

        {/* ── Control panel detail ── */}
        <rect x={210} y={60} width={32} height={20} rx={3}
          fill="rgba(66,165,245,0.1)" stroke="#42A5F5" strokeWidth={1} />
        <circle cx={218} cy={70} r={4}
          fill="rgba(66,165,245,0.6)" />
        <circle cx={232} cy={70} r={4}
          fill="rgba(100,200,100,0.5)" />

        {/* Zoom rocker on top of body */}
        <rect x={250} y={46} width={38} height={10} rx={4}
          fill="#0d1117" stroke="#42A5F5" strokeWidth={1.5} />
        <rect x={250} y={46} width={16} height={10} rx={4}
          fill="rgba(66,165,245,0.3)" />
      </svg>
    </div>
  );
};

// ── LensZoom ───────────────────────────────────────────────────────────────
// A black circle centred on the camera lens expands to cover the entire screen.
// Blue iris ring trails the expanding edge for a cinematic feel.
const LensZoom: React.FC<{ frame: number }> = ({ frame }) => {
  if (frame < T.lensStart) return null;

  // Start at r=52 — the lens barrel radius — so it looks like the iris opens
  // outward from inside the glass rather than appearing from nothing.
  const r = interpolate(frame, [T.lensStart, T.lensEnd], [52, 1650], {
    ...CLAMP,
    easing: SMOOTH,
  });

  const ringOpacity = interpolate(
    frame,
    [T.lensStart, T.lensStart + 25, T.lensEnd - 15, T.lensEnd],
    [0, 0.65, 0.45, 0],
    CLAMP,
  );

  return (
    <AbsoluteFill style={{ pointerEvents: "none" }}>
      <svg
        style={{ position: "absolute", width: "100%", height: "100%" }}
        viewBox="0 0 1920 1080"
        preserveAspectRatio="none"
      >
        <circle cx={LENS_SCR_X} cy={LENS_SCR_Y} r={r} fill="black" />
        <circle
          cx={LENS_SCR_X} cy={LENS_SCR_Y}
          r={r + 10}
          fill="none"
          stroke={`rgba(66,165,245,${ringOpacity.toFixed(2)})`}
          strokeWidth={6}
        />
      </svg>
    </AbsoluteFill>
  );
};

// ── TheDevTeam ─────────────────────────────────────────────────────────────
export const TheDevTeam: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const glowUp = frame >= T.bulbOn;

  // Figures and lightbulb fade out together as camera takes over
  const figGroupOp = interpolate(frame, [T.figFadeStart, T.figFadeEnd], [1, 0], CLAMP);

  return (
    <AbsoluteFill>
      <Background frame={frame} glowUp={glowUp} />

      {/* Figures + bulb layer */}
      <div style={{ opacity: figGroupOp, pointerEvents: "none" }}>
        <DevFigure
          frame={frame}
          fps={fps}
          footX={500}
          footY={1020}
          color="#3949AB"
          hairStyle="curly"
          isTalking={true}
          enterFrom="left"
          appearAt={T.dev1In}
        />
        <DevFigure
          frame={frame}
          fps={fps}
          footX={1420}
          footY={1020}
          color="#00897B"
          hairStyle="short"
          isTalking={false}
          enterFrom="right"
          appearAt={T.dev2In}
        />
        <LightBulb frame={frame} fps={fps} />
      </div>

      {/* Speech bubble (has its own earlier fade-out) */}
      <IdeaBubble frame={frame} fps={fps} />

      {/* Camera slides up, then lens zoom covers everything */}
      <CameraGraphic frame={frame} fps={fps} />
      <LensZoom frame={frame} />
    </AbsoluteFill>
  );
};
