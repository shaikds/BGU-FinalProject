import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { F, SVG_HALF } from "./TheFrustration";
import { Field, SMOOTH } from "./SoccerIntro";
import { PERSPECTIVE_PX, TILT_DEG } from "./SoccerIntroSticks";

// ── Helpers ────────────────────────────────────────────────────────────────
const CLAMP = {
  extrapolateLeft: "clamp" as const,
  extrapolateRight: "clamp" as const,
};
// Accelerating ease for the lens zoom — starts slow, rushes at the end
const EASE_IN = Easing.bezier(0.5, 0, 0.9, 1);

// (No top-down pitch constants — the background reuses the 2.5D Field from SoccerIntro.)

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

// ── Background — 2.5D football pitch (same technique as TheFrustration) ────
// Uses the same Field + rotateX perspective wrapper so the scene matches
// the rest of the video series.
const Background: React.FC<{ frame: number; glowUp: boolean }> = ({ frame, glowUp }) => {
  const glowOp = glowUp
    ? interpolate(frame, [T.bulbOn, T.bulbOn + 20], [0, 0.55], CLAMP)
    : 0;

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* ── 2.5D perspective container (mirrors SoccerSticksVisuals) ── */}
      <AbsoluteFill style={{
        perspective: `${PERSPECTIVE_PX}px`,
        perspectiveOrigin: "50% 30%",
      }}>
        <AbsoluteFill style={{
          transformStyle: "preserve-3d",
          transform: `rotateX(${TILT_DEG}deg)`,
          transformOrigin: "50% 50%",
        }}>
          {/* Static field only — no animated players */}
          <Field width={1920} height={1080} />
        </AbsoluteFill>
      </AbsoluteFill>

      {/* Slight darkening overlay to set mood */}
      <div style={{
        position: "absolute", inset: 0,
        background: "rgba(0,0,0,0.18)",
        pointerEvents: "none",
      }} />

      {/* Warm bulb glows (flat, screen-space, one per figure) */}
      {glowOp > 0.01 && (
        <>
          <div style={{
            position: "absolute", inset: 0, pointerEvents: "none",
            background: `radial-gradient(circle 330px at 500px 600px, rgba(255,220,50,${glowOp.toFixed(2)}) 0%, transparent 100%)`,
          }} />
          <div style={{
            position: "absolute", inset: 0, pointerEvents: "none",
            background: `radial-gradient(circle 330px at 1420px 600px, rgba(255,220,50,${glowOp.toFixed(2)}) 0%, transparent 100%)`,
          }} />
        </>
      )}

      {/* Vignette */}
      <div style={{
        position: "absolute", inset: 0,
        background: "radial-gradient(ellipse 90% 85% at 50% 50%, transparent 55%, rgba(0,0,0,0.55) 100%)",
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
  // Which way to shift pupils so the figure looks toward the speech bubble
  lookDir:    "left" | "right";
}

const DevFigure: React.FC<DevFigureProps> = ({
  frame, fps, footX, footY, color, hairStyle, isTalking, enterFrom, appearAt, lookDir,
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

  // Static pose — no body or arm sway
  const bodyShiftX = 0;

  // Arms relaxed at sides, fixed position
  const rax = F.shoulderArmX + 40;
  const ray = F.hipY - 20;
  const lax = -(F.shoulderArmX + 40);
  const lay = F.hipY - 20;

  // Pupil shift so each figure looks toward the speech bubble (up + toward centre)
  // Bubble centre: screen (960, 205). Dev1 head: (500,600) → look right+up.
  // Dev2 head: (1420,600) → look left+up. Shift ≈ 4px horizontal, 3px up.
  const pxShift = lookDir === "right" ? 4 : -4;
  const pyShift = -3; // up toward bubble

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

            {/* Eyes — round, friendly; pupils shifted toward speech bubble */}
            <ellipse cx={-26} cy={F.headY - 12} rx={14} ry={11} fill="white" />
            <circle  cx={-24 + pxShift} cy={F.headY - 10 + pyShift} r={6}  fill="#111" />
            <circle  cx={-20 + pxShift} cy={F.headY - 14 + pyShift} r={3}  fill="rgba(255,255,255,0.7)" />

            <ellipse cx={ 26} cy={F.headY - 12} rx={14} ry={11} fill="white" />
            <circle  cx={ 28 + pxShift} cy={F.headY - 10 + pyShift} r={6}  fill="#111" />
            <circle  cx={ 32 + pxShift} cy={F.headY - 14 + pyShift} r={3}  fill="rgba(255,255,255,0.7)" />

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
// Reusable — pass `scrX` for the horizontal centre on screen.
// Both figures share footY=1020, so bulb screen-Y is always 460
// (head at 1020−420=600, bulb 140px above head, SVG bulb cy=60 → div top=400).
const BULB_SCR_Y = 460;

const LightBulb: React.FC<{ frame: number; fps: number; scrX: number }> = ({
  frame, fps, scrX,
}) => {
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
        left: scrX - 60,
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
// Handheld consumer camcorder silhouette matching the reference icon:
// large circular lens on the left, rounded-rect body to the right,
// grip handle below. SVG 360×260.
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
  const glowOp  = preGlow * 0.85;

  // REC light blinks every ~20 frames
  const recOn    = frame >= T.cameraIn + 15;
  const recBlink = recOn ? (Math.floor(frame / 20) % 2 === 0 ? 1 : 0.2) : 0;

  return (
    <div
      style={{
        position: "absolute",
        left: CAM_DIV_L,
        top:  CAM_DIV_T,
        width:  360,
        height: 260,
        transform: `translateY(${enterY}px)`,
        pointerEvents: "none",
      }}
    >
      <svg width={360} height={260} style={{ overflow: "visible" }}>
        <defs>
          {/* Deep lens gradient — pure black at outer edge for clean fade-to-black when zooming */}
          <radialGradient id="vcam-lens" cx="38%" cy="35%" r="70%">
            <stop offset="0%"   stopColor="#0c1840" />
            <stop offset="45%"  stopColor="#030810" />
            <stop offset="75%"  stopColor="#000000" />
            <stop offset="100%" stopColor="#000000" />
          </radialGradient>
        </defs>

        {/* ── Body (rounded-rect to the right of lens) ── */}
        <rect x={108} y={18} width={230} height={182} rx={22}
          fill="#1a1a2e" stroke="#42A5F5" strokeWidth={3} />

        {/* Body zoom slider slot */}
        <rect x={162} y={40} width={128} height={11} rx={5}
          fill="#0d1117" stroke="#42A5F5" strokeWidth={1.5} />
        <rect x={162} y={40} width={52} height={11} rx={5}
          fill="rgba(66,165,245,0.45)" />

        {/* Small button on body */}
        <circle cx={308} cy={62} r={8}
          fill="#0d1117" stroke="#42A5F5" strokeWidth={1.5} />
        <circle cx={308} cy={62} r={3} fill="#42A5F5" />

        {/* ── Grip / handle (bottom-left, the icon's distinctive feature) ── */}
        <rect x={80} y={183} width={96} height={74} rx={16}
          fill="#141420" stroke="#42A5F5" strokeWidth={2.5} />
        {[198, 211, 224, 237].map((y, i) => (
          <line key={i} x1={88} y1={y} x2={168} y2={y}
            stroke="rgba(66,165,245,0.2)" strokeWidth={1} />
        ))}

        {/* ── Lens assembly (dominant left feature, matching icon shape) ── */}

        {/* Pre-zoom glow halo */}
        {glowOp > 0.01 && (
          <circle cx={115} cy={115} r={100 + preGlow * 35}
            fill="none"
            stroke={`rgba(66,190,255,${glowOp.toFixed(2)})`}
            strokeWidth={10}
          />
        )}

        {/* Outer bezel ring */}
        <circle cx={115} cy={115} r={93}
          fill="#0f0f1e" stroke="#42A5F5" strokeWidth={3} />

        {/* Inner bezel */}
        <circle cx={115} cy={115} r={82}
          fill="#0b0b18" stroke="rgba(66,140,220,0.45)" strokeWidth={2} />

        {/* Lens glass layers — progressively darker → pure black at edge
            so the scene-scale zoom ends in a pure-black screen             */}
        <circle cx={115} cy={115} r={70} fill="url(#vcam-lens)" />
        <circle cx={115} cy={115} r={54}
          fill="#030610" stroke="rgba(60,130,220,0.2)" strokeWidth={1} />
        <circle cx={115} cy={115} r={37}
          fill="#020408" />
        <circle cx={115} cy={115} r={21}
          fill="#010204" />
        <circle cx={115} cy={115} r={9}
          fill="#000000" />

        {/* Glare highlight */}
        <circle cx={90} cy={88} r={7}
          fill="rgba(255,255,255,0.6)" />
        <path d="M 83,96 Q 76,110 80,118"
          stroke="rgba(255,255,255,0.38)"
          strokeWidth={4} fill="none" strokeLinecap="round" />

        {/* ── REC indicator ── */}
        <circle cx={184} cy={168} r={6}
          fill="#ff1744" stroke="#b71c1c" strokeWidth={1.5}
          opacity={recBlink} />
        {recBlink > 0.5 && (
          <circle cx={184} cy={168} r={12}
            fill="none" stroke="rgba(255,23,68,0.3)" strokeWidth={3} />
        )}
      </svg>
    </div>
  );
};
// LensZoom is removed — the scene-scale zoom in TheDevTeam replaces it.

// ── TheDevTeam ─────────────────────────────────────────────────────────────
export const TheDevTeam: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const glowUp = frame >= T.bulbOn;

  // Figures and lightbulb fade out together as camera takes over
  const figGroupOp = interpolate(frame, [T.figFadeStart, T.figFadeEnd], [1, 0], CLAMP);

  // ── Zoom-into-lens transition ──────────────────────────────────────────────
  // The entire scene scales up centred on the camera lens (960, 490).
  // This creates the "flying into the lens" dolly effect — the camera rushes
  // toward the viewer until the dark glass fills and blacks out the screen.
  // The lens glass gradient ends in pure black, so the transition is clean.
  const zoomScale = interpolate(frame, [T.lensStart, T.lensEnd], [1, 22], {
    ...CLAMP,
    easing: EASE_IN,
  });
  const zoomStyle = frame >= T.lensStart
    ? {
        transform: `scale(${zoomScale})`,
        transformOrigin: `${LENS_SCR_X}px ${LENS_SCR_Y}px`,
      }
    : {};

  return (
    <AbsoluteFill style={zoomStyle}>
      <Background frame={frame} glowUp={glowUp} />

      {/* Figures + bulb layer */}
      <div style={{ opacity: figGroupOp, pointerEvents: "none" }}>
        {/* Dev1 — indigo, curly hair, left side, talking, looks right toward bubble */}
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
          lookDir="right"
        />
        {/* Dev2 — teal, short hair, right side, listening, looks left toward bubble */}
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
          lookDir="left"
        />
        <LightBulb frame={frame} fps={fps} scrX={500} />
        <LightBulb frame={frame} fps={fps} scrX={1420} />
      </div>

      {/* Speech bubble (has its own earlier fade-out) */}
      <IdeaBubble frame={frame} fps={fps} />

      {/* Camera slides up; at T.lensStart the entire scene scales into its lens */}
      <CameraGraphic frame={frame} fps={fps} />
    </AbsoluteFill>
  );
};
