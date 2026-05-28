import "./index.css";
import { Composition } from "remotion";
import { SoccerIntro, FPS, DURATION_SECONDS } from "./SoccerIntro";
import { SoccerIntroSticks } from "./SoccerIntroSticks";
import { SoccerMainActor } from "./SoccerMainActor";
import { TheProblem } from "./TheProblem";
import { TheSolution } from "./TheSolution";
import { ThePipeline } from "./ThePipeline";
import { TheResult } from "./TheResult";
import { TheFrustration } from "./TheFrustration";
import { TheScout } from "./TheScout";
import { TheDevTeam } from "./TheDevTeam";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SoccerIntro"
        component={SoccerIntro}
        durationInFrames={FPS * DURATION_SECONDS}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="SoccerIntroSticks"
        component={SoccerIntroSticks}
        durationInFrames={FPS * DURATION_SECONDS}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="SoccerMainActor"
        component={SoccerMainActor}
        durationInFrames={FPS * DURATION_SECONDS}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="TheProblem"
        component={TheProblem}
        durationInFrames={450}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="TheSolution"
        component={TheSolution}
        durationInFrames={600}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="ThePipeline"
        component={ThePipeline}
        durationInFrames={480}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="TheResult"
        component={TheResult}
        durationInFrames={750}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="TheFrustration"
        component={TheFrustration}
        durationInFrames={520}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="TheScout"
        component={TheScout}
        durationInFrames={570}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="TheDevTeam"
        component={TheDevTeam}
        durationInFrames={440}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
