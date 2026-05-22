"use client"

import { useMemo, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Trophy, Shield, Zap, Scale } from "lucide-react"

// Action categorization based on soccer knowledge
const ACTION_CATEGORIES = {
  ATTACKING: [
    "SHOT",
    "GOAL",
    "CROSS",
    "DRIVE",
    "HIGH PASS",
    // "FREE KICK", // Can be attacking
  ],
  DEFENSIVE: [
    "PLAYER SUCCESSFUL TACKLE",
    "BALL PLAYER BLOCK",
    , // Can be defensive
  ],
  NEUTRAL: ["PASS", "THROW IN", "OUT", "HEADER"],
} as const

type PlayerData = Record<string, number>

interface PlayerRankingsProps {
  playerData: PlayerData[]
}

interface RankedPlayer {
  id: number
  score: number
  attacking: number
  defensive: number
  neutral: number
}

type RankingType = "attacking" | "defensive" | "weighted"

export default function PlayerRankings({ playerData }: PlayerRankingsProps) {
  const [selectedRanking, setSelectedRanking] = useState<RankingType>("weighted")

  // Calculate rankings
  const rankings = useMemo(() => {
    const calculateScore = (player: PlayerData, actions: readonly string[]) => {
      return actions.reduce((sum, action) => sum + (player[action] || 0), 0)
    }

    const playerStats = playerData.map((player) => {
      const attacking = calculateScore(player, ACTION_CATEGORIES.ATTACKING)
      const defensive = calculateScore(player, ACTION_CATEGORIES.DEFENSIVE)
      const neutral = calculateScore(player, ACTION_CATEGORIES.NEUTRAL)

      return {
        id: player.id as number,
        attacking,
        defensive,
        neutral,
      }
    })

    // Calculate attacking scores
    const attackingScores = [...playerStats]
      .map((player) => ({
        ...player,
        score: player.attacking,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)

    // Calculate defensive scores
    const defensiveScores = [...playerStats]
      .map((player) => ({
        ...player,
        score: player.defensive,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)

    // Calculate balanced (50/50) scores - weighted combination
    const balancedScores = [...playerStats]
      .map((player) => {
        // Balance score: prefer players with both attacking and defensive actions
        const balance = Math.min(player.attacking, player.defensive) * 2 + player.neutral * 0.5

        return {
          ...player,
          score: balance,
        }
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)

    return { attacking: attackingScores, defensive: defensiveScores, weighted: balancedScores }
  }, [playerData])

  const getPodiumStyles = (rank: number) => {
    switch (rank) {
      case 0: // 1st place
        return {
          container:
            "bg-gradient-to-br from-yellow-500/20 to-yellow-600/30 border-yellow-500 border-2 shadow-[0_0_28px_6px_rgba(234,179,8,0.55),0_0_8px_2px_rgba(234,179,8,0.8)]",
          badge: "bg-yellow-500 text-yellow-950 border-yellow-600",
          medal: "🥇",
          height: "h-40",
        }
      case 1: // 2nd place
        return {
          container:
            "bg-gradient-to-br from-gray-300/20 to-gray-400/30 border-gray-400 border-2 shadow-[0_0_28px_6px_rgba(156,163,175,0.5),0_0_8px_2px_rgba(200,200,200,0.7)]",
          badge: "bg-gray-300 text-gray-900 border-gray-400",
          medal: "🥈",
          height: "h-36",
        }
      case 2: // 3rd place
        return {
          container:
            "bg-gradient-to-br from-orange-600/20 to-orange-700/30 border-orange-600 border-2 shadow-[0_0_28px_6px_rgba(234,88,12,0.5),0_0_8px_2px_rgba(234,88,12,0.8)]",
          badge: "bg-orange-600 text-orange-100 border-orange-700",
          medal: "🥉",
          height: "h-32",
        }
      default:
        return {
          container: "bg-card/80 border-border",
          badge: "bg-primary text-primary-foreground",
          medal: "",
          height: "h-28",
        }
    }
  }

  const currentRanking = rankings[selectedRanking]
  const getRankingConfig = () => {
    switch (selectedRanking) {
      case "attacking":
        return { title: "תקיפה מובילה", icon: <Zap className="h-5 w-5" />, color: "border-red-500/40" }
      case "defensive":
        return { title: "הגנה מובילה", icon: <Shield className="h-5 w-5" />, color: "border-blue-500/40" }
      case "weighted":
        return { title: "משחק מאוזן", icon: <Scale className="h-5 w-5" />, color: "border-purple-500/40" }
    }
  }

  const config = getRankingConfig()

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="h-12 w-12 rounded-xl bg-secondary/20 flex items-center justify-center border-2 border-secondary/40">
          <Trophy className="h-6 w-6 text-secondary" />
        </div>
        <div>
          <h2 className="text-2xl font-bold tracking-tight">דירוג שחקנים מובילים</h2>
          <p className="text-sm text-muted-foreground">השחקנים המצטיינים בכל קטגוריה</p>
        </div>
      </div>

      <div className="flex items-center justify-center gap-2">
        <button
          onClick={() => setSelectedRanking("attacking")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-all ${
            selectedRanking === "attacking"
              ? "bg-red-500/20 border-red-500/60 text-foreground"
              : "border-border hover:border-red-500/40 text-muted-foreground hover:text-foreground"
          }`}
        >
          <Zap className="h-4 w-4" />
          <span className="text-sm font-medium">תקיפה</span>
        </button>
        <button
          onClick={() => setSelectedRanking("defensive")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-all ${
            selectedRanking === "defensive"
              ? "bg-blue-500/20 border-blue-500/60 text-foreground"
              : "border-border hover:border-blue-500/40 text-muted-foreground hover:text-foreground"
          }`}
        >
          <Shield className="h-4 w-4" />
          <span className="text-sm font-medium">הגנה</span>
        </button>
        <button
          onClick={() => setSelectedRanking("weighted")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-all ${
            selectedRanking === "weighted"
              ? "bg-purple-500/20 border-purple-500/60 text-foreground"
              : "border-border hover:border-purple-500/40 text-muted-foreground hover:text-foreground"
          }`}
        >
          <Scale className="h-4 w-4" />
          <span className="text-sm font-medium">מאוזן</span>
        </button>
      </div>

      <Card
        className={`${config.color} bg-card/80 backdrop-blur shadow-lg ${
          selectedRanking === "attacking"
            ? "shadow-[0_0_30px_8px_rgba(239,68,68,0.2)]"
            : selectedRanking === "defensive"
              ? "shadow-[0_0_30px_8px_rgba(59,130,246,0.2)]"
              : "shadow-[0_0_30px_8px_rgba(147,51,234,0.2)]"
        }`}
      >
        <CardHeader>
          <div className="flex items-center gap-3">
            <div
              className={`h-10 w-10 rounded-lg bg-primary/20 flex items-center justify-center border-2 border-primary/40`}
            >
              {config.icon}
            </div>
            <CardTitle className="text-xl font-semibold">{config.title}</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          {/* Podium cards */}
          <div className="flex items-end justify-center gap-6">
            {currentRanking.map((player, index) => {
              const styles = getPodiumStyles(index)
              return (
                <div
                  key={player.id}
                  className="flex flex-col items-center flex-1 max-w-[180px] transition-all duration-300 hover:scale-105"
                >
                  <div
                    className={`${styles.container} ${styles.height} w-full rounded-xl p-4 flex flex-col items-center justify-center gap-3 backdrop-blur transition-all duration-300`}
                  >
                    <div className="text-3xl leading-none">{styles.medal}</div>
                    <Badge className={`${styles.badge} text-base font-bold px-3 py-1 border-2`}>#{player.id}</Badge>
                    <div className="text-2xl font-bold text-primary leading-none">{player.score.toFixed(1)}</div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Shared sub-text row — one column per player, same horizontal row */}
          <div className="flex justify-center gap-6 mt-3">
            {currentRanking.map((player) => (
              <div
                key={`stats-${player.id}`}
                className="flex-1 max-w-[180px] flex flex-col items-center gap-0.5"
              >
                <span className="text-[11px] text-muted-foreground">פעולות: {player.score.toFixed(0)}</span>
                <span className="text-[11px] text-muted-foreground">תקיפה: {player.attacking}</span>
                <span className="text-[11px] text-muted-foreground">הגנה: {player.defensive}</span>
                <span className="text-[11px] text-muted-foreground">ניטרלי: {player.neutral}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="border-primary/30 bg-card/60 backdrop-blur">
        <CardContent className="p-3">
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="font-semibold text-foreground">תקיפה:</div>
              <div className="text-muted-foreground">בעיטות, גולים, כניסות, מסירות גבוהות</div>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="font-semibold text-foreground">הגנה:</div>
              <div className="text-muted-foreground">תקיפות מוצלחות, חסימות, נגיחות</div>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="font-semibold text-foreground">מאוזן:</div>
              <div className="text-muted-foreground">שילוב של תקיפה והגנה</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
