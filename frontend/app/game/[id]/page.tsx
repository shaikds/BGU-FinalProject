// PLACE THIS FILE AT:  app/game/[id]/page.tsx
"use client"

import { useState, useMemo, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import {
  fetchGame, fetchPlayerStats, fetchGameStats, getGameTitle,
  type SessionDetail, type PlayerData, type GameStats,
} from "@/lib/api"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { ArrowUpDown, Search, TrendingUp, Trophy, Target, ArrowRight, AlertCircle } from "lucide-react"
import { Bar, BarChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import PlayerRankings from "@/components/player-rankings"

const ACTIONS = [
  "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS",
  "THROW IN", "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE",
  "FREE KICK", "GOAL",
] as const

type ActionType     = (typeof ACTIONS)[number]
type SortField      = "id" | ActionType
type SortDirection  = "asc" | "desc" | null

// ── Skeleton while loading ───────────────────────────────────────────────────
function PageSkeleton() {
  return (
    <main className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-[1600px] space-y-6">
        <div className="space-y-4">
          <Skeleton className="h-12 w-96" />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[0,1,2].map((i) => <Skeleton key={i} className="h-28 w-full" />)}
          </div>
        </div>
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-[500px] w-full" />
      </div>
    </main>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────
export default function GameStatsPage() {
  const params = useParams()
  const router = useRouter()
  const gameId = params.id as string

  // ── Remote state
  const [game,       setGame]       = useState<SessionDetail | null>(null)
  const [playerData, setPlayerData] = useState<PlayerData[]>([])
  const [gameStats,  setGameStats]  = useState<GameStats | null>(null)
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState<string | null>(null)

  // ── UI state (unchanged from original)
  const [searchTerm,     setSearchTerm]     = useState("")
  const [sortField,      setSortField]      = useState<SortField | null>(null)
  const [sortDirection,  setSortDirection]  = useState<SortDirection>(null)
  const [hoveredCell,    setHoveredCell]    = useState<string | null>(null)
  const [selectedAction, setSelectedAction] = useState<ActionType>("PASS")
  const [hoveredBar,     setHoveredBar]     = useState<number | null>(null)

  // ── Fetch all three endpoints in parallel
  useEffect(() => {
    if (!gameId) return
    setLoading(true)
    setError(null)
    Promise.all([fetchGame(gameId), fetchPlayerStats(gameId), fetchGameStats(gameId)])
      .then(([g, ps, gs]) => { setGame(g); setPlayerData(ps); setGameStats(gs) })
      .catch((err) => setError(err instanceof Error ? err.message : "שגיאה בטעינת הנתונים"))
      .finally(() => setLoading(false))
  }, [gameId])

  // ── Derived data (same logic as original, now on real data)
  const filteredAndSortedData = useMemo(() => {
    let filtered = playerData.filter((p) => p.id.toString().includes(searchTerm))
    if (sortField && sortDirection) {
      filtered = [...filtered].sort((a, b) => {
        const av = (a[sortField] as number) || 0
        const bv = (b[sortField] as number) || 0
        return sortDirection === "asc" ? av - bv : bv - av
      })
    }
    return filtered
  }, [playerData, searchTerm, sortField, sortDirection])

  const chartData = useMemo(() => {
    return playerData
      .map((p) => ({ id: p.id, value: (p[selectedAction] as number) || 0 }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 15)
  }, [playerData, selectedAction])

  // ── Helpers (identical to original)
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      if (sortDirection === "asc") setSortDirection("desc")
      else { setSortField(null); setSortDirection(null) }
    } else { setSortField(field); setSortDirection("asc") }
  }

  const getColorIntensity = (value: number) => {
    if (value === 0) return "text-muted-foreground"
    if (value <= 3)  return "bg-primary/20 text-primary border border-primary/30"
    if (value <= 7)  return "bg-primary/40 text-primary-foreground border border-primary/50"
    return "bg-primary text-primary-foreground border border-primary font-bold"
  }

  const getBarColor = (index: number) => {
    if (index === hoveredBar) return "oklch(0.75 0.20 90)"
    if (index < 3)  return "oklch(0.88 0.15 135)"
    if (index < 8)  return "oklch(0.70 0.18 150)"
    return "oklch(0.55 0.12 145)"
  }

  // ── Render guards
  if (loading) return <PageSkeleton />

  if (error) {
    return (
      <main className="min-h-screen bg-background p-4 md:p-8 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <p className="text-xl font-semibold">שגיאה בטעינת המשחק</p>
          <p className="text-muted-foreground">{error}</p>
          <Button variant="outline" onClick={() => router.push("/")}>חזרה לרשימת המשחקים</Button>
        </div>
      </main>
    )
  }

  if (!game) return null

  return (
    <main className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-[1600px] space-y-6">

        {/* ── Page header ─────────────────────────────────────────────────── */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-xl bg-primary/20 flex items-center justify-center border-2 border-primary/40">
                <Trophy className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-balance">
                  {getGameTitle(game)}
                </h1>
                <p className="text-muted-foreground text-sm">
                  {game.date} &middot; {game.time} &middot; כיסוי: {game.coverage_pct}% &middot; {game.num_assigned}/{game.num_events} אירועים
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              onClick={() => router.push("/")}
              className="gap-2 border-primary/30 hover:bg-primary/20 hover:text-primary"
            >
              <ArrowRight className="h-4 w-4" />
              חזרה לרשימת המשחקים
            </Button>
          </div>

          {/* ── Summary cards ───────────────────────────────────────────────── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="border-primary/30 bg-card/80 backdrop-blur shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground font-medium">סך הכל פעולות</p>
                    <p className="text-3xl font-bold text-primary">{gameStats?.totalActions ?? "—"}</p>
                  </div>
                  <div className="h-12 w-12 rounded-xl bg-primary/20 flex items-center justify-center">
                    <TrendingUp className="h-6 w-6 text-primary" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-secondary/40 bg-card/80 backdrop-blur shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground font-medium">סך הכל גולים</p>
                    <p className="text-3xl font-bold text-secondary">{gameStats?.totalGoals ?? "—"}</p>
                  </div>
                  <div className="h-12 w-12 rounded-xl bg-secondary/20 flex items-center justify-center">
                    <Target className="h-6 w-6 text-secondary" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-primary/30 bg-card/80 backdrop-blur shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground font-medium">סך הכל מסירות</p>
                    <p className="text-3xl font-bold text-primary">{gameStats?.totalPasses ?? "—"}</p>
                  </div>
                  <div className="h-12 w-12 rounded-xl bg-primary/20 flex items-center justify-center">
                    <TrendingUp className="h-6 w-6 text-primary" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* ── Player rankings ──────────────────────────────────────────────── */}
        <PlayerRankings playerData={playerData} />

        {/* ── Bar chart ────────────────────────────────────────────────────── */}
        <Card className="border-primary/30 bg-card/80 backdrop-blur shadow-lg">
          <CardHeader>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-primary/20 flex items-center justify-center border-2 border-primary/40">
                  <Trophy className="h-5 w-5 text-primary" />
                </div>
                <CardTitle className="text-xl font-semibold">השוואת שחקנים לפי פעולה</CardTitle>
              </div>
              <Select value={selectedAction} onValueChange={(v) => setSelectedAction(v as ActionType)}>
                <SelectTrigger className="w-[200px] bg-background/70 border-primary/30">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ACTIONS.map((a) => <SelectItem key={a} value={a}>{a}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            {chartData.length === 0 ? (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                אין נתונים לפעולה זו
              </div>
            ) : (
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.30 0.05 145)" opacity={0.4} />
                    <XAxis
                      dataKey="id"
                      label={{ value: "מספר שחקן", position: "insideBottom", offset: -10, fill: "oklch(0.90 0.05 145)", fontSize: 14, fontWeight: 600 }}
                      tick={{ fill: "oklch(0.85 0.08 145)", fontSize: 13, fontWeight: 600 }}
                      tickFormatter={(v) => `#${v}`}
                    />
                    <YAxis
                      label={{ value: "מספר פעולות", angle: -90, position: "insideLeft", fill: "oklch(0.90 0.05 145)", fontSize: 14, fontWeight: 600 }}
                      tick={{ fill: "oklch(0.85 0.08 145)", fontSize: 13, fontWeight: 600 }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: "oklch(0.20 0.03 145)", border: "2px solid oklch(0.65 0.15 140)", borderRadius: "8px", padding: "12px" }}
                      labelStyle={{ color: "oklch(0.95 0.05 145)", fontWeight: "bold" }}
                      itemStyle={{ color: "#ffffff" }}
                      cursor={{ fill: "oklch(0.65 0.12 140 / 0.2)" }}
                      formatter={(value: number) => [`${value} פעולות`, selectedAction]}
                      labelFormatter={(label) => `שחקן #${label}`}
                    />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}
                      onMouseEnter={(_, index) => setHoveredBar(index)}
                      onMouseLeave={() => setHoveredBar(null)}
                    >
                      {chartData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={getBarColor(index)} className="transition-all duration-300" />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            <div className="mt-4 flex flex-wrap items-center justify-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: "oklch(0.88 0.15 135)" }} />
                <span>מובילים (1-3)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: "oklch(0.70 0.18 150)" }} />
                <span>בינוניים (4-8)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: "oklch(0.55 0.12 145)" }} />
                <span>אחרים</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ── Player actions table ─────────────────────────────────────────── */}
        <Card className="border-primary/30 bg-card/80 backdrop-blur shadow-lg">
          <CardHeader>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <CardTitle className="text-xl font-semibold">פירוט פעולות לפי שחקן</CardTitle>
              <div className="relative max-w-sm">
                <Search className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="חיפוש לפי מספר שחקן..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pr-10 bg-background/70 border-primary/30"
                />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-lg border border-border/50">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50 hover:bg-muted/50 border-b-2 border-primary/30">
                    <TableHead className="sticky right-0 bg-muted/50 z-20 min-w-[100px]">
                      <Button variant="ghost" onClick={() => handleSort("id")} className="h-8 font-bold hover:bg-primary/20 hover:text-primary">
                        ID שחקן <ArrowUpDown className="mr-2 h-4 w-4" />
                      </Button>
                    </TableHead>
                    {ACTIONS.map((action) => (
                      <TableHead key={action} className="text-center min-w-[140px]">
                        <Button variant="ghost" onClick={() => handleSort(action)} className="h-8 whitespace-nowrap w-full hover:bg-primary/20 hover:text-primary font-medium">
                          {action} <ArrowUpDown className="mr-2 h-3 w-3" />
                        </Button>
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredAndSortedData.map((player, index) => (
                    <TableRow
                      key={player.id}
                      className="hover:bg-muted/20 transition-colors border-b border-border/30"
                      style={{ animationDelay: `${index * 20}ms` }}
                    >
                      <TableCell className="font-bold sticky right-0 bg-card z-10 border-l border-border/50">
                        <div className="flex items-center gap-2">
                          <div className="h-8 w-8 rounded-lg bg-primary/20 flex items-center justify-center text-primary font-bold border-2 border-primary/40">
                            {player.id}
                          </div>
                        </div>
                      </TableCell>
                      {ACTIONS.map((action) => {
                        const value   = (player[action] as number) || 0
                        const cellKey = `${player.id}-${action}`
                        return (
                          <TableCell
                            key={action}
                            className="text-center"
                            onMouseEnter={() => setHoveredCell(cellKey)}
                            onMouseLeave={() => setHoveredCell(null)}
                          >
                            <div className={`inline-flex items-center justify-center min-w-[48px] h-10 px-3 rounded-lg transition-all duration-200 ${getColorIntensity(value)} ${hoveredCell === cellKey ? "scale-110 shadow-lg" : ""}`}>
                              {value}
                            </div>
                          </TableCell>
                        )
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {filteredAndSortedData.length === 0 && (
              <div className="text-center py-12 text-muted-foreground">
                {searchTerm ? "לא נמצאו שחקנים התואמים את החיפוש" : "אין נתוני שחקנים למשחק זה"}
              </div>
            )}
          </CardContent>
        </Card>

      </div>
    </main>
  )
}
