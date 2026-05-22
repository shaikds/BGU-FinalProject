"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent } from "@/components/ui/card"
import { Calendar, Clock, Users, Trophy, ChevronLeft, Loader2, AlertCircle } from "lucide-react"
import { fetchGames, type Game } from "@/lib/api"

const PAGE_SIZE = 10

export default function GamesListPage() {
  const router = useRouter()
  const [games, setGames]     = useState<Game[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [error, setError]     = useState<string | null>(null)
  const observerRef = useRef<HTMLDivElement>(null)
  const skipRef     = useRef(0)

  const loadMoreGames = useCallback(async () => {
    if (isLoading || !hasMore) return
    setIsLoading(true)
    setError(null)
    try {
      const newGames = await fetchGames(skipRef.current, PAGE_SIZE)
      skipRef.current += newGames.length
      setGames((prev) => {
        const ids   = new Set(prev.map((g) => g.id))
        const fresh = newGames.filter((g) => !ids.has(g.id))
        return [...prev, ...fresh]
      })
      if (newGames.length < PAGE_SIZE) setHasMore(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בטעינת המשחקים")
      setHasMore(false)
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, hasMore])

  // Initial load
  useEffect(() => { loadMoreGames() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Infinite scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting && hasMore && !isLoading) loadMoreGames() },
      { threshold: 0.1 }
    )
    if (observerRef.current) observer.observe(observerRef.current)
    return () => observer.disconnect()
  }, [loadMoreGames, hasMore, isLoading])

  return (
    <main className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-4xl space-y-6">

        {/* Header */}
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-xl bg-primary/20 flex items-center justify-center border-2 border-primary/40">
              <Trophy className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl md:text-4xl font-bold tracking-tight">רשימת משחקים</h1>
              <p className="text-muted-foreground">בחר משחק לצפייה בסטטיסטיקות מפורטות</p>
            </div>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="flex items-center gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-destructive">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <div>
              <p className="font-medium">שגיאה בחיבור לשרת</p>
              <p className="text-sm opacity-80">{error}</p>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isLoading && !error && games.length === 0 && (
          <div className="flex flex-col items-center gap-4 py-20 text-muted-foreground">
            <Trophy className="h-12 w-12 opacity-30" />
            <p className="text-lg">אין משחקים עדיין</p>
            <p className="text-sm">העלה קובץ JSON מ-event_player_linker.py לשרת כדי להתחיל</p>
          </div>
        )}

        {/* Game cards */}
        <div className="space-y-4">
          {games.map((game, index) => (
            <Card
              key={game.id}
              onClick={() => router.push(`/game/${game.id}`)}
              className="border-primary/20 bg-card/80 backdrop-blur shadow-lg cursor-pointer transition-all duration-300 hover:scale-[1.02] hover:shadow-[0_0_30px_8px_rgba(136,204,102,0.15)] hover:border-primary/50 group"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <CardContent className="p-5">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">

                  {/* Teams + score */}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                      <Calendar className="h-4 w-4" />
                      <span>{game.date}</span>
                      <Clock className="h-4 w-4 mr-2" />
                      <span>{game.time}</span>
                    </div>
                    <div className="flex items-center justify-center gap-4">
                      <div className="flex-1 text-left">
                        <p className="text-lg font-semibold truncate">{game.homeTeam}</p>
                      </div>
                      <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-muted/50 border border-primary/20">
                        <span className="text-2xl font-bold text-primary">{game.homeScore}</span>
                        <span className="text-muted-foreground">-</span>
                        <span className="text-2xl font-bold text-primary">{game.awayScore}</span>
                      </div>
                      <div className="flex-1 text-right">
                        <p className="text-lg font-semibold truncate">{game.awayTeam}</p>
                      </div>
                    </div>
                  </div>

                  {/* Stats preview */}
                  <div className="flex items-center gap-6 md:border-r md:border-border/50 md:pr-6">
                    <div className="text-center">
                      <div className="flex items-center gap-1 text-muted-foreground mb-1">
                        <Users className="h-4 w-4" />
                        <span className="text-xs">שחקנים</span>
                      </div>
                      <p className="text-xl font-bold text-primary">{game.players}</p>
                    </div>
                    <div className="text-center">
                      <div className="flex items-center gap-1 text-muted-foreground mb-1">
                        <Trophy className="h-4 w-4" />
                        <span className="text-xs">פעולות</span>
                      </div>
                      <p className="text-xl font-bold text-secondary">{game.totalActions}</p>
                    </div>
                    <ChevronLeft className="h-6 w-6 text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>

                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Infinite scroll sentinel */}
        <div ref={observerRef} className="py-8 flex justify-center">
          {isLoading && (
            <div className="flex items-center gap-3 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <span>טוען משחקים נוספים...</span>
            </div>
          )}
          {!hasMore && games.length > 0 && (
            <p className="text-muted-foreground text-sm">הגעת לסוף הרשימה</p>
          )}
        </div>
      </div>
    </main>
  )
}
