/**
 * lib/api.ts
 * Typed API client — connects the Next.js frontend to the FastAPI backend.
 * Set NEXT_PUBLIC_API_URL in .env.local (default: http://localhost:8000)
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ── Types ────────────────────────────────────────────────────────────────────

export type Game = {
  id: string          // MongoDB ObjectId string
  homeTeam: string
  awayTeam: string
  homeScore: number
  awayScore: number
  date: string
  time: string
  players: number
  totalActions: number
}

export type SessionDetail = Game & {
  video: string
  fps: number
  num_events: number
  num_assigned: number
  coverage_pct: number
  uploaded_at: string
  parameters: {
    window_size: number
    min_confidence: number
    include_keepers: boolean
    distance_weighting: boolean
    temporal_sigma: number
  }
}

/** One row in the player-actions table. key = action type | "id", value = count */
export type PlayerData = { id: number } & Record<string, number>

export type GameStats = {
  totalActions: number
  totalGoals: number
  totalPasses: number
}

// ── API calls ────────────────────────────────────────────────────────────────

export async function fetchGames(skip = 0, limit = 20): Promise<Game[]> {
  return apiFetch<Game[]>(`/sessions?skip=${skip}&limit=${limit}`)
}

export async function fetchGame(id: string): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`/sessions/${id}`)
}

/**
 * Returns per-player action counts:
 *   [{ id: 18, PASS: 3, DRIVE: 7, GOAL: 0, ... }, ...]
 * All 12 action types are always present (0 if none).
 */
export async function fetchPlayerStats(id: string): Promise<PlayerData[]> {
  return apiFetch<PlayerData[]>(`/sessions/${id}/player-stats`)
}

export async function fetchGameStats(id: string): Promise<GameStats> {
  return apiFetch<GameStats>(`/sessions/${id}/stats`)
}

export function getGameTitle(game: Pick<Game, "homeTeam" | "awayTeam">): string {
  return `${game.homeTeam} - ${game.awayTeam}`
}
