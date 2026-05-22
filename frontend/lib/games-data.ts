/**
 * lib/games-data.ts
 * Re-exports from api.ts so existing imports keep working.
 * All mock data generators have been removed — data comes from the backend.
 */
export type { Game } from "@/lib/api"
export { fetchGames, fetchGame, getGameTitle } from "@/lib/api"
