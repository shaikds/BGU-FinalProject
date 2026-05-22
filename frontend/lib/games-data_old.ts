const TEAMS = [
  { home: "מכבי תל אביב", away: "הפועל באר שבע" },
  { home: "מכבי חיפה", away: 'בית"ר ירושלים' },
  { home: "הפועל תל אביב", away: "מכבי נתניה" },
  { home: "בני יהודה", away: "הפועל חיפה" },
  { home: "מכבי פתח תקווה", away: "עירוני קריית שמונה" },
  { home: "הפועל ירושלים", away: "מכבי הרצליה" },
  { home: "סקציה נס ציונה", away: "הפועל עפולה" },
  { home: "מכבי בני ריינה", away: "הפועל ראשון לציון" },
]

export type Game = {
  id: number
  homeTeam: string
  awayTeam: string
  homeScore: number
  awayScore: number
  date: string
  time: string
  players: number
  totalActions: number
}

// Deterministic game generator — same id always returns same data
export function getGameById(id: number): Game {
  const teamPair = TEAMS[(id - 1) % TEAMS.length]
  // Use id as seed for deterministic scores
  const homeScore = (id * 3) % 5
  const awayScore = (id * 7) % 5
  const date = new Date()
  date.setDate(date.getDate() - (id - 1) * 3)

  return {
    id,
    homeTeam: teamPair.home,
    awayTeam: teamPair.away,
    homeScore,
    awayScore,
    date: date.toLocaleDateString("he-IL"),
    time: `${18 + ((id - 1) % 4)}:00`,
    players: 20 + ((id * 2) % 5),
    totalActions: 150 + ((id * 13) % 100),
  }
}

export function generateGames(count: number, startIndex: number): Game[] {
  return Array.from({ length: count }, (_, i) => getGameById(startIndex + i + 1))
}

export function getGameTitle(game: Game): string {
  return `${game.homeTeam} - ${game.awayTeam}`
}
