"""Central configuration for NCAAB prediction pipeline."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

# ── Years ──────────────────────────────────────────────────────────────────
SCRAPE_YEARS = [y for y in range(2015, 2027) if y != 2020]

# Season string for TeamRankings dropdown (e.g. "2024-2025" for the 2025 tournament)
def season_string(year: int) -> str:
    return f"{year - 1}-{year}"

# Stats cutoff date: day before First Four starts each year
# Used as ?date= parameter on TeamRankings
TOURNAMENT_DATES = {
    2015: "2025-03-16",  # corrected below
    2016: "2016-03-14",
    2017: "2017-03-13",
    2018: "2018-03-12",
    2019: "2019-03-18",
    2021: "2021-03-17",
    2022: "2022-03-14",
    2023: "2023-03-13",
    2024: "2024-03-18",
    2025: "2025-03-17",
    2026: "2026-03-16",
}
# Fix 2015
TOURNAMENT_DATES[2015] = "2015-03-16"

# ── TeamRankings ───────────────────────────────────────────────────────────
TEAMRANKINGS_BASE_URL = "https://www.teamrankings.com/ncaa-basketball/stat/"
TEAMRANKINGS_STATS_INDEX = "https://www.teamrankings.com/ncb/stats/"

# Curated stat slugs - ball control focus + offense/defense efficiency
# These map to URLs like: {TEAMRANKINGS_BASE_URL}{slug}?date=YYYY-MM-DD
TEAMRANKINGS_STAT_SLUGS = [
    # Ball control (PRIMARY FOCUS)
    "assist--per--turnover-ratio",
    "assists-per-game",
    "turnovers-per-game",
    "turnovers-per-possession",
    "turnover-pct",
    "assists-per-possession",
    "steals-per-game",
    "steal-pct",

    # Offensive efficiency
    "offensive-efficiency",
    "points-per-game",
    "effective-field-goal-pct",
    "true-shooting-percentage",
    "shooting-pct",
    "three-point-pct",
    "two-point-pct",
    "free-throw-pct",
    "free-throw-rate",
    "floor-percentage",

    # Defensive efficiency
    "defensive-efficiency",
    "opponent-points-per-game",
    "opponent-effective-field-goal-pct",
    "opponent-three-point-pct",
    "opponent-two-point-pct",
    "opponent-turnovers-per-game",
    "opponent-turnover-pct",
    "opponent-free-throw-rate",
    "opponent-assists-per-game",

    # Rebounding
    "total-rebounds-per-game",
    "offensive-rebounds-per-game",
    "defensive-rebounds-per-game",
    "total-rebounding-percentage",
    "offensive-rebounding-pct",
    "defensive-rebounding-pct",

    # Blocks
    "blocks-per-game",
    "block-pct",

    # Scoring margin & tempo
    "average-scoring-margin",
    "possessions-per-game",

    # Win metrics
    "win-pct-all-games",
    "win-pct-close-games",

    # Extra opponent stats
    "opponent-steals-per-game",
    "opponent-blocks-per-game",
    "opponent-offensive-rebounds-per-game",
]

# ── Warren Nolan ───────────────────────────────────────────────────────────
WARRENNOLAN_CONF_URL = "https://www.warrennolan.com/basketball/{year}/net-conference"

# ── NCAA API ───────────────────────────────────────────────────────────────
NCAA_API_BASE = "https://ncaa-api.henrygd.me"

# ── Scraping ───────────────────────────────────────────────────────────────
REQUEST_DELAY = 2.0  # seconds between TeamRankings requests
NCAA_API_DELAY = 0.3  # seconds between NCAA API requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
