"""Fetch tournament games and schools index from NCAA API."""
import json
import time

import polars as pl
import requests

import config


def fetch_schools_index() -> dict:
    """Fetch the schools index from NCAA API. Returns raw JSON."""
    cache_path = config.RAW_DIR / "ncaa_api" / "schools_index.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = f"{config.NCAA_API_BASE}/schools-index"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data


def fetch_tournament_games(year: int) -> pl.DataFrame | None:
    """Fetch all tournament games for a given year from NCAA API scoreboard."""
    games = []

    # Tournament window: March 13-31, April 1-10
    date_ranges = []
    for day in range(13, 32):
        date_ranges.append(f"{year}/{3:02d}/{day:02d}")
    for day in range(1, 11):
        date_ranges.append(f"{year}/{4:02d}/{day:02d}")

    for date_str in date_ranges:
        url = f"{config.NCAA_API_BASE}/scoreboard/basketball-men/d1/{date_str}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            continue

        for entry in data.get("games", []):
            g = entry.get("game", {})
            bracket_round = g.get("bracketRound", "")
            if not bracket_round:
                continue
            if g.get("gameState") != "final":
                continue

            away = g.get("away", {})
            home = g.get("home", {})

            # Extract seeds safely
            away_seed = None
            home_seed = None
            try:
                if away.get("seed"):
                    away_seed = int(away["seed"])
            except (ValueError, TypeError):
                pass
            try:
                if home.get("seed"):
                    home_seed = int(home["seed"])
            except (ValueError, TypeError):
                pass

            games.append({
                "game_id": g.get("gameID", ""),
                "date": g.get("startDate", date_str),
                "bracket_round": bracket_round,
                "bracket_region": g.get("bracketRegion", ""),
                "away_name": away.get("names", {}).get("short", ""),
                "away_seo": away.get("names", {}).get("seo", ""),
                "away_seed": away_seed,
                "away_score": int(away.get("score", 0)),
                "away_winner": away.get("winner", False),
                "home_name": home.get("names", {}).get("short", ""),
                "home_seo": home.get("names", {}).get("seo", ""),
                "home_seed": home_seed,
                "home_score": int(home.get("score", 0)),
                "home_winner": home.get("winner", False),
            })

        time.sleep(config.NCAA_API_DELAY)

    if not games:
        print(f"  [WARN] No tournament games found for {year}")
        return None

    return pl.DataFrame(games)


def run(years: list[int] | None = None):
    """Main entry point: fetch tournament games for all years."""
    years = years or config.SCRAPE_YEARS

    # Fetch schools index first
    print("[NCAA API] Fetching schools index...")
    try:
        schools = fetch_schools_index()
        print(f"  -> {len(schools)} schools cached")
    except Exception as e:
        print(f"  -> Failed: {e}")

    # Fetch tournament games per year
    out_dir = config.RAW_DIR / "ncaa_api" / "tournament_games"
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        out_path = out_dir / f"{year}.parquet"
        if out_path.exists():
            print(f"[NCAA API] {year} - cached, skipping")
            continue

        print(f"[NCAA API] Fetching tournament games for {year}...")
        df = fetch_tournament_games(year)
        if df is not None:
            df.write_parquet(out_path)
            print(f"  -> {len(df)} games saved")
        else:
            print(f"  -> No games found")

    print("[NCAA API] Done.")


if __name__ == "__main__":
    run()
