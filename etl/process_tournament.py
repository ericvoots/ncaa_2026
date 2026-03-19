"""Process NCAA API tournament games into normalized matchup table."""
import polars as pl

import config
from team_index import TeamIndex


def run():
    """Build tournament_matchups.parquet from raw NCAA API game data."""
    print("[ProcessTournament] Building tournament matchups table...")

    team_idx = TeamIndex()
    games_dir = config.RAW_DIR / "ncaa_api" / "tournament_games"
    if not games_dir.exists():
        print("  [SKIP] No tournament game data found")
        return

    all_dfs = []
    for year in config.SCRAPE_YEARS:
        path = games_dir / f"{year}.parquet"
        if not path.exists():
            continue

        df = pl.read_parquet(path)
        if len(df) == 0:
            continue

        # Resolve team names to IDs
        away_ids = []
        home_ids = []
        for row in df.iter_rows(named=True):
            away_ids.append(team_idx.resolve(row["away_name"]))
            home_ids.append(team_idx.resolve(row["home_name"]))

        df = df.with_columns([
            pl.Series("away_team_id", away_ids).cast(pl.Int64),
            pl.Series("home_team_id", home_ids).cast(pl.Int64),
            pl.lit(year).alias("season"),
        ])

        # Normalize: team_a = higher seed (lower number), team_b = lower seed
        # If seeds are equal or missing, use alphabetical on name
        matchups = []
        for row in df.iter_rows(named=True):
            away_seed = row["away_seed"]
            home_seed = row["home_seed"]
            away_id = row["away_team_id"]
            home_id = row["home_team_id"]

            # Determine team_a (better seed) and team_b
            if away_seed is not None and home_seed is not None:
                if away_seed <= home_seed:
                    a_is_away = True
                else:
                    a_is_away = False
            elif away_seed is not None:
                a_is_away = True
            elif home_seed is not None:
                a_is_away = False
            else:
                a_is_away = (row["away_name"] <= row["home_name"])

            if a_is_away:
                matchups.append({
                    "team_a_id": away_id,
                    "team_b_id": home_id,
                    "team_a_name": row["away_name"],
                    "team_b_name": row["home_name"],
                    "team_a_seed": away_seed,
                    "team_b_seed": home_seed,
                    "team_a_score": row["away_score"],
                    "team_b_score": row["home_score"],
                    "winner": 1 if row["away_winner"] else 0,
                    "bracket_round": row["bracket_round"],
                    "bracket_region": row.get("bracket_region", ""),
                    "season": year,
                    "game_id": row.get("game_id", ""),
                    "date": row.get("date", ""),
                })
            else:
                matchups.append({
                    "team_a_id": home_id,
                    "team_b_id": away_id,
                    "team_a_name": row["home_name"],
                    "team_b_name": row["away_name"],
                    "team_a_seed": home_seed,
                    "team_b_seed": away_seed,
                    "team_a_score": row["home_score"],
                    "team_b_score": row["away_score"],
                    "winner": 1 if row["home_winner"] else 0,
                    "bracket_round": row["bracket_round"],
                    "bracket_region": row.get("bracket_region", ""),
                    "season": year,
                    "game_id": row.get("game_id", ""),
                    "date": row.get("date", ""),
                })

        if matchups:
            all_dfs.append(pl.DataFrame(matchups))
            print(f"  {year}: {len(matchups)} games")

    if not all_dfs:
        print("  [ERROR] No matchup data to process")
        return

    combined = pl.concat(all_dfs, how="diagonal_relaxed")

    # QA checks
    print(f"\n  QA Checks:")
    print(f"    Total games: {len(combined)}")
    for year in combined["season"].unique().sort().to_list():
        year_games = combined.filter(pl.col("season") == year)
        print(f"    {year}: {len(year_games)} games")

    null_ids = combined.filter(
        pl.col("team_a_id").is_null() | pl.col("team_b_id").is_null()
    )
    if len(null_ids) > 0:
        print(f"    [WARN] {len(null_ids)} games with unresolved team IDs")

    out_path = config.PROCESSED_DIR / "tournament_matchups.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path)
    print(f"\n  -> Saved {len(combined)} matchups to {out_path}")


if __name__ == "__main__":
    run()
