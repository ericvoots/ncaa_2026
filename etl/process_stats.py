"""Process raw TeamRankings stat files into a single wide team-season table."""
import polars as pl

import config
from team_index import TeamIndex


def process_year(year: int, team_idx: TeamIndex) -> pl.DataFrame | None:
    """Load all stat parquets for a year and join into a wide table."""
    year_dir = config.RAW_DIR / "teamrankings" / str(year)
    if not year_dir.exists():
        print(f"  [SKIP] No data for {year}")
        return None

    parquets = list(year_dir.glob("*.parquet"))
    if not parquets:
        print(f"  [SKIP] No parquet files for {year}")
        return None

    # Start with the first file
    base_df = pl.read_parquet(parquets[0])
    if "team_name" not in base_df.columns:
        print(f"  [WARN] No team_name column in {parquets[0].name}")
        return None

    # Join all other stat files on team_name
    for pq in parquets[1:]:
        try:
            df = pl.read_parquet(pq)
            if "team_name" not in df.columns:
                continue
            # Get stat columns (everything except team_name)
            stat_cols = [c for c in df.columns if c != "team_name"]
            # Only join new columns
            existing = set(base_df.columns)
            new_cols = [c for c in stat_cols if c not in existing]
            if new_cols:
                base_df = base_df.join(
                    df.select(["team_name"] + new_cols),
                    on="team_name",
                    how="left",
                )
        except Exception as e:
            print(f"  [WARN] Error reading {pq.name}: {e}")

    # Resolve team names to team_ids
    team_ids = []
    for name in base_df["team_name"].to_list():
        tid = team_idx.resolve(name)
        team_ids.append(tid)

    base_df = base_df.with_columns([
        pl.Series("team_id", team_ids).cast(pl.Int64),
        pl.lit(year).alias("season"),
    ])

    # Drop rows without a team_id match
    unmatched = base_df.filter(pl.col("team_id").is_null())
    if len(unmatched) > 0:
        print(f"  [WARN] {len(unmatched)} unmatched teams in {year}")
        for name in unmatched["team_name"].to_list()[:10]:
            print(f"    - {name}")

    base_df = base_df.filter(pl.col("team_id").is_not_null())

    # Deduplicate: if multiple TeamRankings names resolve to the same team_id,
    # keep the first occurrence (they should have identical stats)
    base_df = base_df.unique(subset=["team_id"], keep="first")

    return base_df


def run():
    """Process all years into a single team_season_stats.parquet."""
    print("[ProcessStats] Building team-season stats table...")

    team_idx = TeamIndex()
    if team_idx.df is None:
        print("  [ERROR] Team index not found. Run build_team_index first.")
        return

    all_dfs = []
    for year in config.SCRAPE_YEARS:
        print(f"  Processing {year}...")
        df = process_year(year, team_idx)
        if df is not None:
            all_dfs.append(df)
            print(f"    -> {len(df)} teams, {len(df.columns)} columns")

    if not all_dfs:
        print("  [ERROR] No data to process")
        return

    # Stack all years - use diagonal concat to handle different column sets
    combined = pl.concat(all_dfs, how="diagonal_relaxed")

    out_path = config.PROCESSED_DIR / "team_season_stats.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path)
    print(f"  -> Saved {len(combined)} rows, {len(combined.columns)} cols to {out_path}")


if __name__ == "__main__":
    run()
