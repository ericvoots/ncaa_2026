"""Process Warren Nolan conference rankings into conference_strength table."""
import polars as pl

import config


def run():
    """Load all Warren Nolan data and combine into conference_strength.parquet."""
    print("[ProcessConferences] Building conference strength table...")

    wn_dir = config.RAW_DIR / "warrennolan"
    if not wn_dir.exists():
        print("  [SKIP] No Warren Nolan data found")
        return

    all_dfs = []
    for year in config.SCRAPE_YEARS:
        path = wn_dir / f"{year}.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            if "season" not in df.columns:
                df = df.with_columns(pl.lit(year).alias("season"))
            all_dfs.append(df)
            print(f"  {year}: {len(df)} conferences")

    if not all_dfs:
        print("  [ERROR] No conference data found")
        return

    combined = pl.concat(all_dfs, how="diagonal_relaxed")

    out_path = config.PROCESSED_DIR / "conference_strength.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path)
    print(f"  -> Saved {len(combined)} rows to {out_path}")


if __name__ == "__main__":
    run()
