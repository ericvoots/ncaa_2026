"""Scrape team stats from TeamRankings.com for all configured years."""
import time
from pathlib import Path

import polars as pl
import requests
from bs4 import BeautifulSoup

import config


def scrape_stat(stat_slug: str, date: str) -> pl.DataFrame | None:
    """Scrape a single stat page for a given date.

    Returns DataFrame with columns:
        team_name, {stat}_season, {stat}_last3, {stat}_last1,
        {stat}_home, {stat}_away, {stat}_prior
    """
    url = f"{config.TEAMRANKINGS_BASE_URL}{stat_slug}?date={date}"
    headers = {"User-Agent": config.USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="tr-table")
    if not table:
        # Try finding any table with datatable class
        table = soup.find("table")
    if not table:
        print(f"  [WARN] No table found for {stat_slug} on {date}")
        return None

    rows_data = []
    tbody = table.find("tbody")
    if not tbody:
        print(f"  [WARN] No tbody for {stat_slug} on {date}")
        return None

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        # Extract team name - try data-sort first, then text
        team_td = tds[1]
        team_name = team_td.get("data-sort", "") or team_td.get_text(strip=True)
        if not team_name:
            team_name = team_td.get_text(strip=True)

        # Extract values - use data-sort for clean numeric, fallback to text
        values = []
        for td in tds[2:]:
            raw = td.get("data-sort", "") or td.get_text(strip=True)
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                values.append(None)

        # Pad to 6 values: season, last3, last1, home, away, prior
        while len(values) < 6:
            values.append(None)

        rows_data.append({
            "team_name": team_name,
            f"{stat_slug}_season": values[0],
            f"{stat_slug}_last3": values[1],
            f"{stat_slug}_last1": values[2],
            f"{stat_slug}_home": values[3],
            f"{stat_slug}_away": values[4],
            f"{stat_slug}_prior": values[5],
        })

    if not rows_data:
        print(f"  [WARN] No data rows for {stat_slug} on {date}")
        return None

    return pl.DataFrame(rows_data)


def scrape_all_stats(year: int) -> None:
    """Scrape all configured stat slugs for a given tournament year."""
    date = config.TOURNAMENT_DATES.get(year)
    if not date:
        print(f"  [SKIP] No tournament date configured for {year}")
        return

    out_dir = config.RAW_DIR / "teamrankings" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(config.TEAMRANKINGS_STAT_SLUGS)
    for i, slug in enumerate(config.TEAMRANKINGS_STAT_SLUGS, 1):
        out_path = out_dir / f"{slug}.parquet"
        if out_path.exists():
            print(f"  [{i}/{total}] {slug} - cached, skipping")
            continue

        print(f"  [{i}/{total}] {slug} (date={date})...")
        df = scrape_stat(slug, date)
        if df is not None:
            df.write_parquet(out_path)
            print(f"    -> {len(df)} teams saved")
        else:
            print(f"    -> FAILED")

        if i < total:
            time.sleep(config.REQUEST_DELAY)


def run(years: list[int] | None = None):
    """Main entry point: scrape TeamRankings for all years."""
    years = years or config.SCRAPE_YEARS
    for year in years:
        print(f"\n[TeamRankings] Scraping year {year}...")
        scrape_all_stats(year)
    print("\n[TeamRankings] Done.")


if __name__ == "__main__":
    run()
