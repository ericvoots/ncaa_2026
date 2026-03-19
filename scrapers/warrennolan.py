"""Scrape conference NET rankings from WarrenNolan.com."""
import time

import polars as pl
import requests
from bs4 import BeautifulSoup

import config


def scrape_conference_rankings(year: int) -> pl.DataFrame | None:
    """Scrape conference NET rankings for a given year."""
    url = config.WARRENNOLAN_CONF_URL.format(year=year)
    headers = {"User-Agent": config.USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Find the main data table
    table = soup.find("table")
    if not table:
        print(f"  [WARN] No table found for {year}")
        return None

    rows_data = []
    tbody = table.find("tbody")
    rows_iter = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]  # skip header

    for tr in rows_iter:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        # Parse columns: Conference, Rank, NC Rec, NC WP, Leader, NET
        conference = tds[0].get_text(strip=True)
        if not conference or conference.lower() == "conference":
            continue

        try:
            rank = int(tds[1].get_text(strip=True))
        except (ValueError, IndexError):
            rank = None

        # NC Record like "185-23"
        nc_rec = tds[2].get_text(strip=True) if len(tds) > 2 else ""
        nc_wins, nc_losses = None, None
        if "-" in nc_rec:
            parts = nc_rec.split("-")
            try:
                nc_wins = int(parts[0])
                nc_losses = int(parts[1])
            except ValueError:
                pass

        # NC Win %
        nc_wp = None
        if len(tds) > 3:
            try:
                nc_wp = float(tds[3].get_text(strip=True))
            except ValueError:
                pass

        # Conference leader
        leader = tds[4].get_text(strip=True) if len(tds) > 4 else None

        # Leader NET ranking
        leader_net = None
        if len(tds) > 5:
            try:
                leader_net = int(tds[5].get_text(strip=True))
            except ValueError:
                pass

        rows_data.append({
            "conference": conference,
            "conf_rank": rank,
            "nc_wins": nc_wins,
            "nc_losses": nc_losses,
            "nc_win_pct": nc_wp,
            "conf_leader": leader,
            "conf_leader_net": leader_net,
            "season": year,
        })

    if not rows_data:
        print(f"  [WARN] No conference data for {year}")
        return None

    return pl.DataFrame(rows_data)


def run(years: list[int] | None = None):
    """Main entry point: scrape Warren Nolan for all years."""
    years = years or config.SCRAPE_YEARS
    out_dir = config.RAW_DIR / "warrennolan"
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        out_path = out_dir / f"{year}.parquet"
        if out_path.exists():
            print(f"[WarrenNolan] {year} - cached, skipping")
            continue

        print(f"[WarrenNolan] Scraping {year}...")
        df = scrape_conference_rankings(year)
        if df is not None:
            df.write_parquet(out_path)
            print(f"  -> {len(df)} conferences saved")
        else:
            print(f"  -> FAILED")
        time.sleep(1)

    print("[WarrenNolan] Done.")


if __name__ == "__main__":
    run()
