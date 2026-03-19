"""Build canonical team index from NCAA API schools data + TeamRankings names."""
import json
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

import config
from team_index import MANUAL_OVERRIDES, _normalize


def _build_from_ncaa_api() -> list[dict]:
    """Build initial team list from NCAA API schools index."""
    schools_path = config.RAW_DIR / "ncaa_api" / "schools_index.json"
    if not schools_path.exists():
        print("  [WARN] Schools index not found, run NCAA API scraper first")
        return []

    with open(schools_path) as f:
        data = json.load(f)

    teams = []
    # The schools index is typically a list of {slug, name, ...} or similar
    if isinstance(data, list):
        for i, school in enumerate(data):
            if isinstance(school, dict):
                teams.append({
                    "team_id": i + 1,
                    "canonical_name": school.get("name", school.get("short", "")),
                    "ncaa_slug": school.get("slug", school.get("seo", "")),
                    "alt_names": "[]",
                })
            elif isinstance(school, str):
                teams.append({
                    "team_id": i + 1,
                    "canonical_name": school,
                    "ncaa_slug": school.lower().replace(" ", "-"),
                    "alt_names": "[]",
                })
    elif isinstance(data, dict):
        # Could be keyed by slug or id
        for i, (key, val) in enumerate(data.items()):
            name = val if isinstance(val, str) else val.get("name", key)
            teams.append({
                "team_id": i + 1,
                "canonical_name": name,
                "ncaa_slug": key,
                "alt_names": "[]",
            })

    return teams


def _collect_teamrankings_names() -> set[str]:
    """Collect all unique team names from scraped TeamRankings data."""
    tr_dir = config.RAW_DIR / "teamrankings"
    names = set()
    if not tr_dir.exists():
        return names

    for year_dir in tr_dir.iterdir():
        if not year_dir.is_dir():
            continue
        # Read one parquet to get team names
        for pq in year_dir.glob("*.parquet"):
            try:
                df = pl.read_parquet(pq)
                if "team_name" in df.columns:
                    names.update(df["team_name"].to_list())
            except Exception:
                pass
            break  # One file per year is enough
    return names


def _add_alt_names(teams: list[dict], tr_names: set[str]) -> list[dict]:
    """Add TeamRankings name variants as alt_names using fuzzy matching."""
    # Build lookup from canonical names
    canonical_map = {}
    for t in teams:
        norm = _normalize(t["canonical_name"])
        canonical_map[norm] = t["team_id"]

    all_canonicals = list(canonical_map.keys())
    id_to_alts: dict[int, list[str]] = {t["team_id"]: [] for t in teams}

    # Also add all manual override mappings
    override_canonical_to_id = {}
    for variant, canonical in MANUAL_OVERRIDES.items():
        norm_c = _normalize(canonical)
        if norm_c in canonical_map:
            tid = canonical_map[norm_c]
            id_to_alts[tid].append(variant)
            override_canonical_to_id[variant] = tid

    for tr_name in tr_names:
        norm = _normalize(tr_name)

        # Skip if already a canonical
        if norm in canonical_map:
            continue

        # Check overrides
        if tr_name in MANUAL_OVERRIDES:
            canonical = MANUAL_OVERRIDES[tr_name]
            norm_c = _normalize(canonical)
            if norm_c in canonical_map:
                tid = canonical_map[norm_c]
                id_to_alts[tid].append(tr_name)
            continue

        # Fuzzy match
        result = process.extractOne(norm, all_canonicals, scorer=fuzz.WRatio, score_cutoff=85)
        if result:
            matched, score, _ = result
            tid = canonical_map[matched]
            id_to_alts[tid].append(tr_name)
            if score < 90:
                print(f"  [FUZZY {score:.0f}] '{tr_name}' -> '{matched}'")
        else:
            print(f"  [UNMATCHED] '{tr_name}' - no match found")

    # Update alt_names
    for t in teams:
        alts = list(set(id_to_alts[t["team_id"]]))
        t["alt_names"] = json.dumps(alts)

    return teams


def run():
    """Build and save the team index."""
    print("[TeamIndex] Building team index...")

    teams = _build_from_ncaa_api()
    if not teams:
        # Fallback: build from TeamRankings names directly
        print("  [INFO] No NCAA API data, building from TeamRankings names...")
        tr_names = _collect_teamrankings_names()
        teams = []
        for i, name in enumerate(sorted(tr_names), 1):
            canonical = MANUAL_OVERRIDES.get(name, name)
            teams.append({
                "team_id": i,
                "canonical_name": canonical,
                "ncaa_slug": canonical.lower().replace(" ", "-").replace("'", ""),
                "alt_names": json.dumps([name]) if name != canonical else "[]",
            })
        # Deduplicate by canonical name
        seen = {}
        deduped = []
        for t in teams:
            if t["canonical_name"] not in seen:
                seen[t["canonical_name"]] = t
                deduped.append(t)
            else:
                # Merge alt names
                existing = seen[t["canonical_name"]]
                existing_alts = json.loads(existing["alt_names"])
                new_alts = json.loads(t["alt_names"])
                merged = list(set(existing_alts + new_alts))
                existing["alt_names"] = json.dumps(merged)
        teams = deduped
        # Reassign IDs
        for i, t in enumerate(teams, 1):
            t["team_id"] = i

    else:
        tr_names = _collect_teamrankings_names()
        if tr_names:
            print(f"  -> Found {len(tr_names)} TeamRankings names to match")
            teams = _add_alt_names(teams, tr_names)

    # Save
    out_path = config.RAW_DIR / "team_index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(teams)
    df.write_parquet(out_path)
    print(f"  -> Saved {len(teams)} teams to {out_path}")
    return df


if __name__ == "__main__":
    run()
