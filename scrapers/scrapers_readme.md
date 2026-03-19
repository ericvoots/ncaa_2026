# Scrapers Module -- NCAAB Data Collection Pipeline

This document provides an exhaustive, richly detailed guide to the `scrapers` module within the NCAAB prediction pipeline. The scrapers are responsible for the foundational act of reaching out across the internet, pulling in raw basketball data from multiple authoritative sources, and persisting that data locally as Apache Parquet files for downstream feature engineering and model training. Every scraper in this module is designed to be idempotent (safe to re-run), respectful of external services (rate-limited), and deterministic in its output structure.

---

## Table of Contents

- [1. Overview of the Scrapers Module](#1-overview-of-the-scrapers-module)
  - [1.1 What Does This Module Do? (The Simple Version)](#11-what-does-this-module-do-the-simple-version)
  - [1.2 Architectural Role in the Pipeline](#12-architectural-role-in-the-pipeline)
  - [1.3 Shared Conventions and Configuration](#13-shared-conventions-and-configuration)
- [2. ncaa_api.py -- NCAA Tournament Game Results](#2-ncaa_apipy----ncaa-tournament-game-results)
  - [2.1 Purpose and Motivation](#21-purpose-and-motivation)
  - [2.2 Data Source: The NCAA API](#22-data-source-the-ncaa-api)
  - [2.3 Date Window and Iteration Strategy](#23-date-window-and-iteration-strategy)
  - [2.4 Game Filtering and Field Extraction](#24-game-filtering-and-field-extraction)
  - [2.5 Schools Index Caching](#25-schools-index-caching)
  - [2.6 Output Schema](#26-output-schema)
  - [2.7 Rate Limiting](#27-rate-limiting)
- [3. teamrankings.py -- Team Statistical Profiles](#3-teamrankingspy----team-statistical-profiles)
  - [3.1 Purpose and Motivation](#31-purpose-and-motivation)
  - [3.2 Data Source: TeamRankings.com](#32-data-source-teamrankingscom)
  - [3.3 Date Parameter and Pre-Tournament Cutoff](#33-date-parameter-and-pre-tournament-cutoff)
  - [3.4 Stat Slugs and the Scraping Loop](#34-stat-slugs-and-the-scraping-loop)
  - [3.5 HTML Parsing with BeautifulSoup](#35-html-parsing-with-beautifulsoup)
  - [3.6 Output Schema](#36-output-schema)
  - [3.7 Rate Limiting](#37-rate-limiting)
- [4. warrennolan.py -- Conference NET Rankings](#4-warrennolanpy----conference-net-rankings)
  - [4.1 Purpose and Motivation](#41-purpose-and-motivation)
  - [4.2 Data Source: WarrenNolan.com](#42-data-source-warrennolancom)
  - [4.3 HTML Parsing and Column Extraction](#43-html-parsing-and-column-extraction)
  - [4.4 Output Schema](#44-output-schema)
  - [4.5 Rate Limiting](#45-rate-limiting)
- [5. File and Directory Structure](#5-file-and-directory-structure)
- [6. Running the Scrapers](#6-running-the-scrapers)
- [7. Caching and Idempotency](#7-caching-and-idempotency)
- [8. Error Handling Philosophy](#8-error-handling-philosophy)
- [9. Configuration Dependencies](#9-configuration-dependencies)
- [10. Dependencies](#10-dependencies)

---

## 1. Overview of the Scrapers Module

### 1.1 What Does This Module Do? (The Simple Version)

Imagine you want to predict who will win a basketball game in the NCAA March Madness tournament. Before you can make any predictions, you need information -- lots of it. You need to know what happened in past tournaments (who played, who won, what were the scores), how good each team was during the regular season (shooting percentages, turnovers, rebounds), and how strong their conference was overall. This module is the part of the system that goes out and collects all of that information from websites, like a very patient researcher visiting three different libraries and carefully copying data into notebooks.

Think of it like this: there are three "researchers" in this module. One goes to the NCAA's official records and writes down every tournament game result. Another visits a statistics website and records about 42 different performance measurements for every single college basketball team. The third visits a website that ranks basketball conferences and writes down how each conference performed. All three researchers save their notes in a special, efficient format (Parquet files) so that the rest of the pipeline can read them very, very quickly.

At the intermediate level, these scrapers implement three distinct data ingestion strategies -- JSON API consumption, HTML table scraping via BeautifulSoup, and a second variant of HTML table scraping against a differently structured site -- each targeting a specific external data source and producing standardized Polars DataFrames serialized to Apache Parquet. The module is the "extract" stage of what could be described as a lightweight ETL (Extract, Transform, Load) architecture, where the "transform" and "load" stages happen in subsequent pipeline modules.

At the most advanced and rigorous level, the scrapers collectively operationalize a temporal-consistency-preserving data acquisition protocol. Every data point is anchored to a specific point in time (pre-tournament cutoff dates for team stats, post-game finality for tournament results) to ensure that no information leaks from the future into the training set. This is a fundamental requirement for any supervised learning system that operates on time-series or event-driven data: the feature vector for a prediction at time T must contain only information available strictly before time T. The scrapers encode this constraint directly into their request parameterization (the `?date=` parameter on TeamRankings) and their response filtering (the `gameState == "final"` check on NCAA API results).

### 1.2 Architectural Role in the Pipeline

Within the broader NCAAB prediction pipeline, the scrapers module occupies the "data ingestion" layer -- it is the very first stage of the pipeline, responsible for transforming external, ephemeral web resources into durable, local, structured datasets. Nothing downstream can function without the raw data that these scrapers produce. The feature engineering layer reads the Parquet files produced here; the model training layer depends on those features; and the prediction layer depends on the trained models. The scrapers are, in the most literal sense, the foundation upon which the entire predictive apparatus is built.

From a software engineering perspective, each scraper follows a consistent architectural pattern: a `run()` function serves as the public entry point, accepting an optional list of years (defaulting to `config.SCRAPE_YEARS`, which spans 2015-2026 excluding the cancelled 2020 season). Each scraper checks for cached output before making any network request, ensuring that re-execution is both safe and efficient. All configuration -- URLs, delays, stat lists, date mappings -- is centralized in the project-level `config.py` module, which keeps the scraper code clean and the operational parameters easy to audit and modify.

### 1.3 Shared Conventions and Configuration

All three scrapers import from `config.py` at the project root and share the following conventions:

| Configuration Constant | Value | Purpose |
|---|---|---|
| `SCRAPE_YEARS` | `[2015, 2016, ..., 2019, 2021, ..., 2026]` | Years to scrape (2020 excluded -- no tournament due to COVID-19) |
| `RAW_DIR` | `data/raw/` | Base output directory for all raw scraped data |
| `REQUEST_DELAY` | `2.0` seconds | Delay between TeamRankings HTTP requests |
| `NCAA_API_DELAY` | `0.3` seconds | Delay between NCAA API HTTP requests |
| `USER_AGENT` | Chrome 120 UA string | Sent with requests to TeamRankings and WarrenNolan to avoid bot detection |
| `TOURNAMENT_DATES` | Dict mapping year to date string | Pre-tournament cutoff dates for TeamRankings queries |

Every scraper persists data as Apache Parquet files via the Polars library. Parquet was chosen over CSV for its columnar compression, type preservation (nullable integers, nullable floats), and dramatically faster read speeds -- properties that matter when downstream stages join and aggregate across many files. Polars was chosen over Pandas for its Rust-backed performance, its native Parquet support without requiring PyArrow as an intermediary, and its stricter type system that surfaces schema issues at write time rather than allowing silent type coercion.

---

## 2. ncaa_api.py -- NCAA Tournament Game Results

### 2.1 Purpose and Motivation

At the most basic level, this scraper answers the question: "What happened in the NCAA tournament?" For every year in the configured range, it retrieves the complete list of tournament games -- First Four, Round of 64, Round of 32, Sweet 16, Elite 8, Final Four, and Championship -- along with the teams involved, their seeds, the final scores, and who won. This is the ground truth label data that the model ultimately learns to predict: given two teams, which one wins?

Without this data, there is literally nothing to predict and nothing to evaluate predictions against. It is the dependent variable, the target column, the thing the entire pipeline exists to forecast. From a machine learning perspective, this scraper produces the labeled dataset that makes supervised learning possible. Each row in the output is, in essence, a training example: "Team A with seed X played Team B with seed Y in round Z, and Team A won by a score of 75-68." The model's job is to learn the mapping from pre-game features (team stats, conference strength, seeding) to the binary outcome (win/loss).

From a data engineering perspective, this scraper also provides the structural scaffold for the entire dataset: the `game_id` field serves as a natural primary key, the `date` field enables temporal ordering, the `bracket_round` and `bracket_region` fields enable stratified analysis (e.g., "How do 12-seeds perform against 5-seeds in the Round of 64?"), and the `away_seo` / `home_seo` fields provide stable team identifiers that can be used for joining with other data sources.

### 2.2 Data Source: The NCAA API

The data source is an unofficial NCAA API hosted at `https://ncaa-api.henrygd.me`. This is a community-maintained mirror/proxy of the NCAA's scoreboard data, organized by sport, division, and date. The specific endpoint pattern used is:

```
https://ncaa-api.henrygd.me/scoreboard/basketball-men/d1/{YYYY}/{MM}/{DD}
```

Each request returns a JSON payload containing all Division I men's basketball games played on that date. The response structure nests game data under a `games` array, where each element contains a `game` object with sub-objects for `away` and `home` teams. The scraper filters this broader set down to tournament games only (those with a non-empty `bracketRound` field) and further restricts to games with `gameState == "final"` to ensure only completed games are captured.

Additionally, the scraper fetches a schools index from the `/schools-index` endpoint. This is a comprehensive list of NCAA schools (approximately 1,173 institutions across all divisions) and is cached as a JSON file for potential use in name resolution and cross-referencing.

It is worth noting that because this is a community-maintained API rather than an official NCAA endpoint, its availability and data format could change without notice. However, the structured JSON format is substantially more reliable and easier to parse than HTML scraping would be, making it the preferred data source for game-level results.

### 2.3 Date Window and Iteration Strategy

The NCAA tournament does not happen on a single day -- it unfolds across roughly four weeks, from mid-March through early April. Rather than attempting to predict the exact dates of games (which shift slightly from year to year based on scheduling), the scraper takes a deliberately generous approach: it iterates over every single date from **March 13 through March 31**, and then from **April 1 through April 10**, for each year.

This creates a date window of 29 days per year (19 in March + 10 in April). On the vast majority of these dates, there will be no tournament games at all -- the API will return either a 404 or an empty game list, which the scraper silently ignores. This brute-force approach is intentionally over-inclusive to guarantee that no tournament game is missed regardless of scheduling variations across years. The First Four typically begins around March 14-19, and the National Championship game is typically played on the first Monday of April, so the March 13 through April 10 window comfortably encompasses every possible tournament date.

The date strings are formatted as `YYYY/MM/DD` (e.g., `2024/03/21`) using Python f-strings with zero-padded month and day values to match the API's URL path structure. The construction logic is:

```python
for day in range(13, 32):       # March 13 through March 31
    date_ranges.append(f"{year}/{3:02d}/{day:02d}")
for day in range(1, 11):        # April 1 through April 10
    date_ranges.append(f"{year}/{4:02d}/{day:02d}")
```

From a computational complexity standpoint, this means approximately 29 HTTP requests per year, multiplied by the number of years being scraped. At a rate of 0.3 seconds between requests, a single year takes roughly 9 seconds of wall-clock time. For the full 11-year range, that is approximately 100 seconds of network time, assuming no caching. This is a very modest load by any standard.

### 2.4 Game Filtering and Field Extraction

For each date that returns game data, the scraper iterates through the `games` array in the JSON response. Each game entry is nested under a `game` key and contains `away` and `home` sub-objects. The scraper applies two filters that must both be satisfied for a game to be included:

1. **`bracketRound` must be non-empty**: This distinguishes tournament games from regular-season, conference tournament, NIT, CBI, or other non-March-Madness games that may appear on the same scoreboard date. Only games explicitly tagged with a bracket round (e.g., "1st Round", "Sweet 16", "Championship") are included. This is the primary discriminator and is remarkably effective because only NCAA tournament games carry this metadata.

2. **`gameState` must equal `"final"`**: This ensures that in-progress, scheduled, or postponed games are excluded. Since the scraper typically runs long after the tournament has concluded, this filter is largely a safety check, but it becomes critically important if the scraper is run during an active tournament window -- you would not want to capture a halftime score as a final score.

For each qualifying game, the following fields are extracted with careful defensive coding:

- **`game_id`**: The NCAA's unique identifier for the game, sourced from `g.get("gameID", "")`.
- **`date`**: The game's start date, sourced from `g.get("startDate", date_str)` with the iterated date string as a fallback.
- **`bracket_round`**: The tournament round name (e.g., "1st Round", "Elite 8", "Final Four").
- **`bracket_region`**: The bracket region (e.g., "East", "South", "Midwest", "West"), sourced from `g.get("bracketRegion", "")`.
- **`away_name` / `home_name`**: Short display names for the teams, extracted from the nested `names.short` path.
- **`away_seo` / `home_seo`**: SEO-friendly name slugs from the `names.seo` path. These are lowercase, hyphenated strings (e.g., "north-carolina") that are highly useful for deterministic cross-referencing with other data sources.
- **`away_seed` / `home_seed`**: Integer tournament seeds (1 through 16). These are parsed inside individual try/except blocks because the `seed` field can occasionally be missing, empty, or non-numeric in the API response. The scraper degrades gracefully by setting the seed to `None` rather than crashing. This is important because First Four games sometimes have unusual seed representations, and the National Championship game might not carry seed information in all API versions.
- **`away_score` / `home_score`**: Final integer scores, with a default of 0 if the field is missing (though this should never happen for `gameState == "final"` games).
- **`away_winner` / `home_winner`**: Boolean flags indicating the winning team, sourced directly from the API's `winner` field.

### 2.5 Schools Index Caching

Before fetching game data, the `run()` function calls `fetch_schools_index()`, which retrieves a comprehensive list of NCAA schools from the `/schools-index` endpoint. This response is cached as a JSON file at `data/raw/ncaa_api/schools_index.json`. If the file already exists, the function reads from disk instead of making a network request, implementing a simple but effective cache-or-fetch pattern. The cache has no expiration or invalidation mechanism -- the schools index changes very infrequently (only when institutions join or leave the NCAA), so a one-time fetch is sufficient for the lifetime of this project.

The parent directory is created on-demand via `mkdir(parents=True, exist_ok=True)`, which means the scraper can be run on a fresh checkout with no pre-existing directory structure and will create everything it needs automatically.

### 2.6 Output Schema

Each year's data is saved as a single Parquet file at:

```
data/raw/ncaa_api/tournament_games/{year}.parquet
```

The DataFrame columns are:

| Column | Type | Description |
|---|---|---|
| `game_id` | String | NCAA unique game identifier |
| `date` | String | Game date (from `startDate` or the query date) |
| `bracket_round` | String | Tournament round name |
| `bracket_region` | String | Bracket region |
| `away_name` | String | Away team short name |
| `away_seo` | String | Away team SEO slug |
| `away_seed` | Int (nullable) | Away team tournament seed (1-16) |
| `away_score` | Int | Away team final score |
| `away_winner` | Bool | Whether the away team won |
| `home_name` | String | Home team short name |
| `home_seo` | String | Home team SEO slug |
| `home_seed` | Int (nullable) | Home team tournament seed (1-16) |
| `home_score` | Int | Home team final score |
| `home_winner` | Bool | Whether the home team won |

A typical year contains approximately 63-67 games (First Four through Championship), so each Parquet file is very small -- a few kilobytes at most.

### 2.7 Rate Limiting

Between each date request, the scraper sleeps for `config.NCAA_API_DELAY` seconds, which is configured at **0.3 seconds**. This is a relatively light rate limit, reflecting the fact that the NCAA API mirror is a lightweight community service and the scraper makes a modest number of requests (approximately 29 per year). The delay is applied after every scoreboard request, regardless of whether data was found. The total sleep time for a full 11-year uncached scrape is approximately 29 * 11 * 0.3 = 96 seconds.

---

## 3. teamrankings.py -- Team Statistical Profiles

### 3.1 Purpose and Motivation

If `ncaa_api.py` tells us *what happened* in past tournaments, then `teamrankings.py` tells us *how good each team was* going into the tournament. This scraper collects approximately 42 distinct statistical measures for every Division I team, covering categories like offensive efficiency, defensive efficiency, ball control (assists, turnovers, steals), rebounding, shooting percentages, block rates, scoring margins, tempo, and win percentages.

Think of it like a report card for every team, taken right before the tournament starts. Instead of grades in math and reading, the report card has numbers for things like "How many points does this team score per game?" and "How often does this team turn the ball over?" and "How well does this team shoot three-pointers?" These report cards become the features -- the input variables -- that the machine learning model uses to make predictions.

At a more sophisticated level, the stat selection in `config.TEAMRANKINGS_STAT_SLUGS` reflects a deliberate emphasis on ball control metrics (assist-to-turnover ratio, turnovers per possession, steal percentage, etc.) alongside standard efficiency measures. This focus is based on the hypothesis that ball control is a particularly strong predictor of tournament success, where single-elimination pressure amplifies the impact of turnovers and composure under duress. Teams that take care of the ball in the regular season tend to maintain that discipline under the heightened pressure of the tournament, while turnover-prone teams are more likely to crack.

From a statistical modeling perspective, the six temporal windows captured per stat (season, last 3 games, last 1 game, home, away, prior season) provide a rich feature space that allows downstream models to distinguish between overall team quality (season average), recent form (last 3, last 1), venue-specific performance (home vs. away), and year-over-year program trajectory (prior season). The away column is of particular interest for tournament prediction because NCAA tournament games are played at neutral sites, and away performance may be a better proxy for neutral-site performance than home performance. The prior season column enables models to capture program stability and regression-to-the-mean effects.

### 3.2 Data Source: TeamRankings.com

TeamRankings.com (`https://www.teamrankings.com/ncaa-basketball/stat/`) is a well-established sports analytics website that publishes detailed statistical tables for NCAA basketball. Each stat has its own page identified by a URL slug (e.g., `offensive-efficiency`, `turnovers-per-game`, `opponent-three-point-pct`). The full URL pattern is:

```
https://www.teamrankings.com/ncaa-basketball/stat/{slug}?date={YYYY-MM-DD}
```

Critically, TeamRankings supports a `date` query parameter that allows the user to retrieve statistics as they stood on a specific historical date. This is not merely a convenience -- it is absolutely essential for preventing data leakage. If we were to scrape current-day statistics for a past year, we would be incorporating information that did not exist at the time of the tournament, which would inflate model performance during training and produce misleadingly optimistic accuracy estimates. The `date` parameter ensures temporal integrity by returning a snapshot of the statistical world as it existed on the requested date.

The HTML structure of each stat page consists of a table (with CSS class `tr-table`) containing one row per team. Each row has a rank column (index 0), a team name column (index 1), and several value columns (indices 2+) representing different time windows. Importantly, each `<td>` element carries a `data-sort` attribute containing the clean numeric value, separate from the human-formatted display text.

### 3.3 Date Parameter and Pre-Tournament Cutoff

The `config.TOURNAMENT_DATES` dictionary maps each tournament year to a specific cutoff date -- typically the day before the First Four games begin. For example:

| Year | Cutoff Date | Rationale |
|---|---|---|
| 2015 | 2015-03-16 | Day before First Four |
| 2016 | 2016-03-14 | Day before First Four |
| 2017 | 2017-03-13 | Day before First Four |
| 2018 | 2018-03-12 | Day before First Four |
| 2019 | 2019-03-18 | Day before First Four |
| 2021 | 2021-03-17 | Day before First Four (no 2020 tournament) |
| 2022 | 2022-03-14 | Day before First Four |
| 2023 | 2023-03-13 | Day before First Four |
| 2024 | 2024-03-18 | Day before First Four |
| 2025 | 2025-03-17 | Day before First Four |
| 2026 | 2026-03-16 | Day before First Four |

These dates are carefully chosen so that when TeamRankings is queried with the date parameter, the returned statistics reflect everything up to and including that date, but nothing from the tournament itself. This is the statistical snapshot that a human forecaster would have had access to when filling out their bracket, and it is precisely the information the model should be allowed to see during training.

This mechanism is what makes the entire feature set historically accurate. The model is never trained on information it would not have had at prediction time -- a fundamental requirement for any honest predictive modeling exercise. In the literature on machine learning for time-series and event prediction, this property is known as "temporal validity" or "point-in-time correctness," and its violation (known as "lookahead bias" or "data leakage") is one of the most common and insidious sources of artificially inflated model performance.

### 3.4 Stat Slugs and the Scraping Loop

The full list of 42 stat slugs is defined in `config.TEAMRANKINGS_STAT_SLUGS` and is organized into thematic categories:

**Ball Control (Primary Focus) -- 8 slugs:**
- `assist--per--turnover-ratio` -- The ratio of assists to turnovers; higher means better ball movement with fewer mistakes
- `assists-per-game` -- Raw assist count per game
- `turnovers-per-game` -- Raw turnover count per game
- `turnovers-per-possession` -- Turnovers normalized by possessions (pace-independent)
- `turnover-pct` -- Percentage of possessions ending in a turnover
- `assists-per-possession` -- Assists normalized by possessions (pace-independent)
- `steals-per-game` -- Defensive disruption via steals
- `steal-pct` -- Percentage of opponent possessions ending in a steal

**Offensive Efficiency -- 10 slugs:**
- `offensive-efficiency`, `points-per-game`, `effective-field-goal-pct`, `true-shooting-percentage`, `shooting-pct`, `three-point-pct`, `two-point-pct`, `free-throw-pct`, `free-throw-rate`, `floor-percentage`

**Defensive Efficiency -- 9 slugs:**
- `defensive-efficiency`, `opponent-points-per-game`, `opponent-effective-field-goal-pct`, `opponent-three-point-pct`, `opponent-two-point-pct`, `opponent-turnovers-per-game`, `opponent-turnover-pct`, `opponent-free-throw-rate`, `opponent-assists-per-game`

**Rebounding -- 6 slugs:**
- `total-rebounds-per-game`, `offensive-rebounds-per-game`, `defensive-rebounds-per-game`, `total-rebounding-percentage`, `offensive-rebounding-pct`, `defensive-rebounding-pct`

**Blocks -- 2 slugs:**
- `blocks-per-game`, `block-pct`

**Scoring Margin and Tempo -- 2 slugs:**
- `average-scoring-margin`, `possessions-per-game`

**Win Metrics -- 2 slugs:**
- `win-pct-all-games`, `win-pct-close-games`

**Additional Opponent Stats -- 3 slugs:**
- `opponent-steals-per-game`, `opponent-blocks-per-game`, `opponent-offensive-rebounds-per-game`

For each year and each stat slug, the `scrape_all_stats()` function constructs the URL, delegates to `scrape_stat()`, and saves the result. Progress is logged with a counter (e.g., `[15/42] turnovers-per-game (date=2024-03-18)...`), providing clear visibility into the scraping process during long-running executions.

### 3.5 HTML Parsing with BeautifulSoup

The scraper uses BeautifulSoup with the `lxml` parser (chosen for speed over the default `html.parser` and for robustness over `html5lib`) to navigate the HTML DOM. The parsing strategy proceeds through several carefully ordered steps:

1. **Table discovery**: First, the scraper searches for a `<table>` element with CSS class `tr-table`, which is TeamRankings' standard stat table class. If this is not found (in case of a CSS class name change), it falls back to the first `<table>` element on the page. If no table is found at all, a warning is logged and `None` is returned.

2. **Body extraction**: The `<tbody>` element is located within the table. If absent, a warning is logged and `None` is returned.

3. **Row iteration**: Each `<tr>` within the `<tbody>` is processed. Rows with fewer than 3 `<td>` cells are skipped as they likely represent header, footer, or separator rows.

4. **Team name extraction** (column index 1): The scraper preferentially reads the `data-sort` attribute from the team name `<td>`. This is a particularly important design decision. TeamRankings renders human-friendly content in the visible text (which may include links, icons, or ranking prefixes), but stores a clean, consistent team name string in the `data-sort` attribute. Only when `data-sort` is absent or empty does the scraper fall back to `get_text(strip=True)`.

5. **Value extraction** (columns index 2 onward): For each remaining `<td>`, the same `data-sort`-first strategy is applied. The `data-sort` attribute contains clean numeric values without percentage signs, commas, or other formatting artifacts. Each value is cast to `float()`; on failure, `None` is substituted. This ensures that a single malformed cell does not crash the entire row or table extraction.

6. **Padding to 6 values**: The extracted values list is padded with `None` entries to ensure it always contains exactly 6 elements, corresponding to the season, last3, last1, home, away, and prior columns. This handles edge cases where a particular stat page might have fewer columns than expected.

### 3.6 Output Schema

Each stat for each year is saved as an individual Parquet file at:

```
data/raw/teamrankings/{year}/{stat_slug}.parquet
```

For example: `data/raw/teamrankings/2024/turnovers-per-game.parquet`

Each file contains a DataFrame with the following columns:

| Column | Type | Description |
|---|---|---|
| `team_name` | String | Team name (from `data-sort` attribute or visible text) |
| `{stat_slug}_season` | Float (nullable) | Full season average |
| `{stat_slug}_last3` | Float (nullable) | Last 3 games average |
| `{stat_slug}_last1` | Float (nullable) | Last 1 game value |
| `{stat_slug}_home` | Float (nullable) | Home games average |
| `{stat_slug}_away` | Float (nullable) | Away games average |
| `{stat_slug}_prior` | Float (nullable) | Prior season value |

With approximately 360 teams per file and 42 stat files per year across 11 years, the TeamRankings scraper produces roughly 462 Parquet files (42 stats x 11 years), comprising the largest single data source in the pipeline by far. The total disk footprint is modest -- each file is typically 10-50 KB -- but the information density is enormous: 42 stats x 6 windows x ~360 teams = approximately 90,720 individual data points per year.

### 3.7 Rate Limiting

Between each stat page request, the scraper sleeps for `config.REQUEST_DELAY` seconds, which is configured at **2.0 seconds**. This is significantly more conservative than the NCAA API delay and reflects the fact that TeamRankings is a commercial website that may implement bot detection or rate limiting. The 2-second delay keeps the scraper's request pattern well within the bounds of reasonable automated access -- approximately 30 requests per minute, far below the threshold that would typically trigger rate-limiting countermeasures.

For a single year, scraping all 42 stats takes approximately 42 x 2 = 84 seconds (about 1 minute and 24 seconds), plus network latency. For the full 11-year range with no cached data, the total scraping time is approximately 924 seconds, or about 15 minutes. In practice, since the scraper skips already-cached files, subsequent runs complete almost instantly -- only newly added stats or years incur network requests.

The scraper also sends a realistic `User-Agent` header (`config.USER_AGENT`, mimicking Chrome 120 on Windows 10) with every request. This is necessary because TeamRankings.com may block or throttle requests that arrive with the default Python `requests` User-Agent string (`python-requests/x.x.x`), which is a common signal for automated scrapers. By presenting as a standard browser, the scraper avoids triggering server-side bot detection heuristics.

Importantly, the sleep is applied after every request **except the last one** in each year's batch (`if i < total: time.sleep(...)`). This is a minor but thoughtful optimization that avoids an unnecessary 2-second wait at the end of each year's scraping cycle.

---

## 4. warrennolan.py -- Conference NET Rankings

### 4.1 Purpose and Motivation

While individual team stats tell you how good a team is in isolation, the strength of the conference a team plays in provides important contextual information. A team with a 25-6 record in a power conference like the Big 12 or SEC has faced much stiffer competition than a team with the same record in a smaller conference. The NET (NCAA Evaluation Tool) ranking system, as aggregated by conference on WarrenNolan.com, provides a concise summary of overall conference strength.

Think of it this way: if you know a team is good, but you also know that everyone they played against all season was also good, then that team's stats are even more impressive. Conference NET rankings help the model understand this context. A team from the number-one-ranked conference has been battle-tested in a way that a team from a lower-ranked conference may not have been, and this can matter enormously in the pressure cooker of the NCAA tournament.

At the doctoral level, conference strength serves as a proxy for schedule strength and competitive environment, partially controlling for the confounding variable of opponent quality that inflates raw per-game statistics for teams in weak conferences. When combined with individual team metrics in a predictive model, conference-level features provide hierarchical context -- a form of implicit regularization against overvaluing teams from weak conferences who may have inflated statistics due to playing against inferior competition. In Bayesian terms, the conference strength prior modulates the likelihood of a team's individual statistics being predictive of tournament success. The non-conference win percentage is particularly informative because it captures how the conference's teams perform against out-of-conference opponents, providing an objective cross-conference comparison that is not available from intra-conference records alone.

### 4.2 Data Source: WarrenNolan.com

WarrenNolan.com (`https://www.warrennolan.com/basketball/{year}/net-conference`) is a well-known college basketball analytics site that publishes conference-level NET rankings. The URL includes the year as a path parameter (sourced from `config.WARRENNOLAN_CONF_URL`), making it straightforward to iterate across seasons.

The page displays a single HTML table with one row per conference. Unlike TeamRankings, there is no date parameter -- the data represents the season-end (or near-season-end) conference standings for each year. The NET ranking system was introduced by the NCAA for the 2018-2019 season, replacing the RPI (Rating Percentage Index), so data availability for earlier years may be limited or absent.

### 4.3 HTML Parsing and Column Extraction

The scraper uses BeautifulSoup with the `lxml` parser to find the first `<table>` element on the page. It then determines where to find data rows: if a `<tbody>` exists, its `<tr>` elements are iterated; otherwise, all `<tr>` elements after the first (assumed to be a header row) are used. This dual strategy handles both well-structured tables with explicit `<thead>`/`<tbody>` separation and simpler tables where the first row is implicitly the header.

For each row, the following columns are extracted from the `<td>` elements by index position:

1. **Column 0 -- Conference name** (`conference`): The text content of the first cell. Rows where this is empty or equals "conference" (case-insensitive) are skipped as header or separator rows. This filter prevents the occasional appearance of repeated header rows within the table body from polluting the output.

2. **Column 1 -- Conference rank** (`conf_rank`): Parsed as an integer via `int(tds[1].get_text(strip=True))`. Falls back to `None` on `ValueError` or `IndexError`, which handles cases where the rank cell is empty or contains non-numeric content.

3. **Column 2 -- Non-conference record** (`nc_wins`, `nc_losses`): A string like `"185-23"` that represents the aggregate non-conference record of all teams in the conference. The string is split on the hyphen character, and each part is parsed as an integer. If the string does not contain a hyphen or the parts cannot be parsed as integers, both `nc_wins` and `nc_losses` are set to `None`. This decomposition into separate wins and losses columns is more useful for downstream computation than the raw string.

4. **Column 3 -- Non-conference win percentage** (`nc_win_pct`): Parsed as a float. Represents the aggregate non-conference winning percentage, providing a single-number summary of conference quality in out-of-conference play.

5. **Column 4 -- Conference leader** (`conf_leader`): The name of the top-ranked team in the conference by NET ranking. This is a plain text extraction with no type conversion.

6. **Column 5 -- Leader NET ranking** (`conf_leader_net`): The NET ranking of the conference's top team, parsed as an integer. This provides a ceiling measure of conference quality -- even if the conference as a whole is mediocre, having a single very highly ranked team can influence tournament outcomes for teams from that conference.

Every row also receives a `season` column set to the `year` parameter being scraped, providing an explicit temporal label that is essential for downstream joins with team-level and game-level data.

### 4.4 Output Schema

Each year's data is saved as a single Parquet file at:

```
data/raw/warrennolan/{year}.parquet
```

The DataFrame columns are:

| Column | Type | Description |
|---|---|---|
| `conference` | String | Conference name (e.g., "Big 12", "SEC", "ACC") |
| `conf_rank` | Int (nullable) | Conference NET ranking (1 = strongest) |
| `nc_wins` | Int (nullable) | Total non-conference wins across all conference teams |
| `nc_losses` | Int (nullable) | Total non-conference losses across all conference teams |
| `nc_win_pct` | Float (nullable) | Non-conference win percentage |
| `conf_leader` | String (nullable) | Name of the conference's top NET-ranked team |
| `conf_leader_net` | Int (nullable) | NET ranking of the conference leader |
| `season` | Int | Tournament year |

Each file contains roughly 32-33 rows (one per Division I conference), so the total data volume from this scraper is very small in absolute terms -- typically under 5 KB per file. However, the information density per row is high for feature engineering purposes, as each row summarizes the aggregate quality of an entire conference.

### 4.5 Rate Limiting

Between each year's request, the scraper sleeps for **1 second** (hardcoded in the `run()` loop via `time.sleep(1)`, not sourced from a config constant). Since there is only one page per year and approximately 11 years to scrape, the total network time for a full uncached run is approximately 11-22 seconds. This is the lightest scraper in the module both in terms of request volume and data volume.

The scraper sends the same `config.USER_AGENT` header as the TeamRankings scraper to avoid bot detection.

---

## 5. File and Directory Structure

The following diagram illustrates the complete file layout of the scrapers module and its output:

```
scrapers/
    __init__.py              # Empty; marks directory as a Python package
    scrapers_readme.md       # This documentation file
    ncaa_api.py              # NCAA tournament game results scraper
    teamrankings.py          # Team statistics scraper (~42 stats per year)
    warrennolan.py           # Conference NET rankings scraper

data/raw/                    # Output directory (created automatically by scrapers)
    ncaa_api/
        schools_index.json   # Cached NCAA schools index (JSON)
        tournament_games/
            2015.parquet     # One file per year
            2016.parquet
            ...
            2026.parquet
    teamrankings/
        2015/                # One directory per year
            assist--per--turnover-ratio.parquet
            assists-per-game.parquet
            turnovers-per-game.parquet
            ...              # ~42 parquet files per year
        2016/
            ...
        ...
    warrennolan/
        2015.parquet         # One file per year
        2016.parquet
        ...
        2026.parquet
```

All output directories are created automatically by the scrapers via `mkdir(parents=True, exist_ok=True)`, so no manual directory creation is required before running the scrapers for the first time.

---

## 6. Running the Scrapers

Each scraper can be executed independently as a standalone script from the project root:

```bash
python scrapers/ncaa_api.py
python scrapers/teamrankings.py
python scrapers/warrennolan.py
```

Each scraper's `run()` function can also be called programmatically with an optional list of years to restrict the scope:

```python
from scrapers import ncaa_api, teamrankings, warrennolan

# Scrape everything for all configured years (2015-2026, excluding 2020)
ncaa_api.run()
teamrankings.run()
warrennolan.run()

# Scrape only specific years
ncaa_api.run(years=[2024, 2025])
teamrankings.run(years=[2025])
warrennolan.run(years=[2024, 2025, 2026])
```

For finer-grained control, individual functions can be called directly:

```python
from scrapers.teamrankings import scrape_stat
from scrapers.ncaa_api import fetch_tournament_games
from scrapers.warrennolan import scrape_conference_rankings

# Scrape a single stat for a single date
df = scrape_stat("offensive-efficiency", "2025-03-17")

# Fetch tournament games for a single year
df = fetch_tournament_games(2025)

# Scrape conference rankings for a single year
df = scrape_conference_rankings(2025)
```

When no `years` argument is provided, the scrapers default to `config.SCRAPE_YEARS`, which includes all years from 2015 through 2026, excluding 2020 (when the NCAA tournament was cancelled due to the COVID-19 pandemic).

---

## 7. Caching and Idempotency

Every scraper checks whether its output file already exists before making any network request. If the target Parquet file (or JSON file, in the case of the schools index) is already present on disk, the scraper prints a "cached, skipping" message and moves on without touching the network. This design choice has several important consequences:

1. **Re-runs are fast and safe**: You can run any scraper as many times as you want without duplicating work, consuming unnecessary bandwidth, or risking rate-limit violations on external servers. An idempotent scraper is a polite scraper.

2. **Incremental updates are natural**: If you add a new year to `config.SCRAPE_YEARS` or a new stat slug to `config.TEAMRANKINGS_STAT_SLUGS`, only the newly configured items will be fetched on the next run. Everything else is already cached and will be skipped.

3. **To force a re-scrape, delete the cached file**: If you suspect data corruption, want to refresh a particular year or stat, or if the upstream data has been corrected, simply delete the relevant Parquet file and re-run the scraper. The scraper will detect the missing file and fetch fresh data.

4. **Pipeline orchestration is simplified**: An upstream orchestration script can unconditionally call all three scrapers' `run()` functions without worrying about redundant work. The caching logic within each scraper handles deduplication transparently.

This caching strategy is file-level, not row-level. If a Parquet file exists, it is assumed to be complete and correct. There is no mechanism for partial re-scraping within a single file, nor is there any checksum validation or last-modified comparison. This simplicity is deliberate -- it avoids the complexity of merge/upsert logic and keeps the scraper code straightforward and easy to reason about.

---

## 8. Error Handling Philosophy

The scrapers adopt a deliberately lenient "warn and continue" error-handling philosophy rather than a "fail fast" approach. When an individual request fails (network timeout, HTTP error, missing table, unparseable data), the scraper logs a warning or error message to stdout and continues to the next item in its iteration. This means that a transient network issue on a single date or stat page will not abort an entire multi-hour scraping session.

Specifically, the error handling patterns are:

- **ncaa_api.py**: The `fetch_tournament_games()` function wraps each date's HTTP request and JSON parsing in a broad `except Exception` block, silently continuing to the next date on any failure. While this suppresses potentially informative error messages, it ensures that a single bad date does not prevent the remaining 28 dates from being processed. The function returns `None` if no games are found for an entire year, which the caller handles by printing a warning and moving on.

- **teamrankings.py**: Catches `requests.RequestException` (the base class for all `requests` library errors) on HTTP failures, prints an `[ERROR]` message with the URL and exception details, and returns `None` for that stat. Missing tables or empty `<tbody>` elements produce `[WARN]` messages. The outer loop continues to the next stat slug regardless.

- **warrennolan.py**: Same pattern as teamrankings -- catches `requests.RequestException`, prints an error, returns `None` on failure, and the outer loop continues to the next year.

- **Numeric parsing everywhere**: Integer and float conversions throughout all three scrapers are wrapped in `try/except (ValueError, TypeError)` blocks, defaulting to `None` on parse failure. This prevents a single malformed cell in an HTML table from crashing the entire extraction.

- **Graceful `None` propagation**: All scraping functions return `None` to indicate failure rather than raising exceptions. Callers check for `None` before writing output files, ensuring that failed scrapes do not produce empty or corrupt Parquet files. A failed scrape simply means no output file is created for that item, and it will be retried on the next run (since the cache check will find no file).

This approach has a trade-off: silent failures can mask systematic issues (e.g., a site redesign that breaks all HTML parsing) behind a wall of `[WARN]` messages that might go unnoticed in long log output. Operators should periodically verify that expected output files are present and contain reasonable row counts (e.g., ~360 teams per TeamRankings file, ~32 conferences per WarrenNolan file, ~63 games per NCAA API file).

---

## 9. Configuration Dependencies

All scrapers are tightly coupled to the `config` module at the project root. The following table summarizes which configuration values each scraper depends on:

| Config Value | `ncaa_api.py` | `teamrankings.py` | `warrennolan.py` |
|---|---|---|---|
| `RAW_DIR` | Yes | Yes | Yes |
| `SCRAPE_YEARS` | Yes | Yes | Yes |
| `NCAA_API_BASE` | Yes | -- | -- |
| `NCAA_API_DELAY` | Yes | -- | -- |
| `TEAMRANKINGS_BASE_URL` | -- | Yes | -- |
| `TEAMRANKINGS_STAT_SLUGS` | -- | Yes | -- |
| `TOURNAMENT_DATES` | -- | Yes | -- |
| `REQUEST_DELAY` | -- | Yes | -- |
| `USER_AGENT` | -- | Yes | Yes |
| `WARRENNOLAN_CONF_URL` | -- | -- | Yes |

Modifying any of these configuration values will immediately affect scraper behavior on the next run. For example, adding a new slug to `TEAMRANKINGS_STAT_SLUGS` will cause the TeamRankings scraper to fetch that stat for all configured years on its next execution, while all previously scraped stats remain cached and untouched.

---

## 10. Dependencies

The scrapers rely on the following Python packages:

| Package | Purpose |
|---|---|
| `polars` | DataFrame construction and Parquet file serialization/deserialization |
| `requests` | HTTP GET requests to all three external data sources |
| `beautifulsoup4` | HTML DOM parsing for TeamRankings and WarrenNolan pages |
| `lxml` | Fast, robust HTML parser backend used by BeautifulSoup |

Additionally, the Python standard library modules `json`, `time`, and `pathlib` are used throughout for JSON I/O, rate-limiting sleeps, and filesystem path manipulation, respectively.

All configuration is imported from the project-level `config.py` module, which must be importable from the working directory. This typically means the scrapers should be run from the project root directory, or the project root should be added to `sys.path` / `PYTHONPATH`.
