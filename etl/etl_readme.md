# ETL Module -- Extract, Transform, Load for NCAAB Tournament Prediction

This module is the backbone of the entire data pipeline. It takes messy, inconsistent raw data scraped from multiple websites and APIs, cleans it up, reconciles team identities across sources, and produces tidy Parquet files that downstream feature engineering and modeling code can consume without worrying about data quality headaches.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Execution Order](#pipeline-execution-order)
- [The Name Resolution Challenge](#the-name-resolution-challenge)
- [File Reference](#file-reference)
  - [\_\_init\_\_.py](#__init__py)
  - [build\_team\_index.py](#build_team_indexpy)
  - [process\_stats.py](#process_statspy)
  - [process\_conferences.py](#process_conferencespy)
  - [process\_tournament.py](#process_tournamentpy)
- [Input and Output Data Map](#input-and-output-data-map)
- [Dependencies and Configuration](#dependencies-and-configuration)
- [Supporting Module: team\_index.py](#supporting-module-team_indexpy)
- [Troubleshooting and Common Pitfalls](#troubleshooting-and-common-pitfalls)

---

## Overview

**The simple version:** Imagine you have a giant pile of basketball stats printed on paper, but every piece of paper uses a slightly different name for the same team. One says "UConn," another says "Connecticut," and a third says "UCONN." Before you can do anything useful, you need to figure out that all three of those names refer to the same school, give that school a single ID number, and then organize all the stats into neat tables. That is what this module does.

**The intermediate version:** The ETL module sits between the raw data scrapers (which pull information from the NCAA API, TeamRankings, and Warren Nolan) and the feature engineering layer. Its responsibilities are threefold. First, it constructs a canonical team identity index that maps every known name variant for every Division I school to a single integer `team_id`. Second, it joins dozens of individual per-stat Parquet files into a single wide table of season-level team statistics, resolving team names to IDs in the process. Third, it normalizes tournament game records into a consistent matchup format suitable for supervised learning (binary classification: did the higher-seeded team win?).

The module uses **Polars** (not Pandas) as its DataFrame library throughout, leveraging Polars' superior performance on columnar operations. All intermediate and output files use the **Apache Parquet** format for type-safe, compressed columnar storage. Each ETL script exposes a `run()` function that can be called programmatically from an orchestrator or executed directly via `python -m etl.<module_name>` thanks to `if __name__ == "__main__": run()` guards.

**The advanced version:** From an information-theoretic perspective, the ETL module performs entity resolution across heterogeneous, independently-maintained naming conventions. The NCAA API uses full institutional names and URL-safe slugs; TeamRankings uses colloquial abbreviations and regional shorthands; Warren Nolan uses yet another convention. The module implements a multi-tier resolution strategy: deterministic manual overrides for known edge cases, exact normalized string matching for trivial variants, and weighted ratio fuzzy matching (via `rapidfuzz.fuzz.WRatio`) with configurable score cutoffs for the long tail of ambiguous cases. The `WRatio` scorer is particularly well-suited to this domain because it is robust to substring reordering and partial matches, which are common in university naming conventions (e.g., "Cal St Fullerton" vs. "Cal State Fullerton"). The pipeline enforces referential integrity by dropping rows that fail resolution and logging warnings for manual review, ensuring that downstream statistical models operate on a clean, fully-joined feature matrix.

---

## Pipeline Execution Order

The ETL scripts must be executed in a specific order because later stages depend on artifacts produced by earlier stages. Running them out of order will result in missing data errors. Think of it like building a house: you need the foundation (team index) before you can put up the walls (stats, conferences, tournaments).

```
1. build_team_index.py    -->  data/raw/team_index.parquet
2. process_stats.py       -->  data/processed/team_season_stats.parquet
3. process_conferences.py -->  data/processed/conference_strength.parquet
4. process_tournament.py  -->  data/processed/tournament_matchups.parquet
```

**Why this order matters in detail:**

- `build_team_index.py` must run first because it creates the `team_index.parquet` file, which is the Rosetta Stone that translates team names from any data source into a universal `team_id`. Without this file, name resolution is impossible and the pipeline cannot produce correctly joined data.
- `process_stats.py` and `process_tournament.py` both instantiate a `TeamIndex` object at runtime, which attempts to load `team_index.parquet` from disk. If the file does not exist, these scripts will abort with an explicit error message telling you to run the team index builder first.
- `process_conferences.py` is technically independent of the team index (it operates on conference-level data, not team-level data), but it is conventionally run after the team index is built so that the entire `data/processed/` directory is populated in one sweep.

Steps 2, 3, and 4 are independent of each other and could theoretically be run in parallel, but in practice they are run sequentially for simplicity and to make log output easier to follow.

---

## The Name Resolution Challenge

This section explains the single hardest problem in the entire ETL pipeline: figuring out that two different strings refer to the same basketball team.

**The simple version:** Different websites call the same team by different names. "UConn" and "Connecticut" are the same school. "NC State" and "North Carolina State" and "N.C. State" and "N Carolina St" are all the same school. If we do not fix this, we would think there are four separate schools instead of one, and all of our statistics would be wrong. It would be like if your teacher thought you were four different students and gave you four different report cards.

**The intermediate version:** The project ingests data from three independent sources -- the NCAA API, TeamRankings, and Warren Nolan. Each source has its own editorial conventions for team names. The NCAA API tends to use full official institutional names (e.g., "Connecticut," "North Carolina State"). TeamRankings uses a mix of abbreviations, colloquialisms, and shortened forms (e.g., "UConn," "NC State," "Fla Gulf Coast"). Warren Nolan uses conference-level data so team name resolution is less critical there, but the inconsistency between the NCAA API and TeamRankings is a significant data engineering challenge.

The problem is compounded by several factors:

1. **Abbreviation diversity.** There is no standard abbreviation scheme. "Saint" might appear as "St.", "St", or "Saint". "State" might appear as "St", "St.", or "State".
2. **Disambiguation qualifiers.** Multiple schools share base names and require geographic qualifiers: "Miami (FL)" vs. "Miami (OH)", "Loyola (IL)" vs. "Loyola (MD)" vs. "Loyola Marymount", "Saint Mary's (CA)" vs. other Saint Mary's institutions.
3. **Institutional name changes.** Some schools change their names over time or use different names in different contexts (e.g., "Houston Baptist" became "Houston Christian").
4. **Regional colloquialisms.** "Ole Miss" for Mississippi, "Pitt" for Pittsburgh, "G'town" for Georgetown.
5. **Acronyms.** "UCF" for Central Florida, "UNLV" for Nevada-Las Vegas, "VCU" for Virginia Commonwealth, "FGCU" for Florida Gulf Coast, "UMBC" for Maryland-Baltimore-County.

To illustrate the severity, here is a sample of the pathological cases this pipeline must handle:

| TeamRankings Name | NCAA Canonical Name | Problem Type |
|---|---|---|
| UConn | Connecticut | Abbreviation |
| UCF | Central Florida | Abbreviation |
| Miami FL | Miami (FL) | Disambiguation (FL vs OH) |
| UMBC | Maryland-Baltimore-County | Extreme abbreviation |
| UALR | Little-Rock | Abbreviation + name change |
| Ole Miss | Mississippi | Colloquial nickname |
| N.C. State | North Carolina State | Punctuated abbreviation |
| SE Missouri St | Southeast Missouri State | Multiple abbreviations stacked |
| Texas A&M-CC | Texas-Am-Corpus | Ampersand + extreme abbreviation |
| SIU Edwardsville | SIUE | Reverse abbreviation problem |
| St. John's | St. John's (NY) | Saint vs St. + disambiguation |

These are not edge cases -- they represent a significant fraction of the 363 Division I programs.

**The advanced version:** The resolution strategy employs a three-tier cascade with monotonically decreasing confidence:

**Tier 1 -- Manual Overrides (deterministic, 100% precision).** The `MANUAL_OVERRIDES` dictionary in `team_index.py` contains approximately 240 hand-curated mappings from known variant names to canonical names. This dictionary is consulted first during resolution and handles all acronyms, well-known colloquialisms, and disambiguation cases. The dictionary was constructed empirically by running the pipeline, observing `[UNMATCHED]` and low-confidence `[FUZZY]` warnings in the logs, and adding corrections iteratively. This tier is the highest priority because it produces zero false positives: every mapping was verified by a human.

The overrides are organized into several conceptual categories:
- **Common abbreviations**: UConn, UCF, UNC, UNLV, SMU, USC, LSU, VCU, BYU, TCU, ETSU, UTSA, UTEP, FGCU, FAU, FDU, NJIT, UAB, FIU, etc.
- **Directional abbreviations**: N/S/E/W/SE/NW/C prefixes mapped to full directional words (e.g., "N Carolina" to "North Carolina", "SE Missouri St" to "Southeast Missouri State").
- **St./State abbreviations**: Dozens of mappings from "Foo St" and "Foo St." to "Foo State" for schools like Iowa State, Michigan State, Ohio State, etc.
- **Saint/St. ecclesiastical names**: St. John's, St. Mary's, St. Peter's, St. Joseph's, St. Louis, St. Francis, Mt. St. Mary's -- each with multiple variant spellings and sometimes geographic disambiguators.
- **Multi-campus disambiguations**: Miami (FL) vs Miami (OH), Loyola (IL) vs Loyola (MD) vs Loyola Marymount, St. Francis (NY) vs Saint Francis (PA), St. Thomas (MN).

**Tier 2 -- Exact Normalized Match (deterministic, high precision).** Both the query name and all canonical/alternative names are passed through a normalization function (`_normalize`) that lowercases the string, strips periods and apostrophes, and collapses whitespace. If the normalized query exactly matches a normalized canonical name or any of its registered alternative names, the match is accepted. This handles trivial variants like "St." vs "St" and capitalization differences.

**Tier 3 -- Fuzzy Matching (probabilistic, configurable precision-recall tradeoff).** For names that survive the first two tiers without a match, the system uses `rapidfuzz.process.extractOne` with the `fuzz.WRatio` scorer. The `WRatio` (Weighted Ratio) scorer computes multiple similarity metrics (simple ratio, partial ratio, token sort ratio, token set ratio) and returns a weighted best score. This is effective for handling word reordering ("Texas A&M Corpus Chris" vs. "Texas-Am-Corpus") and partial matches ("Appalachian St" vs. "Appalachian State"). The score cutoff differs by context: `build_team_index.py` uses a cutoff of 85 when building the index (and logs warnings for scores below 90), while the runtime `TeamIndex.resolve()` method uses a cutoff of 80 to be slightly more permissive during downstream processing. Names that fall below the cutoff are logged as `[UNMATCHED]` for human review.

This cascade design ensures that the overwhelming majority of name resolutions are handled by deterministic, verifiable rules (Tiers 1 and 2), while the fuzzy matching tier serves as a safety net for edge cases that have not yet been added to the manual overrides. Over time, the manual overrides dictionary grows as new edge cases are discovered, progressively shrinking the set of names that rely on fuzzy matching.

---

## File Reference

### \_\_init\_\_.py

**Path:** `/home/stonks/ncaab/etl/__init__.py`

This file is intentionally left empty. Its sole purpose is to mark the `etl/` directory as a Python package, allowing the interpreter to recognize it as a module namespace. This enables import statements like `from etl.build_team_index import run` and allows the orchestrator or other pipeline stages to import ETL functions programmatically. If you are new to Python, think of it like a "this folder is special" sign on a door -- it tells Python "yes, you can import things from inside here."

---

### build\_team\_index.py

**Path:** `/home/stonks/ncaab/etl/build_team_index.py`

**Output:** `data/raw/team_index.parquet`

**The simple version:** This script looks at all the team names from the NCAA website, then looks at all the team names from TeamRankings, and figures out which names on the two lists refer to the same school. It gives every school a number (a `team_id`) and remembers all the different names each school goes by. Then it saves all of this into a file so other scripts can use it later. It is like making a phone book where you list someone's real name and all their nicknames, so anyone can find them no matter what name they use.

**The intermediate version:** This is arguably the most critical script in the entire ETL module. Its job is deceptively simple to state -- "build a lookup table that maps every known team name variant to a single canonical ID" -- but fiendishly complex to execute correctly. The fundamental problem is that there is no universally agreed-upon naming convention for NCAA basketball teams across data providers.

The script has two operating modes depending on what raw data is available:

- **Primary mode (NCAA API available):** It reads the NCAA API schools index (`data/raw/ncaa_api/schools_index.json`), assigns sequential `team_id` values starting from 1, and then cross-references the names against all unique team names found in the TeamRankings Parquet files. The cross-referencing uses the three-tier resolution strategy (manual overrides, exact normalized match, fuzzy match). Each NCAA team gets a list of `alt_names` (stored as a JSON-serialized list within the Parquet column) containing all the TeamRankings name variants that resolved to it.

- **Fallback mode (no NCAA API data):** The script bootstraps the index entirely from TeamRankings names. It collects all unique `team_name` values from the scraped stat Parquet files, applies manual overrides to normalize them to canonical names, deduplicates by canonical name (merging alternative names along the way), and assigns sequential `team_id` values. This ensures the pipeline can operate even without the NCAA API schools index.

If this script fails or produces an incorrect mapping, every downstream join will silently corrupt data. A mismatched team ID means that Team A's offensive stats get paired with Team B's tournament seed, producing garbage features that the model trains on without complaint. This is why the script employs a belt-and-suspenders approach: automated fuzzy matching for the easy cases, plus the meticulously curated dictionary of manual overrides for the hard cases.

**The advanced version:** The function architecture is as follows:

- **`_build_from_ncaa_api() -> list[dict]`**: Reads `schools_index.json` and constructs the initial team list. Handles three possible JSON shapes polymorphically: a list of dictionaries (each with `name`/`short` and `slug`/`seo` fields), a list of plain strings, or a dictionary keyed by slug. Returns an empty list if the file does not exist, which triggers the fallback path in `run()`. Each team dict contains `team_id`, `canonical_name`, `ncaa_slug`, and an initially empty `alt_names` field (serialized as the JSON string `"[]"`).

- **`_collect_teamrankings_names() -> set[str]`**: Iterates over year directories under `data/raw/teamrankings/`, reads one Parquet file per year directory, and extracts the unique values from the `team_name` column. It only needs one file per year because the team name list is consistent across stat files within a given year, so it breaks out of the inner loop after the first successful read. This is a deliberate I/O optimization. Returns a deduplicated set of all team names ever observed across all years.

- **`_add_alt_names(teams, tr_names) -> list[dict]`**: This is the heart of the entity resolution logic. It builds a normalized lookup dictionary from the canonical names, iterates over every TeamRankings name, and attempts to match it using the three-tier cascade. Successfully matched names are appended to the `alt_names` list of their corresponding team. Manual override variants are also added proactively (even if not found in the current TeamRankings data) to pre-populate the lookup for future queries. The function prints diagnostic messages for fuzzy matches with scores below 90 (`[FUZZY XX]`) and for names that could not be matched at all (`[UNMATCHED]`). These log messages are the primary mechanism for discovering new entries that should be added to `MANUAL_OVERRIDES`.

- **`run()`**: Orchestrates the entire build process. Calls `_build_from_ncaa_api()`; if that returns an empty list, enters fallback mode where it builds the index from TeamRankings names alone (with deduplication by canonical name and ID reassignment). If NCAA API data is present and TeamRankings names are also available, calls `_add_alt_names()` to enrich the index. Serializes the final list of team dictionaries to a Polars DataFrame and writes it to Parquet. Returns the DataFrame for optional programmatic use.

**Output schema of `data/raw/team_index.parquet`:**

| Column           | Type   | Description                                                   |
|------------------|--------|---------------------------------------------------------------|
| `team_id`        | Int64  | Unique integer identifier for the team (1-indexed, sequential)|
| `canonical_name` | Utf8   | The primary, authoritative name for the team                  |
| `ncaa_slug`      | Utf8   | URL-safe slug for NCAA API lookups                            |
| `alt_names`      | Utf8   | JSON-serialized list of alternative name strings              |

**Note on `alt_names` serialization:** The `alt_names` column is stored as a JSON string (not a native Polars list type) for maximum compatibility across serialization formats and ease of human inspection. Consumers must call `json.loads()` to deserialize it into a Python list.

**Note on output location:** The team index is written to `data/raw/` rather than `data/processed/` because it is a reference artifact derived directly from raw source data, not a transformation of processed features. It is consumed by other ETL scripts as a lookup resource, placing it conceptually between raw and processed data.

---

### process\_stats.py

**Path:** `/home/stonks/ncaab/etl/process_stats.py`

**Output:** `data/processed/team_season_stats.parquet`

**The simple version:** The scrapers download one file for each basketball statistic (like "points per game" or "turnovers per game") for each year. This script takes all those separate files and glues them together side by side, so you end up with one big table where each row is a team in a given year, and each column is a different statistic. It also replaces the team names with the team ID numbers from the team index. Think of it like combining a bunch of separate report cards (one for shooting, one for rebounding, one for turnovers) into a single comprehensive transcript for every student in every school year.

**The intermediate version:** For each year in the configured `SCRAPE_YEARS` range (2015-2026, excluding 2020 due to COVID cancellation), the script reads all Parquet files from `data/raw/teamrankings/{year}/`. It starts with the first file as a base DataFrame, then performs sequential left joins on the `team_name` column to attach columns from each additional stat file. Only genuinely new columns (those not already present in the base DataFrame) are joined, preventing column duplication.

After all stat columns are assembled, the script resolves each `team_name` to a `team_id` by calling `TeamIndex.resolve()`. Teams that cannot be resolved are logged as warnings (up to 10 names are printed for diagnostic purposes) and then dropped from the output. A deduplication step ensures that if multiple TeamRankings name variants resolved to the same `team_id`, only the first occurrence is kept.

Finally, all per-year DataFrames are vertically concatenated using Polars' `diagonal_relaxed` strategy, which is critical because the set of available stat columns may differ across years (TeamRankings may add or remove statistics over time). The `diagonal_relaxed` concatenation fills missing columns with nulls rather than raising a schema mismatch error.

**The advanced version:** Several design decisions in this script warrant detailed discussion:

- **Left join strategy.** The use of left joins (rather than inner joins) on `team_name` ensures that teams appearing in the base file are retained even if they are missing from some stat files. This preserves the maximum number of rows at the cost of introducing some null values in stat columns. Downstream feature engineering code is expected to handle nulls (e.g., via imputation or exclusion).

- **Column deduplication at join time.** Before each join, the script computes the set difference between the incoming stat columns and the existing base columns. This prevents the proliferation of `_right` suffixed duplicate columns that Polars would otherwise create during joins with overlapping column names. The implementation uses a simple set membership check: `new_cols = [c for c in stat_cols if c not in existing]`.

- **`diagonal_relaxed` concatenation.** This is a Polars-specific concatenation mode that performs a union of column schemas across all DataFrames. If DataFrame A has columns `[team_id, season, stat_x]` and DataFrame B has columns `[team_id, season, stat_x, stat_y]`, the result will have columns `[team_id, season, stat_x, stat_y]` with `stat_y` nulled for rows from DataFrame A. The `relaxed` variant additionally permits type coercion (e.g., Int32 to Int64) to handle minor schema drift between years.

- **The deduplication fix (critical).** Multiple TeamRankings names can resolve to the same canonical `team_id`. For example, both "Connecticut" and "UConn" might appear in the same year's data if different stat pages use different name variants. After `team_id` resolution, these appear as duplicate rows. If this duplicated DataFrame is later joined to tournament matchups on `team_id`, a fan-out occurs: one tournament game produces multiple rows instead of one. The fix is a single line:

  ```python
  base_df = base_df.unique(subset=["team_id"], keep="first")
  ```

  Without this deduplication, a downstream `team_season_stats JOIN tournament_matchups ON team_id` would produce a Cartesian product for affected teams. If Connecticut appeared 3 times, each tournament game involving Connecticut would produce 3 rows, triple-weighting that team in the training set. The insidiousness of this bug is that it produces plausible-looking data -- the extra rows have valid stat values and correct outcomes -- making it nearly invisible without explicit row-count validation.

- **The `team_id` + `season` compound key.** Each row in the output is uniquely identified by the combination of `team_id` and `season`. The deduplication step operates within a single year's DataFrame (before concatenation), so the compound uniqueness constraint is maintained across the full output.

**Internal functions:**

- **`process_year(year: int, team_idx: TeamIndex) -> pl.DataFrame | None`**: Processes a single year's raw stat files into a wide DataFrame. Returns `None` if no data exists for the year. Encapsulates the join, resolution, null filtering, and deduplication logic.

- **`run()`**: Iterates over `config.SCRAPE_YEARS`, calls `process_year()` for each, collects non-null results, concatenates them with `pl.concat(how="diagonal_relaxed")`, and writes to Parquet. Prints per-year summaries (team count, column count) for operational monitoring.

**Approximate output shape:** ~360 rows per season (one per D-I team) times ~11 seasons = ~3,960 rows, with ~45-50 stat columns (corresponding to `config.TEAMRANKINGS_STAT_SLUGS`) plus `team_id`, `team_name`, and `season`.

---

### process\_conferences.py

**Path:** `/home/stonks/ncaab/etl/process_conferences.py`

**Output:** `data/processed/conference_strength.parquet`

**The simple version:** Basketball conferences (like the Big Ten or ACC) are not all equally strong. This script takes the rankings of how good each conference is (scraped from the Warren Nolan website) and saves them in a clean format. Later, the model can use "how strong is this team's conference?" as one of its clues for predicting who will win. It is like knowing whether a student went to a really hard school or an easy school -- the same grades mean different things depending on the difficulty level.

**The intermediate version:** Conference strength is a well-established predictive signal in March Madness modeling. Teams from power conferences (e.g., Big Ten, SEC, ACC, Big 12) play a harder regular-season schedule, which means their win-loss records and efficiency metrics are calibrated against tougher opponents. A 20-win team from the Big 12 has faced far more resistance than a 20-win team from the Patriot League. Conference-level strength ratings provide a contextual multiplier for team-level statistics.

This is the simplest of the four ETL scripts. It iterates over all configured scrape years and reads the corresponding Parquet file from `data/raw/warrennolan/{year}.parquet`. Each file contains conference-level metrics as computed by Warren Nolan's NET-based conference ranking methodology. If a file is missing the `season` column (which can happen depending on the scraper implementation), the script adds it. All per-year DataFrames are then vertically concatenated using `diagonal_relaxed` to handle any schema variation across years, and written to the output path.

**The advanced version:** This script is deliberately simple because conference strength is a conference-level attribute, not a team-level attribute. The thorny team name resolution problem does not apply here. The mapping from teams to conferences is handled downstream during feature engineering, where each team's conference membership is looked up and joined with this conference strength table to produce a team-level conference strength feature.

The Warren Nolan data typically includes metrics derived from the NCAA's NET (NCAA Evaluation Tool) ranking system, which incorporates game results, strength of schedule, game location, scoring margin (capped), and net offensive and defensive efficiency. The specific columns in the output depend on what Warren Nolan publishes for each year, but commonly include conference name, overall conference NET ranking, conference record aggregates, and related strength-of-schedule indicators.

Because `diagonal_relaxed` concatenation is used, columns that appear in some years but not others are filled with null. This is the correct behavior since Warren Nolan's reporting format may evolve over time.

**Output schema of `data/processed/conference_strength.parquet`:**

| Column | Type | Description |
|---|---|---|
| `season` | Int64 | Tournament year |
| *(conference ranking columns)* | Various | Conference names, rankings, NET ratings, and other strength metrics as provided by Warren Nolan |

---

### process\_tournament.py

**Path:** `/home/stonks/ncaab/etl/process_tournament.py`

**Output:** `data/processed/tournament_matchups.parquet`

**The simple version:** The NCAA website gives us raw information about every March Madness game -- who played, what the score was, and what seed each team had. But the data is messy: sometimes one team is listed as "away" and sometimes as "home," and it is not consistent. This script reorganizes every game so that the team with the better seed (the lower number, like a 1-seed) is always called "team_a" and the team with the worse seed is always "team_b." It also figures out if team_a won (1) or lost (0). This makes it much easier for the prediction model to learn patterns, because it always knows which team is supposed to be the favorite.

**The intermediate version:** This script produces the single most important output in the entire pipeline: the labeled training data. Each row represents one NCAA tournament game, with the teams identified by canonical IDs (joinable to stats), their seeds, their scores, and a binary label indicating which team won. This is what the prediction model actually trains on. The stats file tells you "how good was each team during the regular season," and this file tells you "when those teams met in the tournament, who won?" The model learns the relationship between the former and the latter.

The script reads per-year Parquet files from `data/raw/ncaa_api/tournament_games/{year}.parquet`. For each game, it resolves both the away and home team names to `team_id` values using the `TeamIndex`. It then normalizes the matchup orientation so that `team_a` is always the higher-seeded team (the one with the numerically lower seed value). The `winner` column is set to 1 if `team_a` won and 0 if `team_b` won. This consistent orientation is essential for supervised learning because the model is trained to predict the probability that the higher-seeded team wins, rather than predicting an arbitrary "home" or "away" outcome.

After processing all years, the script performs quality assurance checks: it prints the total number of games, a per-season breakdown, and warns about any games where team ID resolution failed. Each tournament typically yields approximately 67 games (63 main bracket + 4 First Four).

**The advanced version:** The matchup normalization logic handles several edge cases with careful determinism:

1. **Both seeds available (normal case).** The team with the lower seed number (better seed) is designated `team_a`. In tournament play, a 1-seed facing a 16-seed means the 1-seed becomes `team_a`. The comparison is `away_seed <= home_seed`; if true, the away team is `team_a`.

2. **Only one seed available.** The team whose seed is known becomes `team_a`. This can occur in play-in games or due to data quality issues where seeding information is partially populated.

3. **Neither seed available.** Alphabetical ordering on team name is used as a deterministic tiebreaker (`away_name <= home_name`). This is arbitrary but guarantees reproducibility -- the same game will always be oriented the same way regardless of when the script is run.

4. **Equal seeds.** When two teams share the same seed number (possible in later rounds where, for example, two 1-seeds meet in the Final Four), the `away_seed <= home_seed` comparison evaluates to `True`, so the away team becomes `team_a`. This is an arbitrary but deterministic convention.

The `winner` encoding convention is particularly useful for modeling because the base rate of `winner=1` directly corresponds to the empirical frequency of higher seeds winning their games, providing a natural calibration reference. Historically, higher seeds win approximately 68% of first-round games, with the rate varying by seed matchup (1 vs. 16 is nearly 99%; 8 vs. 9 is approximately 50%).

Note that team ID resolution is performed via explicit row-by-row iteration (not vectorized Polars expressions) because the `TeamIndex.resolve()` method involves dictionary lookups and optional fuzzy matching that cannot be expressed as columnar operations. This is acceptable because the total number of tournament games across all years is relatively small (~731 rows).

The script preserves games with null team IDs in the output rather than silently dropping them, so that the extent of resolution failures is visible in downstream analysis. However, these rows would typically be filtered out before model training.

**Internal functions:**

- **`run()`**: The sole public function. Instantiates a `TeamIndex`, iterates over years, reads raw game Parquet files, resolves team names, normalizes matchup orientation, collects all matchup dictionaries into DataFrames, concatenates with `diagonal_relaxed`, runs QA checks, and writes to Parquet.

**Output schema of `data/processed/tournament_matchups.parquet`:**

| Column           | Type   | Description                                                        |
|------------------|--------|--------------------------------------------------------------------|
| `team_a_id`      | Int64  | `team_id` of the higher-seeded team                                |
| `team_b_id`      | Int64  | `team_id` of the lower-seeded team                                 |
| `team_a_name`    | Utf8   | Display name of team_a (as it appeared in the source data)         |
| `team_b_name`    | Utf8   | Display name of team_b (as it appeared in the source data)         |
| `team_a_seed`    | Int64  | Tournament seed of team_a (lower number = stronger)                |
| `team_b_seed`    | Int64  | Tournament seed of team_b (higher number = weaker)                 |
| `team_a_score`   | Int64  | Final score of team_a                                              |
| `team_b_score`   | Int64  | Final score of team_b                                              |
| `winner`         | Int64  | Binary label: 1 if team_a won, 0 if team_b won                    |
| `bracket_round`  | Utf8   | Tournament round (e.g., "First Round", "Sweet 16", "Final Four")   |
| `bracket_region` | Utf8   | Tournament region (e.g., "East", "West", "South", "Midwest")      |
| `season`         | Int64  | The season year (e.g., 2024 for the 2023-2024 season)             |
| `game_id`        | Utf8   | Unique game identifier from the NCAA API                           |
| `date`           | Utf8   | Game date string                                                   |

**Approximate output size:** ~731 historical matchups across 11 tournament years (2015-2019, 2021-2026; 2020 excluded due to COVID cancellation).

---

## Input and Output Data Map

The following diagram summarizes the data flow through the ETL module, showing which raw inputs feed into which scripts and what processed outputs they produce.

```
RAW DATA (scraped)                    ETL SCRIPTS                 PROCESSED DATA
========================          ===================          ==========================

data/raw/ncaa_api/
  schools_index.json ──────┐
                           ├──> build_team_index.py ──> data/raw/team_index.parquet
data/raw/teamrankings/     │                                       |
  {year}/*.parquet ────────┘                                       |
       |                                                           v
       |                                                    (used as lookup by)
       |                                                           |
       └───────────────────────> process_stats.py ──────> data/processed/
                                  (uses TeamIndex)           team_season_stats.parquet

data/raw/warrennolan/
  {year}.parquet ──────────────> process_conferences.py ──> data/processed/
                                                             conference_strength.parquet

data/raw/ncaa_api/
  tournament_games/
    {year}.parquet ────────────> process_tournament.py ──> data/processed/
                                  (uses TeamIndex)           tournament_matchups.parquet
```

---

## Dependencies and Configuration

### Python Dependencies

| Package      | Purpose                                                                 |
|--------------|-------------------------------------------------------------------------|
| `polars`     | High-performance DataFrame library used for all Parquet I/O and joins   |
| `rapidfuzz`  | Fuzzy string matching library (C++ backend) for team name resolution    |

### Configuration (from `config.py`)

All ETL scripts import from `/home/stonks/ncaab/config.py`, which provides:

| Constant | Value | Used By |
|---|---|---|
| `RAW_DIR` | `data/raw/` | All scripts (input paths) |
| `PROCESSED_DIR` | `data/processed/` | `process_stats`, `process_conferences`, `process_tournament` (output paths) |
| `SCRAPE_YEARS` | `[2015, 2016, ..., 2019, 2021, ..., 2026]` | All processing scripts (year iteration). 2020 is excluded due to COVID cancellation. |
| `TEAMRANKINGS_STAT_SLUGS` | List of ~45 stat URL slugs | Implicitly defines the stat columns available in `process_stats.py` output |
| `TOURNAMENT_DATES` | Dict mapping year to cutoff date | Used by scrapers (not directly by ETL, but defines when raw data was captured) |

---

## Supporting Module: team\_index.py

**Path:** `/home/stonks/ncaab/team_index.py`

While not inside the `etl/` directory, this module is critically important to the ETL pipeline and is imported by both `build_team_index.py` and the processing scripts. It provides three key components:

1. **`MANUAL_OVERRIDES` dictionary**: Approximately 240 hand-curated variant-to-canonical name mappings. Used by `build_team_index.py` during index construction and by `TeamIndex.resolve()` during runtime resolution. This is the single most labor-intensive artifact in the codebase and must be maintained whenever a new data source introduces a previously unseen name variant, or when a university undergoes a rebrand.

2. **`_normalize(name: str) -> str` function**: String normalization that lowercases, strips periods, normalizes apostrophes (curly to straight), and collapses whitespace via regex. This is the first step in any name comparison and eliminates trivial formatting differences.

3. **`TeamIndex` class**: The runtime name resolution engine. Loads `team_index.parquet` and builds in-memory lookup dictionaries (`_name_to_id` and `_id_to_name`). Key methods:
   - **`resolve(name) -> Optional[int]`**: The primary resolution method. Implements the cascade: manual overrides --> exact normalized match --> fuzzy match (WRatio, `score_cutoff=80`). Note that the runtime cutoff of 80 is slightly lower than the 85 used during index construction, providing a small additional margin for names that only appear at query time.
   - **`resolve_or_warn(name, source) -> Optional[int]`**: Same as `resolve()` but prints a warning on failure. Used in contexts where silent failure is undesirable.
   - **`get_name(team_id) -> Optional[str]`**: Reverse lookup from ID to canonical name.
   - **`df` property**: Exposes the raw team index DataFrame for inspection. Returns `None` if the index file was not found during initialization.

---

## Troubleshooting and Common Pitfalls

### "Team index not found. Run build_team_index first."

This error occurs when `process_stats.py` or `process_tournament.py` is run before `build_team_index.py`. The fix is straightforward: run `build_team_index.py` first. Always follow the execution order documented in [Pipeline Execution Order](#pipeline-execution-order).

### Large numbers of `[UNMATCHED]` warnings during index build

If `build_team_index.py` produces many `[UNMATCHED]` lines, it means TeamRankings is using team names that cannot be resolved to any NCAA API school, even with fuzzy matching at the 85 threshold. The fix is to add explicit entries to the `MANUAL_OVERRIDES` dictionary in `/home/stonks/ncaab/team_index.py`. Each entry maps the unmatched TeamRankings name (as the dictionary key) to the corresponding canonical name from the NCAA API schools index (as the dictionary value). After adding overrides, rebuild the team index.

### `[FUZZY XX]` warnings with low scores (85-89)

These warnings indicate that a fuzzy match was found but with relatively low confidence. The match may or may not be correct. It is strongly recommended to verify each low-confidence match manually and, if correct, add it to `MANUAL_OVERRIDES` to promote it from Tier 3 (probabilistic) to Tier 1 (deterministic) resolution. If the match is incorrect, add the correct mapping to `MANUAL_OVERRIDES` to override the erroneous fuzzy match. Over time, this iterative process drives the number of Tier 3 resolutions toward zero.

### `[WARN] N unmatched teams in YYYY` in process_stats.py

This means N teams in a given year could not be resolved to a `team_id` and were dropped from the output. Common causes: (a) the team index is stale and needs rebuilding after new data was scraped, (b) TeamRankings introduced a new name variant not previously seen, or (c) a new school joined Division I. Resolution: add the missing name to `MANUAL_OVERRIDES`, rebuild the team index, and rerun `process_stats.py`.

### `[WARN] N games with unresolved team IDs` in process_tournament.py

This is a high-severity warning because it means labeled training data is being degraded. Games with null team IDs cannot be properly joined to team statistics during feature engineering. Same resolution as above: update overrides, rebuild index, reprocess.

### Schema drift across years in team_season_stats.parquet

Because TeamRankings may add or remove statistics from year to year, the set of columns in the output Parquet file may not be perfectly rectangular. Columns that exist in some years but not others will contain null values for the missing years. Downstream code should use null-aware operations (e.g., `pl.col("stat").fill_null(strategy="mean")`) or explicitly filter to columns with sufficient data coverage across the year range of interest.

### The COVID-2020 gap

The year 2020 is excluded from `config.SCRAPE_YEARS` because the NCAA tournament was cancelled that year due to the COVID-19 pandemic. There is no tournament data to process and the regular-season statistics are incomplete. The pipeline handles this gracefully by simply not including 2020 in its iteration range. Any analysis spanning 2015-2026 should account for this gap (11 seasons, not 12).

### Deduplication failures manifesting downstream

If you observe that certain teams appear to be over-represented in model training results, or that join operations produce more rows than expected, suspect a deduplication issue in `process_stats.py`. The `unique(subset=["team_id"], keep="first")` call should prevent this, but if new name variants cause the same team to appear under different `team_id` values (due to an incomplete team index), the deduplication will not catch it because the `team_id` values are genuinely different. The root cause in this case is an incomplete team index, and the fix is to update `MANUAL_OVERRIDES` and rebuild.
