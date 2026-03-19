# NCAAB March Madness Prediction Pipeline

An end-to-end NCAA Men's Basketball tournament prediction system that scrapes historical data from multiple sources, engineers hundreds of features focused on ball control and away performance, trains a specialist stacking ensemble of machine learning models, applies conformal prediction for calibrated uncertainty quantification, and predicts the complete 2026 March Madness bracket with realistic upset distributions.

---

## Table of Contents

1.  [Overview](#1-overview)
2.  [Quick Start](#2-quick-start)
3.  [Project Architecture](#3-project-architecture)
4.  [Data Sources](#4-data-sources)
    - [TeamRankings.com](#41-teamrankingscom)
    - [NCAA API](#42-ncaa-api)
    - [Warren Nolan](#43-warren-nolan)
5.  [Pipeline Stages](#5-pipeline-stages)
    - [Scraping](#51-scraping)
    - [Team Index](#52-team-index)
    - [ETL Processing](#53-etl-processing)
    - [Feature Engineering](#54-feature-engineering)
    - [Feature Selection](#55-feature-selection)
    - [Model Training](#56-model-training)
    - [Evaluation](#57-evaluation)
    - [Bracket Prediction](#58-bracket-prediction)
6.  [Machine Learning Architecture](#6-machine-learning-architecture)
    - [Specialist Models](#61-specialist-models)
    - [Stacking Meta-Learner](#62-stacking-meta-learner)
    - [Conformal Prediction](#63-conformal-prediction)
    - [Upset Calibration](#64-upset-calibration)
7.  [Key Results](#7-key-results)
8.  [Configuration](#8-configuration)
9.  [Docker Setup](#9-docker-setup)
10. [Documentation Links](#10-documentation-links)
11. [Dependencies](#11-dependencies)

---

## 1. Overview

**ELI5:** Imagine you are trying to guess which basketball team will win every single game in the big college basketball tournament. Instead of just guessing randomly or picking the team with the better ranking, this project looks at a huge pile of numbers about how each team has played all season -- things like how often they lose the ball, how well they shoot free throws, and how they do when playing at someone else's gym -- and it uses computers to find patterns in those numbers that help predict who wins. Then it draws a big bracket picture showing all the games and who it thinks will win each one.

**Intermediate Explanation:** This repository implements a complete machine learning pipeline for predicting the outcomes of the NCAA Men's Basketball Tournament (commonly known as March Madness). The pipeline is composed of several sequential stages: data collection from three complementary web sources, entity resolution across those sources, extract-transform-load (ETL) processing, feature engineering producing 557+ candidate features, feature selection via correlation analysis and information-theoretic ranking, training of a specialist stacking ensemble, conformal prediction for uncertainty calibration, and finally bracket generation with historically-grounded upset injection. The system is trained on tournament data from the 2015 through 2025 seasons (excluding the cancelled 2020 season due to the COVID-19 pandemic), and the 2026 tournament is the prediction target that is never included in any training data whatsoever.

**Advanced Technical Detail:** The core innovation of this pipeline lies in its specialist stacking architecture. Rather than training a single monolithic model, the system trains four base learners -- each one a specialist in a distinct regime of the tournament prediction problem space. Model A (XGBoost) operates as a generalist across all 112 selected features and all training matchups. Model B (LightGBM) is a close-game specialist trained exclusively on matchups where the absolute seed differential is seven or fewer, using a curated subset of 25 features emphasizing recency indicators, rebounding metrics, and away-performance gaps. Model C (Logistic Regression) targets later-round dynamics (Sweet Sixteen onward) with 23 features centered on free-throw shooting, ball security, and defensive intensity. Model D (Random Forest) provides ensemble diversity through a compact 17-feature representation. A logistic regression meta-learner combines these base predictions with disagreement signals, contextual features, and round-by-seed-difference interaction terms. The entire system is wrapped in a conformal prediction framework that provides round-conditional calibrated prediction sets at 80%, 85%, 90%, and 95% confidence levels, achieving empirical coverage within 0.2 percentage points of the nominal level at every tier.

---

## 2. Quick Start

**ELI5:** Here is how you turn the project on and make it go. It is like pressing the "start" button on a dishwasher -- you put things in, press a button, and the machine does everything in the right order.

**Getting Started:**

The fastest way to run the entire pipeline from scratch is via Docker:

```bash
# Clone the repository and enter the project directory
cd /home/stonks/ncaab

# Build and run everything in a container
docker-compose up --build
```

If you prefer to run outside Docker, ensure you have Python 3.12+ installed:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (scraping through bracket generation)
python pipeline.py
```

The pipeline orchestrator (`pipeline.py`) will execute every stage in the correct order:

1. Scrape data from all three sources
2. Build and resolve the team name index
3. Run ETL processing on stats, conferences, and tournament results
4. Engineer and select features
5. Train the stacking ensemble
6. Evaluate model performance
7. Generate the 2026 bracket prediction and HTML visualization

**Selective Stage Execution:**

The pipeline supports stage-based execution, so you can run individual stages or resume from a specific point if earlier stages have already completed and their output artifacts (parquet files) exist in the `data/` directory.

---

## 3. Project Architecture

**ELI5:** Think of this project like a factory assembly line. Raw materials (basketball statistics from websites) come in at one end. Each station on the assembly line does one specific job -- cleaning, combining, measuring, learning, guessing -- and at the very end, a finished bracket poster rolls off the line.

**Structural Overview:**

The project follows a sequential pipeline architecture where each module reads the output artifacts of the preceding module and writes its own artifacts for the next module downstream. All intermediate data is stored as Polars DataFrames serialized to the efficient Apache Parquet columnar format in the `data/` directory.

```
ncaab/
|-- config.py              # Central configuration: years, URLs, stat slugs, paths
|-- pipeline.py            # Orchestrator: runs all stages in sequence
|-- team_index.py          # Team name resolution via fuzzy matching + overrides
|-- requirements.txt       # Python package dependencies
|-- Dockerfile             # Container image definition (python:3.12-slim)
|-- docker-compose.yml     # Container orchestration
|
|-- scrapers/              # Data acquisition from external sources
|   |-- teamrankings.py    #   ~45 stats per team per year from TeamRankings.com
|   |-- ncaa_api.py        #   Tournament results, seeds, and school index from NCAA API
|   |-- warrennolan.py     #   Conference NET rankings from Warren Nolan
|
|-- etl/                   # Extract-Transform-Load processing
|   |-- build_team_index.py    # Bootstrap team index from NCAA API + fuzzy match TR names
|   |-- process_stats.py       # Join stat parquets into wide tables, resolve names, dedup
|   |-- process_conferences.py # Conference strength features from Warren Nolan data
|   |-- process_tournament.py  # Normalize tournament games into ordered matchups
|
|-- features/              # Feature engineering and selection
|   |-- build_features.py  #   557+ raw features: diffs, pctdiffs, composites, seeds
|   |-- select_features.py #   Correlation dedup + MI + MCC ranking -> 112 features
|
|-- models/                # Machine learning models and prediction
|   |-- stacking.py        #   Specialist stacking ensemble (primary training pipeline)
|   |-- train.py           #   Original simple ensemble trainer (superseded by stacking.py)
|   |-- evaluate.py        #   Model evaluation metrics and diagnostics
|   |-- predict.py         #   Predictions on conference tournament games
|   |-- bracket.py         #   2026 bracket prediction + interactive HTML visualization
|
|-- data/                  # All intermediate and final data artifacts (parquet files)
|-- tests/                 # Test suite
```

**Detailed Architectural Principles:**

The architecture adheres to several deliberate design decisions. First, every module is stateless with respect to its predecessors: it reads parquet files from disk rather than holding in-memory references, which means any stage can be re-run independently as long as its input artifacts exist. Second, all tabular data flows through Polars rather than Pandas, leveraging Polars' lazy evaluation engine, Rust-based execution backend, and superior memory efficiency for the hundreds of columns produced during feature engineering. Third, the team name resolution layer (`team_index.py`) acts as a universal join key factory -- every module that needs to combine data across sources resolves team names through the `TeamIndex` class, ensuring referential integrity across the three fundamentally different naming conventions used by TeamRankings, the NCAA API, and Warren Nolan.

---

## 4. Data Sources

**ELI5:** To predict basketball games, you need to know things about the teams. We get our information from three different websites, kind of like how you might ask three different friends who all watch basketball to tell you what they know. Each friend knows different things, and when you put all their knowledge together, you know a whole lot.

### 4.1 TeamRankings.com

**What It Provides:** TeamRankings.com is the single richest data source in the pipeline, contributing approximately 45 distinct team-level statistical measures for every team in every season from 2015 through 2026. These statistics span a comprehensive range of basketball performance dimensions.

**Statistics Collected (representative sample across categories):**

| Category | Example Stats |
|---|---|
| Offensive Efficiency | Points per game, scoring margin, offensive efficiency rating |
| Defensive Efficiency | Opponent points per game, defensive efficiency rating |
| Ball Control | Assist-to-turnover ratio, turnovers per game |
| Shooting | Field goal percentage, three-point percentage, free-throw percentage |
| Rebounding | Offensive rebounds, defensive rebounds, total rebounding margin |
| Other | Blocks per game, steals per game, win percentage |

**Temporal Dimension:** For every single one of these ~45 statistics, the scraper collects not just a single season-long value, but multiple temporal windows and venue splits. Specifically, the columns captured for each stat include:

- **season** -- The full-season average, representing the most stable and least noisy estimate of a team's ability in that dimension.
- **last3** -- The average over the team's last three games, capturing very recent form and momentum heading into the tournament.
- **last1** -- The most recent single-game value, providing the most current but also the noisiest signal.
- **home** -- The team's average in home games only, which serves as a baseline for venue-adjusted performance.
- **away** -- The team's average in away games only, which is critically important because all tournament games are played at neutral sites (or effectively away for all but host-region teams). The gap between home and away performance for a given stat is one of the most informative engineered features in the entire pipeline.
- **prior** -- The stat value from the prior season, adding a longitudinal dimension that captures program trajectory.

**Date Parameter Strategy:** The scraper uses the `?date=` URL parameter to request statistics as of the day immediately preceding the start of that year's tournament (the First Four date). This is absolutely essential for avoiding data leakage -- if you scraped current-day statistics, the values would include games that occurred after the tournament bracket was set, which the model should not have access to at prediction time.

**Request Volume:** A full scrape across all years, all stats, and all temporal windows generates approximately 400 HTTP requests. Rate limiting is configured in `config.py` to be respectful of the source server.

### 4.2 NCAA API

**What It Provides:** The NCAA API (served via `ncaa-api.henrygd.me`) provides two critical datasets:

1. **Tournament Game Results:** For every completed tournament (2015-2025, excluding 2020), the API returns the complete set of game results including the seeds of both participating teams, the final scores, the bracket round (First Four, Round of 64, Round of 32, Sweet Sixteen, Elite Eight, Final Four, Championship), and the region (East, West, South, Midwest). These results form the target variable for model training: given two teams and their features, which team won?

2. **Schools Index:** A comprehensive directory of 1,173 NCAA schools with their canonical names and identifiers. This index serves as the backbone of the team name resolution system, providing the authoritative "ground truth" name for every school that has ever appeared in the tournament.

### 4.3 Warren Nolan

**What It Provides:** Warren Nolan publishes conference-level NET rankings, which measure the aggregate strength of every conference in college basketball. This data is available from the 2021 season onward through 2026. The conference NET ranking captures the quality of competition a team faces within its conference, which is a signal that team-level statistics alone cannot fully represent. A team that posts modest raw statistics while playing in a brutally strong conference (e.g., the Big 12 or SEC) may in fact be more formidable than its raw numbers suggest, compared to a team with flashy statistics compiled against weaker conference opposition.

**Coverage Limitation:** Because Warren Nolan conference rankings are only available from 2021 onward, features derived from this source carry missing values for the 2015-2019 seasons. The feature engineering and model training stages handle this gracefully through appropriate imputation or by allowing the tree-based models (XGBoost, LightGBM, Random Forest) to route missing values natively.

---

## 5. Pipeline Stages

**ELI5:** The pipeline is like following a recipe step by step. You cannot frost a cake before you bake it, and you cannot bake it before you mix the ingredients. Each step below must happen in order because each step needs the results from the step before it.

### 5.1 Scraping

**Module:** `scrapers/`

**What Happens:** The scraping stage reaches out to each of the three external data sources over HTTP, downloads the raw data, and writes it to parquet files in the `data/` directory. Each scraper is an independent module that can be invoked on its own.

- `teamrankings.py` iterates over every configured stat slug (45 of them) for every configured year (2015-2026 excluding 2020), constructs the appropriate URL with the `?date=` parameter set to the day before that year's tournament, parses the HTML response with BeautifulSoup and lxml, extracts the tabular statistics, and writes one parquet file per stat per year. This is the most time-consuming stage due to the ~400 HTTP requests involved and the rate limiting applied between requests.
- `ncaa_api.py` queries the NCAA API for tournament bracket data for each completed year and for the full schools directory. This is significantly faster since the API returns structured JSON.
- `warrennolan.py` scrapes conference NET rankings for the years 2021 through 2026.

**Output Artifacts:** Raw parquet files in `data/`, organized by source and year.

### 5.2 Team Index

**Module:** `team_index.py`

**ELI5:** Different websites call the same basketball team by different names. One website might say "UConn" while another says "Connecticut" and a third says "Connecticut Huskies." The team index is like a big translation dictionary that figures out they are all the same team.

**Technical Detail:** The `TeamIndex` class is one of the most critical components in the entire pipeline because it is the mechanism by which data from three fundamentally different naming conventions is joined into a single unified dataset. Without accurate name resolution, the entire downstream pipeline would produce garbage.

The resolution strategy operates in two tiers:

1. **Manual Overrides (highest priority):** A hardcoded dictionary of 150+ explicit mappings for teams whose names are sufficiently ambiguous, abbreviated, or inconsistent across sources that automated matching would fail or produce incorrect results. Examples include abbreviations like "UTSA" vs. "UT San Antonio," regional qualifiers like "Miami (FL)" vs. "Miami (OH)," and legacy name changes.

2. **Fuzzy Matching (fallback):** For teams not covered by manual overrides, the system uses the `rapidfuzz` library to perform fuzzy string matching against the canonical NCAA schools index. This handles minor variations in spacing, punctuation, abbreviation, and word order that do not rise to the level of requiring a manual override.

The `resolve()` method is called throughout the ETL layer whenever a team name from any source needs to be converted to the canonical internal identifier used as the join key across all datasets.

### 5.3 ETL Processing

**Module:** `etl/`

**ELI5:** ETL stands for Extract, Transform, Load. It is like taking a big messy pile of LEGOs from different boxes, sorting them by color and size, snapping some together that belong together, throwing away any duplicates, and then putting the organized sets neatly on a shelf so they are ready to build with.

**Submodules:**

- **`build_team_index.py`** -- Bootstraps the master team index by downloading the full NCAA API schools directory (1,173 schools) and then fuzzy-matching every TeamRankings team name against that directory. The result is a lookup table mapping every name variant to a canonical team identifier.

- **`process_stats.py`** -- The workhorse of the ETL layer. For each year, this module reads all of the per-stat parquet files produced by the TeamRankings scraper, joins them into a single wide table where each row is a team-year and each column is a stat-window combination (e.g., `assist_turnover_ratio_season`, `assist_turnover_ratio_away`, `scoring_margin_last3`), resolves all team names to canonical IDs via the team index, and deduplicates any teams that appear multiple times due to naming edge cases. The output is one wide parquet file per year, ready for feature engineering.

- **`process_conferences.py`** -- Reads the Warren Nolan conference NET rankings and transforms them into per-team conference strength features. Each team inherits the aggregate NET ranking of its conference, providing a measure of schedule strength that complements the team-level statistics.

- **`process_tournament.py`** -- Reads the NCAA API tournament results and normalizes every game into an ordered matchup where `team_a` is always the higher-seeded (i.e., lower seed number, therefore "better") team. This normalization is essential because the model learns to predict "does the higher-seeded team win?", which provides a consistent and interpretable target variable. The output includes the computed binary `winner` column (1 if the higher-seeded team won, 0 if the lower-seeded team pulled off an upset).

### 5.4 Feature Engineering

**Module:** `features/build_features.py`

**ELI5:** Imagine you know that Team A scores 80 points per game and Team B scores 70 points per game. That is interesting, but what is even more interesting is the difference: Team A scores 10 more points per game. And what about when they play away from home? Maybe Team A drops to 72 while Team B only drops to 68. That away-game gap tells you something different. Feature engineering takes all the raw numbers and computes hundreds of these kinds of comparisons and combinations that help the computer learn which patterns matter most.

**Feature Categories (557+ total raw features):**

**Seed-Based Features:** These are derived purely from the tournament seeding of the two teams in a matchup. Despite their simplicity, seed-based features are among the most powerful predictors because seeding itself encodes the collective judgment of the NCAA selection committee.

| Feature | Description | Notes |
|---|---|---|
| `seed_diff` | Absolute difference in seeds (team_a - team_b) | Lower is closer matchup |
| `seed_sum` | Sum of both seeds | Proxy for overall matchup quality |
| `seed_product` | Product of both seeds | Amplifies separation for mismatched seeds |
| `seed_pct_diff` | Percentage difference in seeds relative to sum | The number one ranked feature in the entire model |
| `log_seed_ratio` | Logarithm of seed ratio | Handles nonlinear seed effects |

**Stat Differential Features (`diff_{stat}`):** For every one of the ~45 statistics across every temporal window (season, last3, last1, home, away, prior), the pipeline computes the raw arithmetic difference between team_a and team_b. This yields the single largest block of features and captures the absolute advantage one team holds over the other in each measured dimension.

**Percentage Differential Features (`pctdiff_{stat}`):** In addition to raw differences, the pipeline computes percentage differences normalized by the average of the two teams' values. This captures relative rather than absolute advantages, which can be more informative when comparing teams at different absolute performance levels. A 5-point scoring margin difference between two teams averaging 85 points per game is proportionally less significant than the same 5-point difference between two teams averaging 60 points per game.

**Away Performance Gap Features (`diff_awaygap_{stat}`):** For every stat that has both a season-average column and an away-average column, the pipeline computes the "away gap" -- how much a team's performance degrades when playing away versus their season average -- and then takes the differential of that gap between the two teams. This is one of the most novel and important feature families in the entire pipeline. The intuition is that tournament games are played at neutral sites, which for most teams more closely resembles an away environment than a home environment. Teams that maintain their performance levels away from home are systematically more resilient in tournament settings, and this feature directly quantifies that resilience differential.

**Composite Index Features:** Several handcrafted composite features combine multiple raw statistics into single indices that capture higher-level basketball concepts:

- **`efficiency_gap`** -- Combines offensive and defensive efficiency differentials into a single net efficiency advantage score.
- **`ball_control_index`** -- Synthesizes assist-to-turnover ratio, turnover rate, and steal rate into a measure of how well a team maintains possession versus forcing opponent turnovers. Ball control is disproportionately important in tournament play because the single-elimination format amplifies variance, and teams that control possessions reduce variance.
- **`shooting_index`** -- Combines field goal percentage, three-point percentage, and free-throw percentage into a composite shooting proficiency measure.

**Round and Context Features:**

| Feature | Description |
|---|---|
| `round_number` | Numeric encoding of the tournament round (0 = First Four through 6 = Championship) |
| `round_group` | Coarser grouping (0 = early rounds, 1 = Sweet 16/Elite 8, 2 = Final Four, 3 = Championship) |
| `seed_diff_bucket` | Discretized seed difference for interaction modeling |

### 5.5 Feature Selection

**Module:** `features/select_features.py`

**ELI5:** When you have 557 different numbers describing two basketball teams, many of those numbers are saying almost the same thing in slightly different ways. If you try to use all of them, the computer gets confused by the repetition. Feature selection is like picking the 112 most useful and most unique numbers out of the 557, throwing away the ones that are just saying the same thing as another number that is already kept.

**Three-Stage Selection Process:**

**Stage 1: Pearson Correlation Deduplication (threshold = 0.90)**

The first stage identifies pairs of features that are extremely highly correlated with each other (Pearson correlation coefficient >= 0.90). When two features carry essentially the same information, keeping both adds no predictive power but does increase model complexity, training time, and overfitting risk. For each highly-correlated pair, the feature with lower univariate predictive power (measured in Stage 2) is dropped. This stage is purely about removing redundancy.

**Stage 2: Mutual Information Ranking**

After deduplication, the remaining features are ranked by their Mutual Information (MI) with the binary target variable (did the higher-seeded team win?). Mutual Information is a non-parametric, information-theoretic measure that captures both linear and nonlinear statistical dependencies between a feature and the target. Unlike Pearson correlation, MI can detect features that have a strong nonlinear relationship with the outcome even if their linear correlation is weak.

**Stage 3: Matthews Correlation Coefficient (MCC) Combined Ranking**

The features are also ranked by their Matthews Correlation Coefficient with the target. MCC is a balanced measure that accounts for all four cells of the confusion matrix (true positives, true negatives, false positives, false negatives) and is particularly appropriate for binary classification tasks. The final ranking is a combined score that integrates both the MI ranking and the MCC ranking, and the bottom 50% of features by this combined ranking are dropped.

**Result:** The selection pipeline reduces the feature space from 557+ raw features down to 112 selected features that are both individually informative and collectively non-redundant. These 112 features are used by Model A (the XGBoost generalist). The specialist models (B, C, D) use further-curated subsets of these 112 features, as described in the Machine Learning Architecture section.

### 5.6 Model Training

**Module:** `models/stacking.py`

See [Section 6: Machine Learning Architecture](#6-machine-learning-architecture) for the full detailed treatment of the training pipeline, specialist model design, stacking meta-learner, and conformal prediction framework.

**Validation Strategy:** The training pipeline uses Leave-One-Year-Out (LOYO) cross-validation. For each fold, one year's tournament data is held out as the validation set and all other years are used for training. This is the most rigorous temporal validation strategy available because it perfectly simulates the real prediction scenario: the model must predict a tournament it has never seen any games from. There is no random shuffling that could allow temporal leakage.

### 5.7 Evaluation

**Module:** `models/evaluate.py`

**ELI5:** After the computer has learned from old tournaments, we need to check how well it actually does. We show it old tournaments it was not allowed to peek at while learning, and we count how often its guesses are right. We also check that when it says it is "90% sure," it really is right about 90% of the time.

**Metrics Computed:**

- **Accuracy:** Raw percentage of correctly predicted matchup outcomes across all LOYO folds.
- **Log-Loss:** Measures the quality of the predicted probabilities (not just the binary predictions). A model that says "60% chance team A wins" and team A does win gets a better log-loss score than a model that says "51% chance team A wins" for the same correct prediction.
- **Brier Score:** Mean squared error of the predicted probabilities versus the binary outcomes. Similar purpose to log-loss but with different sensitivity characteristics.
- **Calibration:** Empirical assessment of whether the model's predicted probabilities match observed frequencies. A well-calibrated model should win approximately 70% of the matchups it assigns a 70% win probability.
- **Conformal Coverage:** Verification that the conformal prediction intervals achieve their nominal coverage rates (80%, 85%, 90%, 95%) across the LOYO folds.

### 5.8 Bracket Prediction

**Module:** `models/bracket.py`

**ELI5:** This is the grand finale. The computer takes everything it has learned, looks at all 68 teams in the 2026 tournament, and plays out every single game from the first round all the way to the championship. It draws a pretty bracket you can look at in your web browser, showing who it thinks will win each game and how confident it is.

**Bracket Construction:**

The bracket module contains the complete 2026 tournament field definition, sourced from ESPN, organized into four regions with all 68 teams and their seeds. The simulation proceeds round by round:

1. **First Four:** Play-in games to determine the final four at-large/automatic qualifier spots.
2. **Round of 64:** All 32 first-round matchups based on the standard bracket seeding (1 vs 16, 2 vs 15, etc.).
3. **Round of 32:** Winners of adjacent R64 games meet.
4. **Sweet Sixteen, Elite Eight, Final Four, Championship:** Successive rounds until a single champion remains.

**Smart Upset Injection:**

Rather than naively always picking the higher-seeded team or always trusting the raw model probabilities, the bracket predictor employs a historically-calibrated upset injection mechanism. Historical tournament data establishes expected upset rates per round:

| Round | Expected Upsets (approx.) |
|---|---|
| Round of 64 | ~9 out of 32 games |
| Round of 32 | ~3 out of 16 games |
| Sweet Sixteen | ~2 out of 8 games |
| Elite Eight | ~1 out of 4 games |
| Final Four | ~0-1 out of 2 games |

The upset injection system adjusts the model's predicted probabilities to produce a bracket that matches these historical upset distributions in aggregate, rather than either over-predicting or under-predicting upsets. It also incorporates specialist model disagreement as an additional upset signal: when the specialist models disagree about a matchup outcome, the uncertainty is higher, and the system is more willing to predict an upset in that game.

**Output:** The bracket predictor generates an interactive HTML file that visualizes the complete predicted bracket with team names, seeds, predicted win probabilities, confidence intervals from the conformal prediction framework, and color-coded indicators for predicted upsets.

---

## 6. Machine Learning Architecture

**ELI5:** Instead of having one computer brain try to learn everything about basketball, we have four smaller computer brains that each specialize in different situations. One brain is a generalist that knows a little about everything. Another brain is an expert on close games. A third brain is an expert on the later rounds when things get really intense. And the fourth brain provides a different perspective using a different learning style. Then there is a fifth brain -- the "boss brain" -- that listens to all four specialist brains and makes the final decision by figuring out which specialist to trust more in each situation.

### 6.1 Specialist Models

The stacking ensemble is composed of four base learners, each deliberately designed to specialize in a different regime of the tournament prediction problem.

**Model A -- XGBoost Generalist:**

- **Algorithm:** XGBoost (Extreme Gradient Boosting), a regularized gradient-boosted decision tree ensemble.
- **Feature Set:** All 112 features surviving the feature selection pipeline.
- **Training Data:** All available training matchups across all years and all rounds.
- **Role:** This is the backbone of the ensemble. It sees everything, learns from everything, and provides a strong baseline prediction for every matchup. Its primary weakness -- which the other models are designed to compensate for -- is that it optimizes a single loss function across all matchups, which may not be optimal for the specific sub-problems of close games or late-round dynamics.

**Model B -- LightGBM Close-Game Specialist:**

- **Algorithm:** LightGBM (Light Gradient Boosting Machine), a histogram-based gradient boosting framework optimized for speed and efficiency.
- **Feature Set:** 25 carefully curated features emphasizing recent form (last3, last1 temporal windows), rebounding metrics, and away performance gaps.
- **Training Data:** Only matchups where `|seed_diff| <= 7`. This filters out blowout-seeding matchups (1 vs 16, 2 vs 15) that are almost always won by the higher seed, and focuses the model's capacity on the competitive matchups where prediction is actually difficult and valuable.
- **Role:** Close games between similarly-seeded teams are decided by different factors than lopsided matchups. Recent form, rebounding (which determines second-chance opportunities), and the ability to perform away from home become disproportionately predictive when the teams are otherwise evenly matched. Model B is the ensemble's specialist for these high-uncertainty matchups.

**Model C -- Logistic Regression Later-Round Specialist:**

- **Algorithm:** Logistic Regression with L2 regularization, providing a linear model perspective that complements the tree-based models.
- **Feature Set:** 23 features centered on free-throw percentage, ball security (turnovers, steals), and defensive intensity metrics.
- **Training Data:** Only matchups from round 3 (Sweet Sixteen) onward.
- **Role:** As the tournament progresses to the later rounds, the surviving teams are almost universally strong, and the differentiating factors shift. Free-throw shooting becomes critical in close late-game situations. Ball security matters more because each possession is more valuable when the opponent is also excellent. Defensive intensity separates the truly elite from the merely very good. Model C captures these later-round dynamics. Additionally, as a linear model, it provides a qualitatively different hypothesis space from the tree-based models A, B, and D, which improves ensemble diversity.

**Model D -- Random Forest Diversity Model:**

- **Algorithm:** Random Forest, an ensemble of independently-trained decision trees with bagging and random feature subsets.
- **Feature Set:** 17 curated features selected for diversity relative to the other models' feature sets.
- **Training Data:** All available training matchups.
- **Role:** Model D exists primarily to increase the diversity of the ensemble. Ensemble learning theory tells us that the predictive power of a stacked ensemble is maximized when the base learners are both individually accurate and mutually uncorrelated in their errors. Random Forest achieves this through its inherent randomization (bagging + random feature subsets at each split), and by operating on a compact and distinct feature set, its error patterns are deliberately decorrelated from the other three models.

**Diversity Achievement:** The specialist design achieves a maximum inter-model Pearson correlation of r = 0.563, dramatically reduced from the r = 0.874 that was observed in the earlier non-specialist ensemble design (`train.py`). Lower inter-model correlation means the meta-learner has more complementary information to work with, which directly translates to better stacked predictions.

### 6.2 Stacking Meta-Learner

**ELI5:** The meta-learner is the boss who listens to all four specialist advisors and decides what to do. It does not just average their opinions -- it learns when to trust each specialist more or less, depending on the situation.

**Architecture:**

The meta-learner is a Logistic Regression model that takes as input:

1. **Base Model Predictions:** The predicted probability from each of the four specialist models (4 features).
2. **Disagreement Signals:** Measures of how much the specialists disagree with each other, including the standard deviation of the four base predictions and the range (max - min) of the predictions. High disagreement indicates high uncertainty and is itself informative.
3. **Context Features:** Round number and seed difference, which allow the meta-learner to learn regime-dependent weighting. For instance, it can learn to weight Model C more heavily in later rounds and Model B more heavily when the seed difference is small.
4. **Interaction Terms:** Specifically, `round_number x seed_diff` interactions, which allow the meta-learner to capture the joint effect of round and competitiveness on specialist reliability.

**Why Logistic Regression for the Meta-Learner:** Using a simple linear model for the meta-learner is a deliberate choice. The base models are already highly expressive nonlinear learners. Stacking another complex nonlinear model on top would risk overfitting to the training data, especially given the relatively small number of training matchups available (approximately 600 tournament games across 10 seasons). Logistic Regression provides just enough flexibility to learn optimal specialist weighting while maintaining strong regularization and generalization.

### 6.3 Conformal Prediction

**ELI5:** When the computer says "Team A has a 75% chance of winning," how do you know you can trust that number? Conformal prediction is a way of checking and adjusting those numbers so that when the computer says 75%, it really means 75%. It also gives you "error bars" -- like saying "I am 90% sure Team A's true win probability is somewhere between 65% and 85%."

**Technical Framework:**

Conformal prediction is a distribution-free uncertainty quantification framework that provides finite-sample validity guarantees under only the assumption of exchangeability. In the context of this pipeline, conformal prediction is applied as follows:

1. **Nonconformity Scores:** For each matchup in the calibration set (the LOYO holdout fold), a nonconformity score is computed as the absolute difference between the model's predicted probability and the observed binary outcome. Larger scores indicate worse predictions.

2. **Round-Conditional Calibration:** Rather than computing a single global calibration, the pipeline stratifies nonconformity scores by tournament round. This accounts for the empirical observation that model performance varies systematically across rounds (e.g., the model is typically better calibrated in early rounds where seed-based features dominate, and less certain in later rounds where the surviving teams are all strong).

3. **Prediction Sets at Multiple Confidence Levels:** At each of the four target confidence levels (80%, 85%, 90%, 95%), the system computes the corresponding quantile of the round-conditional nonconformity score distribution. For a new prediction, the conformal interval is the predicted probability plus or minus this quantile value, clipped to [0, 1].

4. **Calibration Verification:** The empirical coverage of these prediction sets is verified across all LOYO folds, confirming that the actual coverage rates are within 0.2 percentage points of the nominal levels at every tier. This provides strong evidence that the uncertainty estimates are well-calibrated and trustworthy.

### 6.4 Upset Calibration

**ELI5:** Computers that predict basketball often make a boring mistake: they just always pick the better-seeded team. Real March Madness is exciting because of upsets. Upset calibration is how we make the computer's bracket look like a real tournament bracket, with the right number of surprises in each round.

**Methodology:**

The upset calibration module adjusts the raw model probabilities to produce a bracket with a historically-realistic distribution of upsets. This is not about making the model artificially pick random upsets -- it is about correcting a well-known systematic bias in classification models. When a model outputs a 60% probability for the higher-seeded team, picking the higher-seeded team maximizes expected accuracy for that individual game. But when you do this for every single game in a 63-game bracket, you end up with a bracket that has far fewer upsets than any real tournament, which is both unrealistic and suboptimal for bracket pool scoring systems that typically reward correct upset picks with bonus points.

The calibration operates per round, using the following inputs:

- **Historical Base Rates:** The empirical frequency of upsets in each tournament round across all training years.
- **Raw Model Probabilities:** The stacking ensemble's predicted probability for the higher-seeded team in each matchup.
- **Specialist Disagreement:** When the four base models disagree substantially about a matchup, the system treats this as evidence of genuine uncertainty and increases the probability of predicting an upset for that game.

The output is a set of adjusted probabilities that, when used for bracket picks, produces a predicted bracket with approximately the right number of upsets in each round as measured against the historical distribution.

---

## 7. Key Results

**ELI5:** After all that work, here is how well the computer actually does. It gets about 71 out of 100 games right, which is better than just always picking the team with the better seed. And when it tells you how confident it is, those confidence numbers are really trustworthy.

**Cross-Validated Performance (Leave-One-Year-Out):**

| Metric | Value | Context |
|---|---|---|
| CV Accuracy | **71.4%** | Across all LOYO folds (2015-2025 excl. 2020) |
| Higher-Seed Baseline | 70.9% | Always picking the better-seeded team |
| Improvement over Baseline | +0.5 pp | Statistically meaningful given the small margin available |

**Why +0.5 pp Matters More Than It Looks:**

The higher-seed baseline of 70.9% is an extraordinarily strong baseline for tournament prediction. The NCAA selection committee is very good at its job, and seeds capture an enormous amount of information. The remaining ~29% of games that are upsets are inherently high-variance events that are difficult to predict even in principle. Gaining 0.5 percentage points over this baseline represents meaningful signal extraction from the feature engineering and specialist model design, and this improvement is consistent across multiple LOYO folds rather than being driven by a single lucky year.

**Model Diversity:**

| Metric | Value |
|---|---|
| Max Inter-Model Correlation | r = 0.563 |
| Previous (Non-Specialist) Correlation | r = 0.874 |
| Improvement | 35.6% reduction in max correlation |

The dramatic reduction in inter-model correlation confirms that the specialist design is successfully producing diverse base learners whose errors are substantially decorrelated.

**Conformal Prediction Calibration:**

| Nominal Level | Empirical Coverage | Deviation |
|---|---|---|
| 80% | ~80.0% | < 0.2 pp |
| 85% | ~85.1% | < 0.2 pp |
| 90% | ~89.9% | < 0.2 pp |
| 95% | ~95.1% | < 0.2 pp |

The conformal prediction framework achieves empirical coverage within 0.2 percentage points of the nominal level at every confidence tier, confirming that the uncertainty estimates produced by the system are well-calibrated and trustworthy.

---

## 8. Configuration

**Module:** `config.py`

**ELI5:** The configuration file is like the control panel of the whole project. It has all the knobs and dials -- which years to look at, where to find the websites, how long to wait between web requests, and what statistics to collect.

**Key Configuration Parameters:**

```python
# Years included in the pipeline (2020 excluded due to COVID cancellation)
SCRAPE_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

# Tournament date anchors (day before First Four) for each year
# Used to set the ?date= parameter for TeamRankings scraping
TOURNAMENT_DATES = {
    2015: "2015-03-17",
    2016: "2016-03-14",
    # ... one entry per year ...
    2026: "2026-03-16",
}
```

**Stat Slugs:** The configuration file contains the complete list of 45 TeamRankings stat slugs that are scraped. These slugs map directly to the URL paths on TeamRankings.com and include statistics covering offensive efficiency, defensive efficiency, assist-to-turnover ratio, win percentage, scoring margin, rebounding, blocks, steals, free-throw percentage, three-point percentage, and many more.

**Directory Paths:** All intermediate and final data artifact paths (parquet files, HTML output) are defined centrally in the configuration, ensuring that every module in the pipeline reads from and writes to consistent locations.

**Rate Limiting:** HTTP request rate limiting parameters are configured here to ensure respectful and reliable scraping that does not overwhelm the source servers.

---

## 9. Docker Setup

**ELI5:** Docker is like a lunchbox that holds everything the project needs to run -- the right version of Python, all the special libraries, and the code itself -- so that it works exactly the same on anyone's computer without them having to install anything.

**Container Specification:**

The project ships with a `Dockerfile` based on the `python:3.12-slim` image, which provides a minimal Python 3.12 runtime on Debian. The Dockerfile installs all dependencies from `requirements.txt` and copies the project code into the container.

**Docker Compose:**

The `docker-compose.yml` file provides a convenient single-command interface for building and running the entire pipeline:

```bash
# Build the container image and run the full pipeline
docker-compose up --build

# Run in detached mode (background)
docker-compose up --build -d

# View logs
docker-compose logs -f
```

**Volume Mounts:** The `data/` directory is typically mounted as a Docker volume so that output artifacts (parquet files, HTML bracket visualization) persist on the host filesystem after the container exits. This means you can run the pipeline in Docker and then open the generated bracket HTML file directly on your host machine.

**Reproducibility:** By containerizing the pipeline, the Docker setup ensures that the exact same Python version, library versions, and system dependencies are used regardless of the host environment. This eliminates "works on my machine" problems and makes results fully reproducible.

---

## 10. Documentation Links

The following subdirectories contain their own detailed README documentation. Refer to these for deeper dives into each subsystem:

| Subdirectory | Description | Path |
|---|---|---|
| `scrapers/` | Data acquisition modules for all three external sources | [`scrapers/scrapers_readme.md`](scrapers/scrapers_readme.md) |
| `etl/` | Extract-Transform-Load processing pipeline | [`etl/etl_readme.md`](etl/etl_readme.md) |
| `features/` | Feature engineering and feature selection | [`features/features_readme.md`](features/features_readme.md) |
| `models/` | ML model training, evaluation, and bracket prediction | [`models/models_readme.md`](models/models_readme.md) |

---

## 11. Dependencies

**ELI5:** These are the special tools the project borrows from other people who have already built them. It is like how a baker uses an oven that someone else built rather than building their own oven from scratch.

**Core Dependencies (from `requirements.txt`):**

| Package | Version | Purpose |
|---|---|---|
| `polars` | >= 1.0 | High-performance DataFrame library (Rust-backed). Used for all tabular data throughout the pipeline instead of Pandas, providing superior memory efficiency and execution speed for the hundreds of columns generated during feature engineering. |
| `requests` | latest | HTTP client for synchronous web scraping of TeamRankings and Warren Nolan. |
| `beautifulsoup4` | latest | HTML parsing library used to extract tabular statistics from TeamRankings and Warren Nolan web pages. |
| `lxml` | latest | Fast XML/HTML parser backend for BeautifulSoup. Provides significantly faster parsing than the default Python html.parser. |
| `httpx` | latest | Modern async-capable HTTP client used for NCAA API requests. |
| `rapidfuzz` | latest | High-performance fuzzy string matching library (C++ backed). Used in the team name resolution system to match team names across sources with different naming conventions. |
| `scikit-learn` | latest | Machine learning library providing Logistic Regression (Model C and meta-learner), Random Forest (Model D), feature selection utilities, and evaluation metrics. |
| `xgboost` | latest | Gradient boosting library providing the XGBoost algorithm used for Model A (the generalist). |
| `lightgbm` | latest | Gradient boosting library providing the LightGBM algorithm used for Model B (close-game specialist). |
| `numpy` | latest | Numerical computing foundation used throughout for array operations and mathematical transformations. |

**Installation:**

```bash
pip install -r requirements.txt
```

All dependencies are pinned to minimum versions in `requirements.txt` to ensure compatibility while allowing patch-level updates for security fixes.

---

*This pipeline was built to predict the 2026 NCAA Men's Basketball Tournament. The 2026 season is the prediction target and is never included in any training data. Training data spans the 2015 through 2025 tournaments, excluding the cancelled 2020 season.*
