# Features Module: Matchup-Level Feature Engineering and Selection for NCAA Basketball Tournament Prediction

This document provides an exhaustive, deeply detailed walkthrough of the **features** module, which is responsible for transforming raw team-level season statistics and tournament bracket metadata into rich, matchup-level features that a machine learning model can consume to predict the outcome of NCAA Men's Basketball Tournament games. Every section of this document is structured to begin with the simplest possible explanation -- the kind you might give to a five-year-old -- and then progressively ramp up into the rigorous mathematical and statistical detail that a doctoral researcher would expect. The goal is that no matter your background, you will find a level of explanation here that meets you exactly where you are.

---

## Table of Contents

- [1. Module Overview](#1-module-overview)
  - [1.1 What This Module Does (Simple Explanation)](#11-what-this-module-does-simple-explanation)
  - [1.2 Architectural Role Within the Pipeline](#12-architectural-role-within-the-pipeline)
  - [1.3 Data Flow Diagram](#13-data-flow-diagram)
- [2. \_\_init\_\_.py](#2-__init__py)
- [3. Feature Construction Pipeline (build\_features.py)](#3-feature-construction-pipeline-build_featurespy)
  - [3.1 Purpose and High-Level Summary](#31-purpose-and-high-level-summary)
  - [3.2 Input / Output Contract](#32-input--output-contract)
  - [3.3 The Join Strategy](#33-the-join-strategy)
  - [3.4 Round Features](#34-round-features)
    - [3.4.1 \_normalize\_round()](#341-_normalize_round)
    - [3.4.2 round\_group](#342-round_group)
  - [3.5 Seed Features](#35-seed-features)
    - [3.5.1 seed\_diff](#351-seed_diff)
    - [3.5.2 seed\_sum and seed\_product](#352-seed_sum-and-seed_product)
    - [3.5.3 seed\_pct\_diff (Rank #1 Feature)](#353-seed_pct_diff-rank-1-feature)
    - [3.5.4 log\_seed\_ratio (Rank #2 Feature)](#354-log_seed_ratio-rank-2-feature)
    - [3.5.5 seed\_diff\_bucket](#355-seed_diff_bucket)
    - [3.5.6 Why seed\_pct\_diff and log\_seed\_ratio Are the Most Powerful Features](#356-why-seed_pct_diff-and-log_seed_ratio-are-the-most-powerful-features)
  - [3.6 Stat Differentials](#36-stat-differentials)
    - [3.6.1 Raw Differentials (diff\_{stat})](#361-raw-differentials-diff_stat)
    - [3.6.2 Percentage Differentials (pctdiff\_{stat})](#362-percentage-differentials-pctdiff_stat)
  - [3.7 Away Performance Gap Features](#37-away-performance-gap-features)
  - [3.8 Composite Features](#38-composite-features)
    - [3.8.1 efficiency\_gap](#381-efficiency_gap)
    - [3.8.2 ball\_control\_index](#382-ball_control_index)
    - [3.8.3 shooting\_index](#383-shooting_index)
  - [3.9 Conference Strength Features (Placeholder)](#39-conference-strength-features-placeholder)
  - [3.10 Output Summary](#310-output-summary)
- [4. Feature Selection Pipeline (select\_features.py)](#4-feature-selection-pipeline-select_featurespy)
  - [4.1 Purpose and High-Level Summary](#41-purpose-and-high-level-summary)
  - [4.2 The Selection Pipeline at a Glance](#42-the-selection-pipeline-at-a-glance)
  - [4.3 Step 1: Relevance Scoring](#43-step-1-relevance-scoring)
    - [4.3.1 Mutual Information (MI)](#431-mutual-information-mi)
    - [4.3.2 Matthews Correlation Coefficient (MCC)](#432-matthews-correlation-coefficient-mcc)
    - [4.3.3 Combined Ranking](#433-combined-ranking)
  - [4.4 Step 2: Pearson Correlation Deduplication](#44-step-2-pearson-correlation-deduplication)
  - [4.5 Step 3: Bottom 50% Pruning](#45-step-3-bottom-50-pruning)
  - [4.6 Output and Results](#46-output-and-results)
  - [4.7 Key Results: Top-Ranked Features](#47-key-results-top-ranked-features)
- [5. Mathematical Appendix](#5-mathematical-appendix)
  - [5.1 Percentage Difference Derivation](#51-percentage-difference-derivation)
  - [5.2 Why seed\_pct\_diff Outperforms seed\_diff: An Information-Theoretic Deep Dive](#52-why-seed_pct_diff-outperforms-seed_diff-an-information-theoretic-deep-dive)
  - [5.3 Composite Feature Mathematics](#53-composite-feature-mathematics)
  - [5.4 MCC Deep Dive](#54-mcc-deep-dive)
- [6. Quick Reference: Feature Counts](#6-quick-reference-feature-counts)
- [7. Usage](#7-usage)

---

## 1. Module Overview

### 1.1 What This Module Does (Simple Explanation)

Imagine you have two basketball teams about to play each other in the big March Madness tournament. You know all sorts of things about each team -- how many points they score, how many rebounds they grab, how often they steal the ball, and dozens more numbers. But those numbers are about each team *separately*. What you really want to know is: when these two specific teams face off, which one has the advantage, and by how much?

This module takes all of that information for *both* teams and mashes it together into a single row that says "here is how Team A compares to Team B across every dimension we can measure." Then, because there are hundreds of those comparisons and many of them are saying the same thing in slightly different ways, a second step looks at all of them and figures out which ones actually matter for predicting who wins. It throws away the rest so the computer does not get confused by noise.

Think of it like preparing ingredients for a recipe. The first script (`build_features.py`) chops all the vegetables, measures all the spices, and lays everything out on the counter. The second script (`select_features.py`) is the head chef who tastes each ingredient, throws away the ones that add nothing to the dish, and sends only the essential ones to the stove (the model).

### 1.2 Architectural Role Within the Pipeline

The `features` module sits between the **data processing** stage (which cleans and normalizes raw NCAA data into Parquet files) and the **modeling** stage (which trains classifiers on the resulting feature matrix). It is a pure transformation layer: it reads immutable processed data, applies deterministic computations, and writes feature artifacts that downstream consumers (model training, inference, evaluation) treat as their canonical input.

In software-engineering terms, this module implements the **feature store pattern** at a batch level. Every feature is derived from two upstream tables (`tournament_matchups.parquet` and `team_season_stats.parquet`), and the derivation logic is version-controlled, reproducible, and idempotent -- running `build_features.run()` twice on the same inputs produces byte-identical output.

At a more advanced architectural level, the separation of feature construction from feature selection is deliberate: it allows the construction step to be as permissive and exploratory as possible (generate every conceivable feature, even if many are redundant), while the selection step enforces parsimony and decorrelation as a principled dimensionality reduction step. This two-phase approach avoids the common trap of prematurely discarding features during construction that might have been valuable in combination with others.

### 1.3 Data Flow Diagram

```
tournament_matchups.parquet ──┐
                              ├──> build_features.py ──> matchup_features.parquet
team_season_stats.parquet ────┘                                   |
                                                                  |
                                                                  v
                                                       select_features.py
                                                                  |
                                                                  v
                                                    selected_features.json
                                                                  |
                                                                  v
conference_strength.parquet ──> (placeholder, not yet integrated) Model training
```

**Input artifacts:**
- `data/processed/tournament_matchups.parquet` -- one row per tournament game, containing team IDs, seeds, scores, bracket round, season, and a binary `winner` column.
- `data/processed/team_season_stats.parquet` -- one row per team-season, containing approximately 125 statistical columns covering offensive efficiency, defensive efficiency, shooting percentages, rebounding, turnovers, steals, assists, and their `_away` variants.

**Output artifacts:**
- `data/features/matchup_features.parquet` -- the fully joined and feature-engineered matchup table (557+ columns).
- `data/features/selected_features.json` -- a JSON file listing the names of the features that survived selection (approximately 112), along with the original count.

---

## 2. \_\_init\_\_.py

**File path:** `features/__init__.py`

This file is empty. Its sole purpose is to mark the `features/` directory as a Python package so that other modules in the pipeline can write `import features` or `from features import build_features` without encountering `ModuleNotFoundError`. It carries no logic, no re-exports, and no `__all__` declaration.

In practical terms, its presence is a standard Python convention. If you were to delete this file, any `import features.build_features` statement elsewhere in the codebase would fail because Python would no longer recognize the directory as a package namespace.

---

## 3. Feature Construction Pipeline (build\_features.py)

**File path:** `features/build_features.py`

### 3.1 Purpose and High-Level Summary

Think of this script as a factory that takes raw ingredients (game records and team statistics) and assembles them into a rich, model-ready dataset. For every historical tournament matchup, it attaches each team's full season stats, then engineers hundreds of derived features that quantify the *gap* between the two teams along every measurable dimension.

At a more technical level, `build_features.py` performs the following operations in sequence:

1. **Left-joins** team season statistics onto the matchup table twice -- once for `team_a` (the higher-seeded team) and once for `team_b` (the lower-seeded team), prefixing columns with `a_` and `b_` respectively.
2. **Normalizes** the messy `bracket_round` string column into clean integer round numbers (0 through 6) and a coarser `round_group` (0 through 3).
3. **Engineers seed-derived features** including raw difference, sum, product, percentage difference, logarithmic ratio, and a bucketed categorical version.
4. **Computes pairwise stat differentials** for every stat column: both a raw difference (`diff_{stat}`) and a percentage difference (`pctdiff_{stat}`).
5. **Derives away performance gap features** for stats that have both `_season` and `_away` variants, capturing how much worse each team performs on the road.
6. **Constructs composite features** that blend multiple related statistics into single, interpretable indices (efficiency gap, ball control index, shooting index).

The design philosophy is rooted in the principle that raw team-level statistics are uninformative for pairwise classification tasks; what matters is the *contrast* between two teams along each measurable axis. The feature construction step therefore computes both arithmetic differences and percentage differences for every available statistic, yielding a representation that is antisymmetric under relabeling (swapping Team A and Team B flips the sign of every feature, which is exactly what a linear or logistic model needs).

The entry point is `run()`, which is called by the pipeline orchestrator.

### 3.2 Input / Output Contract

| Aspect | Detail |
|---|---|
| **Primary inputs** | `config.PROCESSED_DIR / "tournament_matchups.parquet"`, `config.PROCESSED_DIR / "team_season_stats.parquet"` |
| **Optional input** | `config.PROCESSED_DIR / "conference_strength.parquet"` (currently a stub; not yet integrated) |
| **Output** | `config.FEATURES_DIR / "matchup_features.parquet"` |
| **Idempotency** | Yes -- deterministic given the same inputs |
| **Dependencies** | `polars`, `numpy`, `config` |

### 3.3 The Join Strategy

The join is the very first substantive operation and it is worth understanding carefully, because it determines the shape of every downstream computation.

Starting from the `team_season_stats` table (which contains columns like `team_id`, `team_name`, `season`, and approximately 125 stat columns), the code performs two separate renames-and-joins:

1. **Team A join:** All stat columns are renamed with an `a_` prefix (e.g., `offensive-efficiency_season` becomes `a_offensive-efficiency_season`). The `team_id` column is renamed to `team_a_id` and the join is performed on `["team_a_id", "season"]`. The `team_name` column is dropped to avoid ambiguity.

2. **Team B join:** Identically, but with a `b_` prefix and joining on `["team_b_id", "season"]`.

Both joins are `left` joins, meaning that if a team's stats are missing for a given season, the corresponding columns will be null rather than dropping the matchup row. This is a deliberate design choice: missing data is handled downstream (e.g., the helper function `_safe_diff` returns `None` when either operand is `None`, and the `pctdiff` computation clips denominators to a minimum of 0.001 to avoid division by zero).

After both joins, each matchup row contains the full statistical profile of both teams, side by side, ready for differential computation.

### 3.4 Round Features

#### 3.4.1 \_normalize\_round()

At the simplest level, this function takes the messy round name that comes from the NCAA API (things like `"SWEET 16&#174;"` or `"First Round"` or just `"3"`) and turns it into a clean number from 0 to 6.

**The mapping is:**

| Round Name | Integer | Description |
|---|:---:|---|
| First Four | 0 | Play-in games |
| Round of 64 (R64) | 1 | First full round |
| Round of 32 (R32) | 2 | Second full round |
| Sweet 16 | 3 | Regional semifinal |
| Elite 8 | 4 | Regional final |
| Final Four | 5 | National semifinal |
| Championship | 6 | National championship |

**Messy input handling:** The function is deliberately defensive. It performs the following normalization steps before matching:

1. Converts the input to lowercase.
2. Strips the HTML entity `&#174;` (the registered trademark symbol that the NCAA API sometimes appends to round names like "Sweet 16").
3. Strips the literal Unicode character `(r)` (the same symbol, but sometimes delivered as the actual character rather than the HTML entity).
4. Strips leading and trailing whitespace.

If none of the string-matching rules fire, the function attempts a numeric fallback by casting the cleaned string to an integer, which handles cases where the API returns bare numeric round identifiers (`"1"`, `"2"`, etc.). If even that fails, the function returns `-1` to signal an unrecognized round.

**Historical naming ambiguity:** The function includes a comment about the NCAA's confusing nomenclature changes. Prior to 2011, what is now called the "First Four" was called the "First Round," and the current "Round of 64" was called the "Second Round." Since the pipeline operates on data from 2015 onward, the post-2011 naming convention applies, but the code documents this subtlety for maintainability.

#### 3.4.2 round\_group

After computing `round_number`, the code further coarsens it into a `round_group` integer with four levels:

| round_group | Meaning | round_number values |
|:---:|---|---|
| 0 | First Four | 0 and below |
| 1 | Early rounds | 1, 2 (R64 + R32) |
| 2 | Mid rounds | 3, 4 (Sweet 16 + Elite 8) |
| 3 | Late rounds | 5, 6 (Final Four + Championship) |

The purpose of this grouping is to provide a pooling mechanism for models or analyses that need larger sample sizes per group. Later tournament rounds have very few games per season (only 2 Final Four games and 1 Championship game), so pooling them together yields more statistically stable estimates. The round group feature is particularly useful for stratified analyses and for models that want to learn distinct behavioral regimes at different stages of the tournament without overfitting to the extreme sparsity of late-round data.

### 3.5 Seed Features

Seeding is the single most powerful predictor of tournament outcomes. The module engineers six distinct seed-derived features, each capturing a different aspect of the seed relationship between the two teams. This section describes each feature individually, then devotes an extended subsection to explaining why the normalized variants (`seed_pct_diff` and `log_seed_ratio`) dominate the feature importance rankings.

**Convention:** In the matchup table, `team_a` is always the higher-seeded team (lower seed number, e.g., a 1-seed) and `team_b` is the lower-seeded team (higher seed number, e.g., a 16-seed). This means `seed_diff` is always zero or negative.

#### 3.5.1 seed\_diff

```
seed_diff = team_a_seed - team_b_seed
```

The simplest possible seed feature. A 1-seed playing a 16-seed yields `seed_diff = 1 - 16 = -15`. This feature is highly predictive in early rounds where seed gaps are large, but its predictive power degrades substantially in later rounds where both teams tend to be highly seeded (e.g., a 1-seed vs. a 2-seed gives `seed_diff = -1`, the same value as a 7-seed vs. an 8-seed in the first round, despite the matchups being qualitatively very different in terms of overall team quality).

#### 3.5.2 seed\_sum and seed\_product

```
seed_sum = team_a_seed + team_b_seed
seed_product = team_a_seed * team_b_seed
```

These features encode the *absolute quality level* of the matchup, as opposed to the *relative gap* captured by `seed_diff`. A `seed_sum` of 3 (1-seed vs. 2-seed) indicates an elite matchup; a `seed_sum` of 24 (8-seed vs. 16-seed) indicates a much lower-quality one. The product amplifies this signal nonlinearly: matchups involving a very low seed and a very high seed produce disproportionately large products (1 * 16 = 16 vs. 8 * 9 = 72), which helps distinguish between matchups that have the same seed_sum but very different competitive dynamics.

#### 3.5.3 seed\_pct\_diff (Rank #1 Feature)

```
seed_pct_diff = (team_a_seed - team_b_seed) / max((team_a_seed + team_b_seed) / 2, 0.001)
```

This is the **single most predictive feature in the entire feature set**, consistently ranking #1 in the feature selection pipeline across multiple seasons of data. Rather than computing the raw gap between seeds, this feature computes the gap *relative to the average seed level of the matchup*. The denominator is clipped to a minimum of 0.001 to prevent division by zero in the degenerate case where both seeds are zero (which should never occur in practice but is handled defensively).

A detailed explanation of why this feature is so powerful is provided in [Section 3.5.6](#356-why-seed_pct_diff-and-log_seed_ratio-are-the-most-powerful-features).

#### 3.5.4 log\_seed\_ratio (Rank #2 Feature)

```
log_seed_ratio = ln(team_a_seed) - ln(team_b_seed) = ln(team_a_seed / team_b_seed)
```

This is the **second most predictive feature in the entire feature set**. Seeds are cast to `Float64` before the logarithm is applied to avoid integer-domain errors. The logarithmic transformation has several desirable mathematical properties:

1. **Diminishing returns:** The difference between a 1-seed and a 2-seed (log ratio = -0.693) is treated as much more significant than the difference between a 14-seed and a 15-seed (log ratio = -0.069), which aligns with empirical win-rate data.
2. **Multiplicative invariance:** The feature captures proportional rather than absolute differences.
3. **Antisymmetry under swap:** Swapping the teams negates the log ratio, which is a clean mathematical property for a model to learn from.

A detailed explanation of why this feature is so powerful is provided in [Section 3.5.6](#356-why-seed_pct_diff-and-log_seed_ratio-are-the-most-powerful-features).

#### 3.5.5 seed\_diff\_bucket

```
seed_diff_bucket =
    0  if |seed_diff| <= 3   (close matchup)
    1  if 4 <= |seed_diff| <= 7  (moderate mismatch)
    2  if |seed_diff| >= 8   (heavy favorite)
```

This is a coarse categorical encoding of the seed gap. It allows models (especially tree-based ones) to easily learn distinct behavioral regimes: close matchups behave differently from blowout-seeding mismatches, and a three-level bucket captures that without overfitting to individual seed pairings. The thresholds (3, 7) were chosen to roughly correspond to the natural breakpoints in historical upset rates.

#### 3.5.6 Why seed\_pct\_diff and log\_seed\_ratio Are the Most Powerful Features

This subsection provides an extended, multi-level explanation of why these two features consistently dominate every feature importance analysis, ranking #1 and #2 respectively across all selection methods (mutual information, MCC, and combined rankings).

**ELI5 explanation:** Imagine you are comparing two runners in a race. Runner A is the fastest in the world, and Runner B is the 4th fastest. The gap between them (3 places) is *huge* -- Runner A is expected to win easily. Now imagine Runner C is the 10th fastest and Runner D is the 13th fastest. The gap is still 3 places, but this time it barely matters -- they are both "pretty good but not great" runners, and either could win. The simple subtraction (3 places) cannot tell these situations apart. But if you say "the gap is 3 out of an average ranking of 2.5" for the first case and "the gap is 3 out of an average ranking of 11.5" for the second case, suddenly the numbers correctly reflect that the first gap is much more meaningful. That is exactly what `seed_pct_diff` does.

**Intermediate explanation: the variance collapse problem.** The raw `seed_diff` feature works beautifully in the Round of 64, where matchups span the full range of seed pairings: 1-vs-16, 2-vs-15, 3-vs-14, all the way to 8-vs-9. The `seed_diff` values range from -15 to -1, providing a wide, information-rich signal. But as the tournament progresses and lower-seeded teams are eliminated, the surviving teams converge toward the top of the seed rankings. By the Elite Eight, typical matchups are 1-vs-2, 1-vs-3, 2-vs-3, or 3-vs-4 -- and `seed_diff` values are compressed into a narrow band of roughly [-3, 0]. This is **variance collapse**: the feature has lost nearly all of its discriminative range precisely in the rounds where accurate prediction is most valuable (and most difficult).

Consider the following concrete examples showing how `seed_diff` fails to distinguish meaningfully different matchups while the normalized variants preserve the signal:

| Matchup | Round | seed_diff | seed_pct_diff | log_seed_ratio |
|---|---|:---:|:---:|:---:|
| 1 vs 16 | R64 | -15 | -1.882 | -2.773 |
| 1 vs 4 | Sweet 16 | -3 | -1.200 | -1.386 |
| 4 vs 7 | Sweet 16 | -3 | -0.545 | -0.560 |
| 1 vs 2 | Elite 8 | -1 | -0.667 | -0.693 |
| 3 vs 4 | Elite 8 | -1 | -0.286 | -0.288 |
| 7 vs 8 | R64 | -1 | -0.133 | -0.134 |
| 1 vs 3 | Final Four | -2 | -1.000 | -1.099 |
| 2 vs 3 | Final Four | -1 | -0.400 | -0.405 |

Notice how `seed_diff = -1` occurs in three very different contexts (1-vs-2 in the Elite Eight, 3-vs-4 in the Elite Eight, and 7-vs-8 in the first round), but `seed_pct_diff` and `log_seed_ratio` correctly assign very different magnitudes to each. The 1-vs-2 matchup gets `seed_pct_diff = -0.667`, reflecting that the gap represents a 67% difference relative to the matchup level, while the 7-vs-8 matchup gets `seed_pct_diff = -0.133`, reflecting the much smaller relative significance of the same absolute gap.

**Advanced explanation: the latent strength model.** To understand *why* these transformations work, we need a model of the relationship between seeds and true team strength. Seeds are not linearly spaced in terms of competitive ability. Empirical data overwhelmingly supports a **concave, approximately logarithmic or power-law** relationship between seed number and true latent strength:

- The gap between the #1 and #2 teams in a region is enormous (the #1 team earned the top overall seed by dominating its conference and having the best resume in the country).
- The gap between the #8 and #9 teams is negligible (they are often interchangeable mid-major conference champions or bubble teams).
- The gap between the #15 and #16 teams is essentially zero.

If we model true latent team strength as a decreasing convex function of seed, such as:

```
strength(s) = s^(-gamma)    for some gamma > 0
```

or equivalently:

```
ln(strength(s)) = -gamma * ln(s)
```

then the log-odds of Team A winning a matchup scale (under a Bradley-Terry or logistic model) as:

```
logit(P(A wins)) ~ beta * [ln(strength(s_a)) - ln(strength(s_b))]
                 = beta * gamma * [ln(s_b) - ln(s_a)]
                 = -beta * gamma * log_seed_ratio
```

This reveals that **`log_seed_ratio` is a sufficient statistic** for the seed-based component of win probability under a power-law strength model. It is, in a very precise sense, the *correct* transformation of seeds for predicting outcomes.

The `seed_pct_diff` feature approximates this behavior through a different mathematical pathway. Using the first-order Taylor expansion of the logarithm around equal seeds (`s_a ~ s_b ~ s_bar`):

```
ln(s_a / s_b) = ln(1 + (s_a - s_b) / s_b) ~ (s_a - s_b) / s_b ~ seed_pct_diff
```

This approximation is tightest when the seeds are similar (i.e., in late rounds), which is precisely the regime where the raw `seed_diff` fails. The two features are thus closely related theoretically but not identical, which is why they both survive the Pearson correlation deduplication step (their correlation is high but typically below the 0.90 threshold) and why keeping both is valuable.

**Information-theoretic formalization:** The superiority of these features can be quantified through conditional entropy. Define `W` as the binary winner variable, `D = seed_diff`, `P = seed_pct_diff`, and `R = round_number`. The claim is:

```
H(W | P, R >= 3) < H(W | D, R >= 3)
```

That is, the conditional entropy of the winner given `seed_pct_diff` (conditioned on being in a late round) is lower than the conditional entropy given `seed_diff`. This means `seed_pct_diff` retains strictly more predictive information in late rounds. This is confirmed empirically by the mutual information scores computed in the feature selection pipeline, where `seed_pct_diff` consistently achieves the highest MI with the target variable.

### 3.6 Stat Differentials

This is where the bulk of the 557+ features come from. For each of the approximately 125 statistical columns in the team season stats table, the code computes two derived features per stat.

#### 3.6.1 Raw Differentials (diff\_{stat})

```
diff_{stat} = a_{stat} - b_{stat}
```

**ELI5:** If Team A scores 80 points per game and Team B scores 70, the difference is 10. We compute this simple subtraction for every single number we know about each team.

**Intermediate:** This is the absolute gap in a given statistic between the two teams. For example, `diff_offensive-efficiency_season` captures how many more points per 100 possessions Team A generates relative to Team B. The magnitude and sign directly encode which team is stronger along that dimension and by how much.

**Advanced:** The raw differential preserves the original units of measurement, which can be important for interpretability and for models that can exploit the absolute magnitude of a gap (e.g., the difference between a 2-point efficiency gap and a 20-point efficiency gap is meaningful in terms of expected point differential). However, it does not account for the baseline level of the statistic, which is why the percentage differential is computed alongside it. The downstream feature selection pipeline is responsible for choosing between the `diff_` and `pctdiff_` variants based on empirical predictive power.

#### 3.6.2 Percentage Differentials (pctdiff\_{stat})

```
pctdiff_{stat} = (a_{stat} - b_{stat}) / max((a_{stat} + b_{stat}) / 2, 0.001)
```

**ELI5:** Instead of just the difference, we ask "how big is the difference compared to the average?" A difference of 2 points matters a lot more when both teams average 5 points than when both teams average 100 points.

**Intermediate:** This is the symmetric percentage difference (the same formula used for `seed_pct_diff`, applied to every stat). It normalizes the gap by the average value, making features comparable across statistics that operate on very different scales. For instance, a raw difference of 2.0 in offensive efficiency (which typically ranges from 90 to 120) means something very different from a raw difference of 2.0 in three-point percentage (which typically ranges from 0.28 to 0.40). The percentage difference places both on a common, dimensionless scale.

**Advanced:** The denominator is clipped to a minimum of 0.001 to prevent division-by-zero errors for statistics that can sum to zero. The percentage difference is formally a symmetric relative percent difference (RPD), which is scale-invariant: if both teams' stats are multiplied by a constant, the percentage difference is unchanged. This property makes it particularly valuable for models that are sensitive to feature magnitudes (logistic regression, SVMs) without requiring a separate standardization step.

**Together, the raw and percentage differentials for ~125 stats produce approximately 250 features.** The intentional redundancy between `diff_` and `pctdiff_` variants is resolved downstream by the Pearson correlation deduplication in the feature selection pipeline.

### 3.7 Away Performance Gap Features

This is one of the more nuanced feature engineering ideas in the module, and it is rooted in a key insight about NCAA tournament basketball: **all tournament games are played at neutral sites**. A team that performs significantly worse on the road than at home may be disproportionately disadvantaged in a neutral-site setting, even if its overall season stats look strong.

**ELI5:** Some basketball teams play great in their own gym but fall apart when they have to play somewhere else. Since every tournament game is in a gym that belongs to neither team, we want to know: how much worse does each team play when they are away from home? If Team A stays steady and Team B gets a lot worse, that is a big advantage for Team A in the tournament.

**Intermediate:** For every stat that has both a `_season` variant (full season average) and an `_away` variant (away-game average), the code computes three derived features:

1. **Per-team away gap:**
   ```
   a_awaygap_{base} = a_{base}_season - a_{base}_away
   b_awaygap_{base} = b_{base}_season - b_{base}_away
   ```
   A positive value means the team performs *better* overall than on the road (i.e., they have a significant home-court advantage that inflates their season stats). A large positive away gap is a red flag for tournament performance.

2. **Differential of away gaps:**
   ```
   diff_awaygap_{base} = a_awaygap_{base} - b_awaygap_{base}
   ```
   This captures which team's stats are more "inflated" by home games. If Team A has a much larger away gap than Team B, Team A's season stats are less reflective of their true neutral-site ability.

**Advanced:** The underlying hypothesis being tested by these features is that home-court advantage is heterogeneous across teams, and that teams with smaller home-away performance gaps are better adapted to the neutral-court environment of the NCAA tournament. This is a well-documented phenomenon in sports analytics literature: teams that rely heavily on crowd energy, familiarity with their home floor, or favorable referee dynamics tend to underperform their regular-season metrics in tournament play. By explicitly modeling the away-performance gap and computing the differential between the two teams in a matchup, we introduce a latent variable proxy for "tournament readiness" that is not captured by raw season averages alone.

The away gap features add approximately 3 features per stat that has both `_season` and `_away` variants (two per-team gaps plus one differential), contributing meaningfully to the total feature count.

### 3.8 Composite Features

These are hand-crafted features that combine multiple related statistics into single, interpretable indices. They are designed to capture higher-order basketball concepts that no single stat adequately represents. Each composite feature is a weighted linear combination of component stat differentials, with weights chosen based on domain knowledge about the relative importance of each component and the degree of information overlap between them.

#### 3.8.1 efficiency\_gap

**ELI5:** For each team, we figure out how much better their offense is than their defense. Then we compare: whose gap is bigger? The team that is a lot better on offense than defense, compared to their opponent, has the advantage.

**Intermediate:**

```
efficiency_gap = (OE_a - DE_a) - (OE_b - DE_b)
```

where `OE` is offensive efficiency (points scored per 100 possessions) and `DE` is defensive efficiency (points allowed per 100 possessions). Net efficiency (OE - DE) is the single best summary statistic for overall team quality in tempo-free basketball analytics (the metric that KenPom, BartTorvik, and other widely-respected ranking systems use as their core signal).

**Advanced / Mathematical detail:** Formally, let `OE_a`, `DE_a`, `OE_b`, `DE_b` denote the offensive and defensive efficiency ratings for teams A and B respectively. Define each team's *net efficiency margin* (NEM) as:

```
NEM_a = OE_a - DE_a
NEM_b = OE_b - DE_b
```

The efficiency gap is:

```
efficiency_gap = NEM_a - NEM_b = (OE_a - DE_a) - (OE_b - DE_b)
```

This can be algebraically decomposed as:

```
efficiency_gap = (OE_a - OE_b) - (DE_a - DE_b)
              = diff_offensive_efficiency - diff_defensive_efficiency
```

This decomposition reveals that the efficiency gap is positive when Team A's offensive advantage exceeds its defensive disadvantage (or when Team A is superior on both sides of the ball). Under a simplified points-per-possession model where the expected score in a game is `Score_A ~ OE_a * Possessions / 100` and `Score_B ~ DE_a * Possessions / 100` (ignoring the interaction between offensive and defensive efficiencies), the expected margin of victory for Team A is approximately:

```
E[Score_A - Score_B] ~ (NEM_a - NEM_b) * Possessions / 200
                      = efficiency_gap * Possessions / 200
```

This makes `efficiency_gap` a first-order approximation to the expected point differential, normalized out of the tempo of play.

The source code references the columns `a_offensive-efficiency_season`, `a_defensive-efficiency_season`, and their `b_` counterparts. A guard clause (`if all(c in matchups.columns for c in [oe_a, de_a, oe_b, de_b])`) ensures the feature is only computed if all four columns are present in the joined data, providing graceful degradation if the upstream stats table is incomplete.

#### 3.8.2 ball\_control\_index

**ELI5:** This number tells you which team is better at taking care of the basketball. A team that passes well (lots of assists), steals the ball a lot, and does not fumble the ball away (few turnovers) has a high ball control score. We compare the two teams' ball control to see who has the edge.

**Intermediate:**

```
ball_control_index = (ATO_a - ATO_b) + 0.5 * (STL_a - STL_b) - 0.3 * (TOV_a - TOV_b)
```

where:
- `ATO` = assist-to-turnover ratio
- `STL` = steals per game
- `TOV` = turnovers per game

**Advanced / Mathematical detail:** Let us define the ball control index (BCI) rigorously as a weighted linear combination of three differential components:

```
BCI = 1.0 * Delta_ATO + alpha * Delta_STL - beta * Delta_TOV
```

where:
- `Delta_ATO = ATO_a - ATO_b` (assist-to-turnover ratio differential; weight = 1.0)
- `Delta_STL = STL_a - STL_b` (steals differential; weight alpha = 0.5)
- `Delta_TOV = TOV_a - TOV_b` (turnovers differential; weight beta = 0.3)

**Justification of weights:**

The assist-to-turnover ratio receives implicit weight 1.0 because it is the most holistic single measure of ball handling quality. It captures the fundamental tradeoff between creating scoring opportunities (assists) and surrendering possession (turnovers) in a single, self-normalizing ratio. Teams with high ATO ratios are disciplined offensive teams that take care of the ball while generating open shots for teammates.

Steals receive a half-weight (`alpha = 0.5`) because they represent the *defensive* side of ball control -- the ability to force the opponent into errors. However, steals are noisier than ATO ratio (a team that gambles aggressively for steals may also give up defensive positioning, leading to easy baskets for the opponent), and they are partially redundant with ATO ratio (a team that forces many turnovers will indirectly inflate its opponent's ATO ratio). The 0.5 weight balances the additional signal against the noise and redundancy.

Turnovers receive a negative weight (`-beta = -0.3`) because a positive `Delta_TOV` (Team A turns the ball over more than Team B) is a *disadvantage* for Team A, so subtracting it aligns the sign convention with the rest of the index. The weight is only 0.3 (lower than the 0.5 for steals) because turnovers are already substantially captured in the ATO differential -- the ATO ratio is defined as `assists / turnovers`, so any change in turnovers directly affects ATO. Including raw turnovers at full weight would therefore double-count the turnover signal.

Expanding the formula fully:

```
BCI = (ATO_a - ATO_b) + 0.5 * (STL_a - STL_b) - 0.3 * (TOV_a - TOV_b)
    = ATO_a - ATO_b + 0.5 * STL_a - 0.5 * STL_b - 0.3 * TOV_a + 0.3 * TOV_b
```

The feature degrades gracefully: if steals or turnovers columns are missing from the input data, it still computes using only the available components, thanks to conditional inclusion in the code.

#### 3.8.3 shooting\_index

**ELI5:** This number tells you which team is better at making baskets. It combines how well they shoot overall (giving extra credit for three-pointers), how well they shoot from far away, and how often they get to shoot free throws (the easiest shots in basketball because nobody is guarding you).

**Intermediate:**

```
shooting_index = (eFG_a - eFG_b) + 0.5 * (3P_a - 3P_b) + 0.3 * (FTR_a - FTR_b)
```

where:
- `eFG` = effective field goal percentage: `(FG + 0.5 * 3PM) / FGA`, which gives three-pointers 1.5x credit
- `3P` = three-point field goal percentage
- `FTR` = free throw rate (free throw attempts per field goal attempt)

**Advanced / Mathematical detail:** The shooting index (SI) is formally defined as:

```
SI = 1.0 * Delta_eFG + gamma * Delta_3P + delta * Delta_FTR
```

where:
- `Delta_eFG = eFG%_a - eFG%_b` (effective field goal percentage differential; weight = 1.0)
- `Delta_3P = 3P%_a - 3P%_b` (three-point percentage differential; weight gamma = 0.5)
- `Delta_FTR = FTR_a - FTR_b` (free throw rate differential; weight delta = 0.3)

**Justification of weights:**

Effective field goal percentage receives full weight because it is the most comprehensive single measure of shooting efficiency in basketball. Its formula, `eFG% = (FGM + 0.5 * 3PM) / FGA`, already adjusts for the added value of three-point shots by counting each made three as 1.5 made field goals. This means eFG% naturally captures the value of both two-point and three-point shooting in a single metric.

Three-point percentage receives a half-weight (`gamma = 0.5`) because it provides *additional* signal beyond what eFG% captures. Specifically, two teams can have the same eFG% through very different shooting profiles: one relying on high-percentage two-point shots (layups, dunks) and the other relying on high-volume three-point shooting. In March Madness, three-point shooting ability is particularly important because (a) it enables comebacks from large deficits, (b) it can compensate for size and athleticism mismatches against more talented opponents, and (c) it is the primary mechanism by which mid-major upsets occur (a hot shooting night from beyond the arc). The 0.5 weight adds this extra signal without double-counting the three-point information already embedded in eFG%.

Free throw rate receives a 0.3 weight (`delta = 0.3`) because the ability to draw fouls and earn free throw attempts is a complementary axis of offensive efficiency that is largely independent of field goal shooting accuracy. FTR captures the team's aggressiveness in attacking the basket, drawing contact, and earning "free" points that do not require a made field goal. This is particularly valuable in the tournament context, where referees may call games tighter or looser than the regular season, and teams that attack the rim generate more robust scoring opportunities.

The complete expansion:

```
SI = (eFG%_a - eFG%_b) + 0.5 * (3P%_a - 3P%_b) + 0.3 * (FTR_a - FTR_b)
   = eFG%_a - eFG%_b + 0.5 * 3P%_a - 0.5 * 3P%_b + 0.3 * FTR_a - 0.3 * FTR_b
```

Like the ball control index, the shooting index is constructed so that positive values favor Team A, and it degrades gracefully when component columns are missing.

### 3.9 Conference Strength Features (Placeholder)

The code contains a placeholder for conference strength features. If the file `conference_strength.parquet` exists in the processed data directory, a log message is printed ("Adding conference features..."), but no features are currently derived from it. The comment in the code indicates that a team-to-conference mapping is needed before this feature group can be implemented.

This is a known extension point for future development. When implemented, it would likely include features such as:
- Conference average net efficiency
- Number of tournament teams from each conference (a proxy for conference depth)
- Conference-adjusted strength-of-schedule differential between Team A and Team B
- Historical conference tournament performance metrics

### 3.10 Output Summary

After all features are computed, the final DataFrame is written to `config.FEATURES_DIR / "matchup_features.parquet"`. The function prints a comprehensive summary including:

- Total number of matchup rows
- Total number of columns (typically 557+)
- Count of engineered features (identified by prefix patterns: `diff_`, `pctdiff_`, `a_awaygap_`, `b_awaygap_`, `diff_awaygap_`, `seed_`, `efficiency_`, `ball_control_`, `shooting_`)
- A sample of the first 10 column names for quick visual verification
- A sanity-check Pearson correlation between `seed_diff` and `winner`, which should be meaningfully negative (higher seeds -- lower seed numbers -- tend to win)

---

## 4. Feature Selection Pipeline (select\_features.py)

**File path:** `features/select_features.py`

### 4.1 Purpose and High-Level Summary

**ELI5:** After we build all those features (sometimes over 500 of them!), many of them are saying the same thing in slightly different ways, and some of them are completely useless. If we give the computer too many features, it gets confused and starts memorizing the training data instead of learning real patterns (this is called "overfitting"). So we use three clever tricks to figure out which features are the most useful and throw away the rest.

**Intermediate:** The feature selection pipeline implements a three-stage process: (1) score every feature's relevance to the binary target (`winner`) using two complementary methods -- mutual information and Matthews correlation coefficient; (2) deduplicate highly correlated features using the Pearson correlation matrix; (3) prune the bottom half of remaining features by combined relevance rank. The result is a lean, decorrelated feature set that maximizes predictive power per feature.

**Advanced:** The pipeline addresses the dual curse of (a) the bias-variance tradeoff (too many features increases variance and overfitting risk, especially given the relatively small sample size of ~600 tournament games over a decade of data) and (b) multicollinearity (many of the `diff_` and `pctdiff_` variants of the same base statistic are near-perfectly correlated, which inflates coefficient standard errors in linear models and creates splits on essentially identical information in tree models). The three-stage approach is deliberately heterogeneous in its scoring methods to avoid the blind spots of any single method: MI captures nonlinear relationships, MCC captures optimal binary splits, and Pearson correlation captures linear redundancy.

### 4.2 The Selection Pipeline at a Glance

```
557 input features
        |
        v
  +---------------------------+
  |  Step 1: Compute MI &     |
  |  MCC relevance scores     |
  |  for each feature         |
  +-----------+---------------+
              |
              v
  +---------------------------+
  |  Step 2: Rank features    |
  |  by combined MI + MCC     |
  |  average rank             |
  +-----------+---------------+
              |
              v
  +---------------------------+
  |  Step 3: Pearson dedup    |
  |  Drop one of any pair     |
  |  with |r| > 0.90          |
  |  (keep higher-ranked)     |
  +-----------+---------------+
              |
              v
  +---------------------------+
  |  Step 4: Drop bottom      |
  |  50% by combined rank     |
  +-----------+---------------+
              |
              v
     ~112 selected features
```

### 4.3 Step 1: Relevance Scoring

The first stage computes two independent measures of how well each feature predicts the binary `winner` target variable. Using two complementary methods rather than one reduces the risk of missing features that are predictive through mechanisms that one method is blind to (e.g., MI can miss features whose signal is concentrated at a single threshold, while MCC can miss features with smooth, continuous relationships).

#### 4.3.1 Mutual Information (MI)

**ELI5:** Mutual information asks: "If I tell you the value of this feature, how much does that help you guess whether Team A or Team B won?" Features that help you guess a lot get a high score. Features that tell you nothing get a score of zero.

**Intermediate:** MI is computed using `sklearn.feature_selection.mutual_info_classif` with the following parameters:

- `discrete_features=False` -- all features are treated as continuous.
- `random_state=42` -- for reproducibility.
- `n_neighbors=5` -- the number of nearest neighbors used in the Kraskov-Stoegbauer-Grassberger (KSG) entropy estimator.

MI is a non-parametric measure that captures *any* statistical dependency between a feature and the target, including nonlinear relationships that Pearson correlation would miss entirely.

**Advanced:** Formally, mutual information is defined as:

```
I(X; Y) = H(Y) - H(Y | X)
        = sum_{y in {0,1}} integral p(x, y) * ln(p(x, y) / (p(x) * p(y))) dx
```

where `H(Y)` is the entropy of the target, `H(Y | X)` is the conditional entropy of the target given the feature, and `p(x, y)` is the joint density. Because the target is binary and the features are continuous, `mutual_info_classif` uses a hybrid estimator that treats `Y` as discrete and `X` as continuous, applying the KSG estimator within each class.

The KSG estimator approximates MI as:

```
I(X; Y) = psi(k) - <psi(n_x + 1) + psi(n_y + 1)> + psi(N)
```

where `psi` is the digamma function, `k` is the number of neighbors, `n_x` and `n_y` are the neighbor counts in the marginal spaces, and `N` is the total sample count. This nonparametric estimator has the key property that it is *consistent* (converges to the true MI as `N -> infinity`) and can detect arbitrary nonlinear dependencies, making it a powerful complement to the threshold-based MCC approach.

NaN values in the feature matrix are replaced with 0.0 before computation to ensure numerical stability.

#### 4.3.2 Matthews Correlation Coefficient (MCC)

**ELI5:** MCC works like this: we draw a line through the feature values (like "above average" vs. "below average") and then check whether the teams above the line tend to win and the teams below tend to lose. We try a few different lines and pick the one that gives the best match between "above the line" and "winner."

**Intermediate:** For each feature, the MCC scoring procedure discretizes the continuous feature into a binary indicator using multiple quantile thresholds (by default, the 25th, 50th, and 75th percentiles), computes the Matthews correlation coefficient between each binarized feature and the binary target, and retains the maximum absolute MCC across all thresholds. This approach converts MCC -- which is normally defined for binary-binary comparisons -- into a measure of the best achievable binary split of a continuous feature.

**Advanced:** The Matthews Correlation Coefficient is defined for two binary variables as:

```
MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
```

where TP, TN, FP, FN are the entries of the 2x2 confusion matrix. MCC ranges from -1 (perfect inverse prediction) to +1 (perfect agreement), with 0 indicating no better than random chance. It is widely regarded as one of the most balanced measures for binary classification because it uses all four quadrants of the confusion matrix and is robust to class imbalance. A classifier that always predicts the majority class achieves MCC = 0, not a misleadingly high value as accuracy would give.

**The discretization procedure** works as follows for each feature:

1. Compute quantile boundaries using `np.linspace(0, 1, n_bins + 1)[1:-1]` where `n_bins=4`, yielding quantiles at `[0.25, 0.5, 0.75]`.
2. For each quantile boundary `q`, compute `threshold = np.quantile(feature_values, q)`.
3. Binarize: `feature_binary = (feature_values > threshold).astype(int)`.
4. Compute `|MCC|` between `feature_binary` and `winner`. The absolute value is taken because we care about the *strength* of the relationship regardless of direction.
5. Retain the maximum `|MCC|` across all threshold boundaries.

This is conceptually similar to the first split of a decision tree evaluated using MCC rather than Gini impurity or information gain. The advantage is that it provides a standardized, interpretable score that can be directly compared across features and combined with MI scores. The use of multiple quantile thresholds (rather than just the median) allows the procedure to detect features whose predictive signal is concentrated at non-median points in the distribution -- for example, a feature that is only discriminative in its extreme tails.

#### 4.3.3 Combined Ranking

After computing MI and MCC scores for all features, the code produces a combined relevance ranking:

1. **Rank MI scores** in descending order: `mi_ranks = argsort(argsort(-mi_scores))`. The double `argsort` converts scores to dense ranks (0 = best, N-1 = worst).
2. **Rank MCC scores** identically: `mcc_ranks = argsort(argsort(-mcc_scores))`.
3. **Average the ranks:** `combined_ranks = (mi_ranks + mcc_ranks) / 2.0`.

This rank-averaging approach is a form of *rank aggregation* from social choice theory (a simplified Borda count). Rather than combining raw scores (which would require choosing arbitrary weights and dealing with different scales and distributional shapes), rank averaging treats both metrics as equally important ordinal signals. A feature that ranks #1 in MI but #100 in MCC gets a combined rank of 50.5, which is a reasonable compromise.

The top 20 features by combined rank are printed for diagnostic purposes, showing each feature's MI score, MCC score, and combined rank.

### 4.4 Step 2: Pearson Correlation Deduplication

**ELI5:** If two features are basically saying the same thing (like "how many points Team A scores more" and "what percentage more points Team A scores"), we only need to keep one of them. We look at every pair of features, and if they are more than 90% similar, we throw away the less useful one.

**Intermediate:** The Pearson correlation matrix is computed via `np.corrcoef` on the NaN-imputed feature matrix (NaN values replaced with 0.0). For every pair of features `(i, j)` where `|r_ij| > 0.90` (the default threshold):

- The feature with the *higher* (worse) combined relevance rank is dropped.
- If feature `i` is dropped, the inner loop breaks and proceeds to the next `i`, since `i` is no longer a candidate for further comparisons.

**Advanced:** The deduplication algorithm implements a greedy heuristic on the correlation graph. Define an undirected graph `G = (V, E)` where `V` is the set of features and `(i, j) in E` iff `|r_ij| > tau` (default `tau = 0.90`). The algorithm iterates through vertices in index order; for each vertex `i` not already marked for removal, it scans all neighbors `j > i` and marks the less-relevant member of the pair `(i, j)` for removal. If `i` is the less-relevant member, it is marked and the scan moves to the next `i`; otherwise, `j` is marked.

This greedy approach does not guarantee a globally optimal maximum independent set (which is NP-hard), but it performs well in practice for the correlation structures encountered in sports statistics, where features tend to form tight clusters (e.g., all shooting-related `diff_` and `pctdiff_` variants are mutually correlated) and the within-cluster relevance ordering provides a clear hierarchy.

The threshold of 0.90 is a configurable hyperparameter (`pearson_threshold`). Lower thresholds would remove more redundancy but risk discarding features that carry partially independent signals; higher thresholds would retain more near-duplicates.

### 4.5 Step 3: Bottom 50% Pruning

**ELI5:** After we have removed the copycat features, we rank all the surviving features by how useful they are (based on a combination of the two scoring methods we used earlier). Then we keep the top half and throw away the bottom half. We always keep at least 10 features, just in case.

**Intermediate:** Among the features that survived Pearson deduplication, the code sorts by combined relevance rank and retains only the top fraction:

```python
n_to_keep = max(10, int(len(surviving_names) * (1 - drop_bottom_pct)))
```

With `drop_bottom_pct=0.50` (the default), this step removes the bottom half of surviving features. The `max(10, ...)` guard ensures that at least 10 features survive even if the arithmetic would produce fewer, preventing the pipeline from reducing the feature set to an unusably small number.

**Advanced:** The 50% pruning threshold is a hyperparameter that balances feature parsimony (reducing overfitting, improving training speed, enhancing interpretability) against the risk of discarding marginally useful features. In the context of ~600 training examples with ~200+ deduplicated features, aggressive pruning is well-justified by the bias-variance tradeoff: the variance reduction from halving the feature set typically outweighs the slight increase in bias from losing marginally relevant predictors.

### 4.6 Output and Results

The `run()` function orchestrates the full pipeline:

1. Loads `matchup_features.parquet` from the features directory.
2. Filters to rows where `winner` is not null (needed for supervised scoring).
3. Identifies all engineered feature columns (excluding metadata columns like IDs, names, seeds, scores, and the target variable; also excluding raw `a_` and `b_` prefixed team-level stat columns, which are redundant with their differential counterparts).
4. Calls `select_features()` with the default parameters (`pearson_threshold=0.90`, `drop_bottom_pct=0.50`, `mcc_bins=4`).
5. Writes the result to `data/features/selected_features.json`:

```json
{
  "selected_features": ["seed_pct_diff", "log_seed_ratio", ...],
  "n_original": 557
}
```

### 4.7 Key Results: Top-Ranked Features

Based on the combined MI+MCC ranking, the top features consistently include:

| Rank | Feature | Why It Matters |
|---|---|---|
| #1 | `seed_pct_diff` | Normalized seed gap; recovers signal in later rounds where raw seed_diff is compressed |
| #2 | `log_seed_ratio` | Logarithmic seed ratio; captures proportional rather than absolute seed differences |
| #3+ | `seed_diff_bucket`, `efficiency_gap`, `shooting_index`, `ball_control_index` | Core domain-knowledge features encoding matchup type, overall quality gap, and specific skill dimensions |

The dominance of seed-based features at the top is consistent with decades of tournament analysis: seeding is the single strongest predictor available before the game is played. The engineered variants (`seed_pct_diff`, `log_seed_ratio`) outrank the raw `seed_diff` because they encode the information in forms that remain discriminative across the full range of tournament rounds, as detailed in [Section 3.5.6](#356-why-seed_pct_diff-and-log_seed_ratio-are-the-most-powerful-features).

A sanity check at the end of the pipeline verifies that key domain-informed features (`seed_diff`, `efficiency_gap`, `ball_control_index`, `shooting_index`) survived selection, printing `KEPT` or `DROPPED` for each.

---

## 5. Mathematical Appendix

### 5.1 Percentage Difference Derivation

The symmetric percentage difference used throughout the module (for `seed_pct_diff`, `pctdiff_{stat}`, etc.) is defined as:

```
d_%(a, b) = (a - b) / ((a + b) / 2) = 2(a - b) / (a + b)
```

**Properties:**
- **Bounded range (for positive inputs):** When `a, b > 0`, the value lies in `(-2, +2)`.
- **Antisymmetry:** `d_%(a, b) = -d_%(b, a)`.
- **Scale invariance:** `d_%(ka, kb) = d_%(a, b)` for any `k > 0`.
- **Zero-centered:** `d_%(a, a) = 0`.

The scale-invariance property is what makes this metric particularly valuable for comparing features that live on different scales. A 2-point gap in offensive efficiency (scale ~100) and a 2-point gap in three-point percentage (scale ~0.33) are qualitatively very different, but their percentage differences correctly reflect this by normalizing each gap relative to its baseline.

**Connection to logarithmic differences:** For small relative differences (i.e., when `|a - b| << (a + b) / 2`), the symmetric percentage difference is approximately equal to the difference of logarithms:

```
d_%(a, b) ~ ln(a) - ln(b)
```

This is because `ln(a/b) = ln(1 + (a-b)/b) ~ (a-b)/b ~ 2(a-b)/(a+b)` when the relative difference is small. This mathematical relationship explains why `seed_pct_diff` and `log_seed_ratio` are highly correlated but not identical, and why both can survive the Pearson deduplication threshold of 0.90.

### 5.2 Why seed\_pct\_diff Outperforms seed\_diff: An Information-Theoretic Deep Dive

The superiority of `seed_pct_diff` over `seed_diff` can be understood at multiple levels of mathematical sophistication.

**Level 1: Variance collapse.** In early rounds, `seed_diff` spans [-15, -1] with rich variance. In the Elite Eight and beyond, it compresses to approximately [-3, 0]. The variance of `seed_diff | round >= 4` is dramatically lower than `Var(seed_diff)`, which means any model that relies on `seed_diff` in late rounds has very little signal to work with. `seed_pct_diff` avoids this collapse because the denominator shrinks in proportion to the numerator, maintaining a wider effective range.

**Level 2: Nonlinear strength model.** Tournament seeds encode an ordinal ranking where the strength gap between adjacent seeds is *not constant*. If true team strength follows a power law:

```
strength(s) = C * s^(-gamma)    for gamma > 0, C > 0
```

then the probability of Team A (seed `s_a`) beating Team B (seed `s_b`) under a Bradley-Terry model is:

```
P(A wins) = strength(s_a) / (strength(s_a) + strength(s_b))
          = s_a^(-gamma) / (s_a^(-gamma) + s_b^(-gamma))
```

The log-odds of Team A winning are:

```
logit(P(A wins)) = gamma * [ln(s_b) - ln(s_a)] = -gamma * log_seed_ratio
```

This shows that `log_seed_ratio` is the **natural sufficient statistic** for seed-based win probability under a power-law strength model. `seed_diff`, by contrast, would only be sufficient under a linear strength model (`strength(s) = C - beta * s`), which is a poor fit to empirical data.

**Level 3: Conditional mutual information.** Let `W` be the winner, `D = seed_diff`, `P = seed_pct_diff`, `L = log_seed_ratio`, and `R = round_number`. We can formalize the claim that the normalized features are more informative in late rounds:

```
I(W; P | R >= 3) > I(W; D | R >= 3)
I(W; L | R >= 3) > I(W; D | R >= 3)
```

where `I(X; Y | Z)` is conditional mutual information. Intuitively, conditioning on `R >= 3` (mid-to-late rounds) restricts the sample to matchups where seeds are similar, causing `D` to have near-zero entropy (and therefore near-zero MI with anything), while `P` and `L` retain meaningful entropy because they amplify small absolute differences through division or logarithmic transformation.

**Level 4: Fisher information.** For the statistically inclined, we can consider the Fisher information that each feature carries about the latent strength parameter `gamma`. Under the power-law Bradley-Terry model, the Fisher information of `log_seed_ratio` about `gamma` is constant (it does not depend on the absolute seed values), whereas the Fisher information of `seed_diff` about `gamma` depends on the seed magnitudes and vanishes as both seeds approach 1. This explains why `log_seed_ratio` maintains consistent statistical efficiency across all rounds, while `seed_diff` becomes inefficient in late rounds.

### 5.3 Composite Feature Mathematics

This section provides the complete mathematical specification of all three composite features in a unified notation.

**Notation:** For any statistic `X`, let `X_a` and `X_b` denote Team A's and Team B's values respectively, and let `Delta_X = X_a - X_b` denote the differential.

**Efficiency Gap:**

```
EG = (OE_a - DE_a) - (OE_b - DE_b)
   = Delta_OE - Delta_DE
```

where `OE` = offensive efficiency (pts/100 possessions) and `DE` = defensive efficiency (pts allowed/100 possessions). Positive values favor Team A.

**Ball Control Index:**

```
BCI = 1.0 * Delta_ATO + 0.5 * Delta_STL - 0.3 * Delta_TOV
```

Weight vector: `w = [1.0, 0.5, -0.3]`, applied to the differential vector `[Delta_ATO, Delta_STL, Delta_TOV]`.

The negative sign on `Delta_TOV` accounts for the fact that turnovers are a *negative* indicator -- more turnovers is worse. The weights were chosen to reflect domain knowledge:
- ATO ratio (weight 1.0): most comprehensive single ball-handling metric
- Steals (weight 0.5): additional defensive ball-control signal, but noisier
- Turnovers (weight -0.3): partially captured in ATO, so down-weighted to avoid double-counting

**Shooting Index:**

```
SI = 1.0 * Delta_eFG + 0.5 * Delta_3P + 0.3 * Delta_FTR
```

Weight vector: `w = [1.0, 0.5, 0.3]`, applied to the differential vector `[Delta_eFG, Delta_3P, Delta_FTR]`.

Weights rationale:
- eFG% (weight 1.0): most comprehensive shooting efficiency metric, already accounts for 3P value
- 3P% (weight 0.5): supplementary signal for three-point reliance, partially redundant with eFG%
- FTR (weight 0.3): complementary scoring axis (drawing fouls), largely independent of shooting accuracy

**General form:** All three composites share the same structure: a weighted sum of stat differentials, `Composite = w^T * Delta_stats`. This linear structure ensures that all composites are antisymmetric under team swap (swapping A and B negates the composite), which is a necessary property for any feature used in a symmetric classification task.

### 5.4 MCC Deep Dive

The Matthews Correlation Coefficient, originally introduced by Brian Matthews in 1975 for evaluating protein secondary structure predictions, is formally equivalent to the Pearson correlation coefficient between two binary random variables. Given a confusion matrix:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

```
MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
```

**Key properties:**
- **Range:** [-1, +1].
- **Balanced:** Unlike accuracy, F1-score, or precision/recall, MCC uses all four cells of the confusion matrix and produces a high score only if the predictor performs well on *both* positive and negative classes.
- **Class-imbalance robust:** A classifier that always predicts the majority class achieves MCC = 0 (not a misleadingly high value, as accuracy would give).
- **Undefined edge case:** If any of the four sums in the denominator is zero (i.e., the predictor is constant), MCC is undefined. The `sklearn.metrics.matthews_corrcoef` implementation handles this by returning 0.0.

**Connection to chi-squared:** MCC is related to the Pearson chi-squared statistic for 2x2 contingency tables:

```
MCC = sqrt(chi^2 / N)
```

where `N` is the total sample count. This relationship shows that MCC is a normalized effect size measure, whereas chi-squared is an unnormalized test statistic that scales with sample size.

In the context of this feature selection pipeline, the "predictor" is the binarized feature (is the feature above a given threshold?) and the "actual" class is the binary `winner` target. By sweeping across quartile thresholds and taking the maximum absolute MCC, the pipeline effectively identifies the optimal univariate binary split for each feature -- a miniature decision stump evaluated using MCC.

---

## 6. Quick Reference: Feature Counts

| Stage | Approximate Count | Description |
|---|---|---|
| Raw team stat columns joined | ~250 (125 per team) | `a_*` and `b_*` prefixed |
| Raw stat differentials (`diff_*`) | ~125 | One per stat column |
| Percentage stat differentials (`pctdiff_*`) | ~125 | One per stat column |
| Away performance gap features | Variable | 3 per stat with `_season`/`_away` pair |
| Seed features | 6 | `seed_diff`, `seed_sum`, `seed_product`, `seed_pct_diff`, `log_seed_ratio`, `seed_diff_bucket` |
| Round features | 2 | `round_number`, `round_group` |
| Composite features | 3 | `efficiency_gap`, `ball_control_index`, `shooting_index` |
| **Total engineered features** | **~557+** | Exact count depends on input stat columns |
| **After selection** | **~112** | Survived MI+MCC scoring + Pearson dedup + bottom 50% pruning |

---

## 7. Usage

Both scripts expose a `run()` function and can be executed either as standalone scripts or as part of the broader pipeline.

**Standalone execution:**

```bash
# Build all matchup-level features
python -m features.build_features

# Run feature selection on the built features
python -m features.select_features
```

**Programmatic usage:**

```python
from features import build_features, select_features

# Step 1: Build features
build_features.run()

# Step 2: Select features
selected = select_features.run()
print(f"Selected {len(selected)} features: {selected}")
```

**Custom feature selection parameters:**

```python
import polars as pl
from features.select_features import select_features

df = pl.read_parquet("data/features/matchup_features.parquet")
feature_cols = [c for c in df.columns if c.startswith("diff_") or c.startswith("seed_")]

selected = select_features(
    df,
    feature_cols,
    pearson_threshold=0.85,   # stricter deduplication
    drop_bottom_pct=0.60,     # more aggressive pruning
    mcc_bins=6,               # finer MCC threshold search
    verbose=True,
)
```
