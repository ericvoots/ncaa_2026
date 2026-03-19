# Models Module: Specialist Stacking Ensemble for NCAA Tournament Prediction

This document provides a thorough and exhaustive explanation of the `models/` module, which constitutes the predictive engine of the NCAAB March Madness forecasting system. The module implements a specialist stacking ensemble architecture augmented with conformal prediction for calibrated uncertainty quantification, and a full bracket generation pipeline with historically-calibrated upset injection. Every section in this document begins with a simple, intuitive explanation suitable for a newcomer, and progressively deepens into the mathematical and architectural details that an advanced practitioner or researcher would require.

---

## Table of Contents

- [1. Module Overview](#1-module-overview)
- [2. File Inventory](#2-file-inventory)
  - [2.1 `__init__.py`](#21-__init__py)
  - [2.2 `stacking.py` -- The Specialist Stacking Engine](#22-stackingpy----the-specialist-stacking-engine)
  - [2.3 `train.py` -- Legacy Training Module](#23-trainpy----legacy-training-module)
  - [2.4 `evaluate.py` -- Cross-Validation Evaluation](#24-evaluatepy----cross-validation-evaluation)
  - [2.5 `predict.py` -- Inference and Prediction Generation](#25-predictpy----inference-and-prediction-generation)
  - [2.6 `bracket.py` -- Full Bracket Prediction and HTML Visualization](#26-bracketpy----full-bracket-prediction-and-html-visualization)
- [3. The Specialist Stacking Architecture](#3-the-specialist-stacking-architecture)
  - [3.1 Why Stacking? The Intuition](#31-why-stacking-the-intuition)
  - [3.2 The Four Base Models (Level 0)](#32-the-four-base-models-level-0)
    - [3.2.1 Model A: XGBoost General-Purpose Classifier](#321-model-a-xgboost-general-purpose-classifier)
    - [3.2.2 Model B: LightGBM Close-Game Specialist](#322-model-b-lightgbm-close-game-specialist)
    - [3.2.3 Model C: Logistic Regression Late-Round Specialist](#323-model-c-logistic-regression-late-round-specialist)
    - [3.2.4 Model D: Random Forest Diversity Model](#324-model-d-random-forest-diversity-model)
  - [3.3 The Meta-Learner (Level 1)](#33-the-meta-learner-level-1)
  - [3.4 Reducing Inter-Model Correlation](#34-reducing-inter-model-correlation)
  - [3.5 Leave-One-Year-Out Cross-Validation](#35-leave-one-year-out-cross-validation)
  - [3.6 Performance Summary](#36-performance-summary)
- [4. Conformal Prediction: Calibrated Uncertainty](#4-conformal-prediction-calibrated-uncertainty)
  - [4.1 What Is Conformal Prediction? (The Simple Version)](#41-what-is-conformal-prediction-the-simple-version)
  - [4.2 Nonconformity Scores](#42-nonconformity-scores)
  - [4.3 Round-Conditional Calibration](#43-round-conditional-calibration)
  - [4.4 Prediction Sets and Coverage Guarantees](#44-prediction-sets-and-coverage-guarantees)
  - [4.5 Confidence Tiers: HIGH, MED, LOW](#45-confidence-tiers-high-med-low)
- [5. Upset Calibration Methodology](#5-upset-calibration-methodology)
  - [5.1 Why Calibrate for Upsets?](#51-why-calibrate-for-upsets)
  - [5.2 Historical Upset Base Rates](#52-historical-upset-base-rates)
  - [5.3 The Calibration Function](#53-the-calibration-function)
  - [5.4 Global Upset Injection](#54-global-upset-injection)
- [6. Bracket Prediction Pipeline](#6-bracket-prediction-pipeline)
  - [6.1 Bracket Definition and Data Sources](#61-bracket-definition-and-data-sources)
  - [6.2 Round-by-Round Simulation](#62-round-by-round-simulation)
  - [6.3 HTML Visualization](#63-html-visualization)
- [7. Serialized Artifacts](#7-serialized-artifacts)
- [8. Dependencies](#8-dependencies)
- [9. Usage](#9-usage)

---

## 1. Module Overview

Imagine you are trying to predict who will win every single game in the NCAA March Madness basketball tournament. You could ask one very smart friend, but even better, you could ask four friends who each specialize in different kinds of games, and then have a fifth friend who is really good at listening to the other four and making a final decision. That is, at the simplest possible level, what this module does.

At a more technical level, the `models/` module implements a **two-level stacking ensemble** (also known as a "super learner" or "blender") that combines four heterogeneous base classifiers, each trained on a deliberately distinct subset of features and/or data, into a single meta-learner whose predictions are then wrapped in **conformal prediction sets** to provide distribution-free uncertainty quantification. The full pipeline further incorporates **historically-calibrated upset injection** to ensure that the generated brackets exhibit realistic upset rates consistent with the empirical distribution observed across the 2015--2025 tournament seasons.

The module achieves 71.4% cross-validated accuracy (compared to a 70.9% baseline of always picking the higher-seeded team) with a log-loss of 0.554, while providing calibrated confidence intervals that allow downstream consumers of the predictions to distinguish between games where the model is highly certain and games that are essentially coin flips.

---

## 2. File Inventory

### 2.1 `__init__.py`

An empty file whose sole purpose is to mark the `models/` directory as a Python package, allowing other modules in the project to import from it using standard Python import syntax (e.g., `from models.stacking import SpecialistModels`).

### 2.2 `stacking.py` -- The Specialist Stacking Engine

This is the heart of the entire prediction system. It contains approximately 640 lines of Python and defines the following key components:

- **`SpecialistModels`** class: Manages the training of all four base models and the generation of out-of-fold (OOF) predictions via leave-one-year-out cross-validation.
- **`StackingMetaLearner`** class: A logistic regression meta-learner that consumes the base model predictions along with engineered disagreement signals and context features.
- **`ConformalPredictor`** class: Implements round-conditional conformal prediction to produce calibrated prediction sets at multiple confidence levels.
- **`_build_meta_features()`**: Constructs the meta-learner input by concatenating raw base predictions, summary statistics (mean, standard deviation, spread, max confidence), context features, and interaction terms.
- **`_evaluate_by_segment()`**: Provides accuracy breakdowns by round group and seed-difference bucket for diagnostic purposes.
- **`run()`**: Orchestrates the full training pipeline: load data, train specialists with OOF predictions, fit the meta-learner, calibrate conformal prediction, retrain final models on all data, and save all artifacts to disk.

### 2.3 `train.py` -- Legacy Training Module

This is the original, simpler training module that has been **superseded by `stacking.py`**. It implements a straightforward leave-one-year-out cross-validation loop with three models (XGBoost, LightGBM, and Logistic Regression) combined via simple averaging rather than learned stacking. It is retained in the codebase for reference and backward compatibility but is no longer the primary training pathway. The key architectural differences from `stacking.py` are:

- All three models are trained on the same data (no specialist subsetting).
- Combination is via naive arithmetic mean rather than a learned meta-learner.
- No conformal prediction or uncertainty quantification.
- No interaction features or disagreement signals.

### 2.4 `evaluate.py` -- Cross-Validation Evaluation

A lightweight evaluation module that reads the saved `metadata.json` (produced by the legacy `train.py`) and prints a formatted table of leave-one-year-out cross-validation results including per-year accuracy, baseline accuracy, and log-loss. It computes aggregate averages and reports the improvement over the always-pick-higher-seed baseline. Note that `stacking.py` performs its own more comprehensive evaluation inline during training, so this module primarily serves the legacy pipeline.

### 2.5 `predict.py` -- Inference and Prediction Generation

This module generates predictions for the 2026 tournament using the trained stacking pipeline. Its workflow is as follows:

1. **Load artifacts**: Deserializes `specialist_models.pkl`, `meta_learner.pkl`, `conformal.pkl`, and `stacking_metadata.json` from the models directory.
2. **Build feature matrices**: For each of the four specialist models, it extracts the appropriate feature subsets from the 2026 matchup data.
3. **Generate base predictions**: Each specialist model produces P(team_a wins) for every game.
4. **Construct meta-features**: The base predictions are augmented with mean, standard deviation, spread, max confidence, context features, and interaction terms -- exactly mirroring the `_build_meta_features()` function used during training.
5. **Meta-learner inference**: The stacking meta-learner produces final calibrated probabilities.
6. **Conformal prediction sets**: At both 80% and 90% confidence levels, the conformal predictor generates prediction sets that indicate whether the model is confident in one outcome or considers both outcomes plausible.
7. **Confidence assignment**: Each game receives a confidence label:
   - **HIGH**: The 90% conformal prediction set is a singleton (contains only one team). The model is very confident.
   - **MED**: The 80% prediction set is a singleton but the 90% set contains both teams. The model leans one way but acknowledges non-trivial uncertainty.
   - **LOW**: Even the 80% prediction set contains both teams. The model considers this game genuinely uncertain.
8. **Output**: Saves predictions as both Parquet and JSON to the data directory.

### 2.6 `bracket.py` -- Full Bracket Prediction and HTML Visualization

The largest file in the module (approximately 880 lines), `bracket.py` goes beyond individual game prediction to simulate the entire 2026 NCAA tournament bracket round-by-round and produce a polished HTML visualization. Its major components include:

- **`BRACKET_2026`**: A dictionary defining all four regions (East, West, South, Midwest) with 16 seeds each, sourced from the ESPN Selection Sunday broadcast.
- **`FIRST_FOUR`**: Play-in game definitions for the four First Four matchups.
- **`HISTORICAL_UPSET_RATES`**: A lookup table mapping seed differentials to historical upset probabilities derived from 2015--2025 tournament data.
- **`NAME_MAP`**: A translation dictionary converting ESPN bracket names to TeamRankings stat names (e.g., "UConn" to "Connecticut", "Michigan St." to "Michigan St").
- **`_calibrate_with_upset_prior()`**: The upset calibration function that blends model probabilities with historical base rates, specialist disagreement, and model confidence (described in detail in Section 5).
- **`_build_matchup_features()`**: Constructs the full feature dictionary for a single game from raw team stats, including seed features, stat differentials, away-performance gaps, and composite indices (efficiency gap, ball control index, shooting index).
- **`_predict_game()`**: Runs the full stacking inference pipeline for a single game.
- **`predict_bracket()`**: Simulates the entire tournament round-by-round with global upset injection.
- **`generate_html()`**: Produces a dark-themed HTML bracket visualization with confidence-colored game outcomes, upset highlighting, and a Final Four / Championship display.

The final prediction for the 2026 tournament is Michigan (1-seed) defeating Arizona (1-seed) in the Championship with 61.8% probability at MED confidence. The Final Four matchup structure follows the ESPN bracket: East vs. Midwest on the left side, West vs. South on the right side.

---

## 3. The Specialist Stacking Architecture

### 3.1 Why Stacking? The Intuition

Suppose you are a child trying to guess how many jellybeans are in a jar. If you ask your friend who is really good at guessing big numbers, and another friend who is really good at guessing when the jar is almost half-full, and a third friend who tends to guess differently from everyone else, and then you carefully combine their guesses by learning which friend tends to be right in which situations -- you will almost always do better than any single friend alone. That is stacking.

More precisely, stacking (introduced by Wolpert, 1992) is a technique where multiple "base learners" (Level 0 models) each produce predictions, and then a "meta-learner" (Level 1 model) is trained to optimally combine those predictions. The critical insight is that the meta-learner is trained on **out-of-fold predictions** from the base models, not on their in-sample predictions, which prevents overfitting and ensures that the meta-learner learns genuine complementarities rather than memorizing training noise.

What makes this implementation a **specialist** stacking ensemble is that the base models are not merely different algorithms trained on the same data. Instead, each base model is deliberately designed to excel in a particular regime of the prediction space:

- **Model A** sees everything and provides a strong general-purpose baseline.
- **Model B** is trained exclusively on close games and specializes in the nuanced features that distinguish outcomes when seedings are similar.
- **Model C** is trained exclusively on later-round games and specializes in the features that matter most when talent levels converge (free throw shooting under pressure, defensive execution, ball security).
- **Model D** uses a fundamentally different algorithm (Random Forest) on a curated feature set to provide ensemble diversity.

This specialist design serves a dual purpose. First, it reduces the correlation between base model predictions -- a well-known requirement for effective ensembling, since combining highly correlated models yields little improvement over any single model. Second, it allows the meta-learner to implicitly perform **smart routing**: when the context features indicate a close game, the meta-learner can upweight Model B's prediction; when the context features indicate a late-round game, it can upweight Model C.

### 3.2 The Four Base Models (Level 0)

#### 3.2.1 Model A: XGBoost General-Purpose Classifier

**Simple explanation**: This is the "all-rounder" -- a powerful tree-based model that looks at every available statistic and tries to find the best combination of rules to predict who wins.

**Technical details**: Model A is an XGBoost gradient-boosted decision tree classifier trained on all features that survived the upstream feature selection process (stored in `selected_features.json`). Its hyperparameters are:

| Parameter | Value | Rationale |
|---|---|---|
| `max_depth` | 4 | Moderate depth prevents overfitting on small tournament datasets |
| `learning_rate` | 0.05 | Conservative learning rate for better generalization |
| `n_estimators` | 300 | Sufficient trees given the low learning rate |
| `subsample` | 0.8 | Row subsampling for regularization |
| `colsample_bytree` | 0.8 | Feature subsampling per tree for regularization |
| `reg_alpha` | 0.1 | L1 regularization on leaf weights |
| `reg_lambda` | 1.0 | L2 regularization on leaf weights |

Model A is trained on **all available historical matchup data** (no subsetting by game type). It produces well-calibrated probability estimates across the full spectrum of matchup types, from 1-vs-16 blowouts to 8-vs-9 toss-ups. Its feature importance rankings are reported during training and serve as the primary diagnostic for understanding which statistical differentials drive predictions.

#### 3.2.2 Model B: LightGBM Close-Game Specialist

**Simple explanation**: Some games are easy to predict -- when a 1-seed plays a 16-seed, the 1-seed almost always wins. The hard part is predicting what happens when a 5-seed plays a 12-seed, or an 8-seed plays a 9-seed. Model B is trained exclusively on these "close" games so that it becomes an expert at distinguishing outcomes when the teams are evenly matched.

**Technical details**: Model B is a LightGBM gradient-boosted decision tree classifier that is trained **only on games where |seed_diff| <= 7**. This subsetting is critical: by excluding blowout matchups from its training data, Model B's splits and leaf values are optimized entirely for the decision boundary in the close-game regime, where the signal-to-noise ratio is lowest and the marginal value of good features is highest.

Model B uses a curated set of **25 features** specifically chosen for their relevance to close-game outcomes:

- **Recency features** (last 3 games): win percentage, scoring margin, offensive/defensive efficiency, assist-to-turnover ratio, effective field goal percentage, turnovers per game. These capture current form and momentum, which matter disproportionately in close games where one team may be "peaking" at tournament time.
- **Rebounding features**: total rebounding percentage (season and last 3), offensive rebounding percentage, defensive rebounding percentage. Second-chance points are often the difference in tight games.
- **Away-performance gap features**: the difference between a team's season stats and their away stats for offensive efficiency, win percentage, scoring margin, and assist-to-turnover ratio. Tournament games are played at neutral sites (which function more like away games for most teams), so teams with large home-away performance gaps may be overrated by their season numbers.
- **Ball control features**: assist-to-turnover ratio, steals per game, turnovers per game. In close games, the team that takes care of the ball and forces turnovers tends to prevail.

Its hyperparameters are tuned slightly differently from Model A, with a shallower tree depth (`max_depth=3`), more trees (`n_estimators=400`), a lower learning rate (`learning_rate=0.03`), and stronger regularization (`reg_alpha=0.2`, `reg_lambda=1.5`) to account for the smaller training set size. A minimum of 20 close-game samples in the training fold is required before Model B is trained; otherwise, it is excluded from that fold's predictions.

#### 3.2.3 Model C: Logistic Regression Late-Round Specialist

**Simple explanation**: As the tournament progresses, the remaining teams are all very good, and different things start to matter. In the Sweet 16 and beyond, it is not enough to be a good team -- you need to make your free throws under pressure, you need to play solid defense, and you need to take care of the basketball. Model C is trained only on these later-round games to learn exactly which traits separate Final Four teams from the rest.

**Technical details**: Model C is a regularized logistic regression classifier trained **only on games from round 3 (Sweet 16) and later**. The choice of logistic regression (rather than a tree-based method) is intentional: with a much smaller training set (later rounds produce far fewer games per season), a simpler model with explicit regularization is less prone to overfitting. The features are StandardScaled before fitting, and L2 regularization is applied with `C=0.5`.

Model C uses **23 features** emphasizing:

- **Free throw execution**: FT% (season and last 3), free throw rate. In high-pressure late-round games, free throw shooting is often the decisive factor -- teams that choke at the line in the final minutes lose games they should win.
- **Defensive execution**: defensive efficiency (season and away), opponent FG%, blocks per game, steals per game. At this stage of the tournament, the team that can get stops consistently tends to advance.
- **Ball security under pressure**: turnovers per game (season and away), assist-to-turnover ratio (season and away). Late-round pressure causes turnover-prone teams to unravel.
- **Win percentage proxies for clutch performance**: overall win percentage (season and away), close-game win percentage, average scoring margin.
- **Shooting efficiency**: effective FG%, three-point percentage.

A minimum of 15 late-round samples in the training fold is required before Model C is trained.

#### 3.2.4 Model D: Random Forest Diversity Model

**Simple explanation**: If all your friends think the same way, asking more friends does not help much. Model D is deliberately different -- it uses a different algorithm (Random Forest instead of gradient boosting) and looks at a smaller, hand-picked set of statistics, so that its predictions are not just an echo of Models A and B.

**Technical details**: Model D is a scikit-learn `RandomForestClassifier` trained on all available data using a **curated set of 17 features**:

- Seed-derived features: `seed_diff`, `seed_sum`, `seed_pct_diff`, `log_seed_ratio`
- Composite indices: `efficiency_gap`, `ball_control_index`, `shooting_index`
- Key stat differentials: offensive efficiency, defensive efficiency, assist-to-turnover ratio, win percentage, average scoring margin, effective FG%, turnovers per game, total rebounding percentage
- Away-specific stats: assist-to-turnover ratio (away), offensive efficiency (away)

Its hyperparameters are `n_estimators=300`, `max_depth=6`, `min_samples_leaf=10`, with `n_jobs=-1` for parallel tree construction. The deliberate use of a bagging-based method (Random Forest) rather than a boosting-based method provides algorithmic diversity, which is one of the most effective ways to reduce inter-model correlation in ensemble learning (see Breiman, 1996; Dietterich, 2000).

### 3.3 The Meta-Learner (Level 1)

**Simple explanation**: The meta-learner is the "final decision maker" that listens to all four specialist models and decides who to trust for each game. It does not just average their opinions -- it also pays attention to how much the specialists agree or disagree, what round the game is in, and how close the seeds are.

**Technical details**: The meta-learner is a regularized logistic regression (`C=0.5`, `max_iter=1000`) wrapped with a `StandardScaler`. Its input feature vector is constructed by the `_build_meta_features()` function and consists of the following components, concatenated horizontally:

1. **Raw base model predictions** (4 values): The P(team_a wins) output from each specialist model. These are the primary signal.

2. **Ensemble summary statistics** (4 values):
   - **Mean prediction**: The arithmetic mean of the four base probabilities. Provides a simple consensus signal.
   - **Standard deviation**: The standard deviation of the four base probabilities. High values indicate specialist disagreement, which is a signal of game difficulty and uncertainty.
   - **Spread**: max(predictions) - min(predictions). Similar to standard deviation but more sensitive to a single outlier model.
   - **Max confidence**: max(|prediction - 0.5|) across all models. Captures whether at least one model is highly confident, even if others disagree.

3. **Context features** (7 values): `seed_diff`, `seed_sum`, `seed_pct_diff`, `log_seed_ratio`, `round_number`, `round_group`, `seed_diff_bucket`. These give the meta-learner direct access to game context, allowing it to learn regime-dependent weighting (e.g., trust Model B more when `seed_diff_bucket` is 0 or 1).

4. **Interaction terms** (2 values):
   - **round_number x seed_diff_bucket**: Captures the interaction between tournament stage and matchup closeness. A close game in the Elite 8 is qualitatively different from a close game in the Round of 64.
   - **round_number x std_pred**: Captures the interaction between tournament stage and model disagreement. Disagreement in later rounds may have different implications than disagreement in early rounds.

The total meta-feature vector dimensionality is therefore 4 + 4 + 7 + 2 = **17 features**. The choice of logistic regression as the meta-learner (rather than, say, another gradient-boosted tree) is deliberate: the meta-learner's input space is already highly engineered and relatively low-dimensional, so a simple linear model is sufficient and far less prone to overfitting the relatively small number of training samples.

### 3.4 Reducing Inter-Model Correlation

One of the central design goals of this architecture is the reduction of inter-model prediction correlation. When base models in an ensemble produce highly correlated predictions, the ensemble's effective diversity is low, and combining them yields minimal improvement over any single model. This is formalized by the bias-variance-covariance decomposition of ensemble error (Brown et al., 2005):

```
E[ensemble_error] = mean_bias + (1/M) * mean_variance + (1 - 1/M) * mean_covariance
```

where M is the number of base models. Reducing the mean covariance term directly reduces ensemble error. The specialist stacking architecture achieves this through three complementary strategies:

1. **Algorithmic diversity**: XGBoost (boosted trees), LightGBM (boosted trees with histogram-based splitting), Logistic Regression (linear model), and Random Forest (bagged trees) use fundamentally different learning algorithms.

2. **Feature diversity**: Each model operates on a different feature subset (all selected features, 25 close-game features, 23 late-round features, 17 curated features), ensuring that models attend to different aspects of the data.

3. **Data diversity**: Models B and C are trained on deliberately restricted subsets of the training data (close games only, late rounds only), which means their learned decision boundaries are optimized for different regions of the input space.

The result is a reduction in mean pairwise inter-model correlation from **r = 0.874** (which is what you would observe if all models were simply different algorithms trained on the same data and features) to **r = 0.563**. This 35% reduction in correlation translates directly to improved ensemble performance.

### 3.5 Leave-One-Year-Out Cross-Validation

**Simple explanation**: To know if our model is actually good and not just memorizing old games, we play a game: we hide one entire year of tournament results, train the model on everything else, and then see how well it predicts the hidden year. We do this for every year, one at a time.

**Technical details**: The cross-validation scheme is leave-one-year-out (LOYO), where each fold holds out all games from a single tournament season. This is the appropriate cross-validation strategy for this domain because:

1. **Temporal leakage prevention**: Randomly splitting games across years would allow the model to learn from, say, 2023 tournament games when predicting other 2023 tournament games, which inflates accuracy estimates because games within the same tournament share latent factors (e.g., which teams are "hot" that year).

2. **Realistic evaluation**: In production, the model will always be predicting a new, unseen tournament year. LOYO precisely simulates this scenario.

3. **Out-of-fold predictions for stacking**: The OOF predictions generated during LOYO are used as the training data for the meta-learner. Each game's meta-learner training sample consists of base model predictions that were generated when that game's year was held out, ensuring that the meta-learner is trained on genuinely out-of-sample base predictions.

The procedure for each fold (test year `t`):

1. Train all four specialist models on data from all years except `t`.
2. Generate predictions from all four models on the held-out year `t`.
3. Store these predictions as the OOF predictions for year `t`.

After all folds are complete, the full OOF prediction matrix is assembled and used to train the meta-learner. A per-year accuracy breakdown is printed for diagnostic purposes.

### 3.6 Performance Summary

| Metric | Value |
|---|---|
| Stacking CV Accuracy | 71.4% |
| Always-Pick-Higher-Seed Baseline | 70.9% |
| Improvement Over Baseline | +0.5% |
| Log-Loss | 0.554 |
| Simple Average Ensemble Accuracy | (reported during training) |
| Mean Pairwise Model Correlation | 0.563 |

While a 0.5% improvement over the seed baseline may appear modest, it is important to understand that the seed baseline is extraordinarily strong in this domain. Seeds encapsulate a tremendous amount of information (the selection committee's judgment, which itself incorporates team strength, schedule difficulty, injuries, and more), and beating it consistently over a leave-one-year-out evaluation is a meaningful achievement. Furthermore, the model's primary value lies not in marginal accuracy gains but in its **calibrated probability estimates** and **uncertainty quantification**, which enable intelligent bet sizing, bracket optimization, and risk management.

---

## 4. Conformal Prediction: Calibrated Uncertainty

### 4.1 What Is Conformal Prediction? (The Simple Version)

Imagine a weather forecaster who says "I am 90% sure it will rain tomorrow." If you check their forecasts over many days, you want to find that on the days they said "90% sure about rain," it actually rained about 90% of the time. If it only rained 60% of the time, their forecasts are badly calibrated and you cannot trust the confidence numbers.

Conformal prediction is a technique that takes any model's predictions and wraps them in a "prediction set" that has a mathematically guaranteed coverage rate. Instead of saying "Team A will win with 72% probability" (which might be poorly calibrated), conformal prediction says "At the 90% confidence level, the prediction set contains {Team A}" (meaning: if we do this for many games, at least 90% of the time the actual winner will be in the set). When the model is uncertain, the prediction set will contain both teams, honestly admitting "we don't know."

### 4.2 Nonconformity Scores

The mathematical foundation of conformal prediction rests on the concept of **nonconformity scores**. A nonconformity score measures how "surprising" or "unusual" a particular outcome is relative to the model's prediction. For a binary classification problem where the model outputs P(Y=1) = p, the nonconformity score for a sample with true label y is defined as:

```
alpha_i = 1 - p_i          if y_i = 1   (model predicted high probability for the correct class)
alpha_i = p_i              if y_i = 0   (model predicted low probability for team_a, correct since team_b won)
```

Or, more compactly:

```
alpha_i = 1 - f(x_i)_{y_i}
```

where f(x_i)_{y_i} is the model's predicted probability for the true class. When the model is correct and confident, alpha_i is small (close to 0). When the model is wrong or uncertain, alpha_i is large (close to 1).

In the implementation, this is computed as:

```python
scores = np.where(y == 1, 1 - probs, probs)
```

### 4.3 Round-Conditional Calibration

**Simple explanation**: A Round-of-64 game between a 1-seed and a 16-seed is a very different kind of prediction problem than a Final Four game between two 1-seeds. It would be misleading to use the same uncertainty thresholds for both. So we calibrate our confidence levels separately for different stages of the tournament.

**Technical details**: The `ConformalPredictor` class implements **conditional conformal prediction** by partitioning the calibration data into four round groups:

| Group | Rounds | Description |
|---|---|---|
| 0 | First Four | Play-in games |
| 1 | R64 + R32 | Early rounds (most games, most predictable) |
| 2 | Sweet 16 + Elite 8 | Mid rounds (talent converges) |
| 3 | Final Four + Championship | Late rounds (highest uncertainty) |

For each round group `g` and each confidence level `1 - epsilon` (where epsilon is in {0.05, 0.10, 0.15, 0.20}, corresponding to confidence levels 0.95, 0.90, 0.85, 0.80), the calibration procedure computes the quantile threshold:

```
q_hat(g, epsilon) = Q_{ceil((n_g + 1)(1 - epsilon)) / n_g}(alpha_{i : group(i) = g})
```

where Q_p denotes the p-th quantile of the sorted nonconformity scores within round group g, and n_g is the number of calibration samples in that group. The finite-sample correction factor `ceil((n_g + 1)(1 - epsilon)) / n_g` ensures that the coverage guarantee holds exactly (Vovk, Gammerman, and Shafer, 2005).

In code:

```python
q_level = min(conf_level, (np.ceil((n + 1) * conf_level)) / n) if n > 0 else 1.0
q = np.quantile(group_scores, min(q_level, 1.0)) if n > 0 else 1.0
self.thresholds[rg][conf_level] = q
```

### 4.4 Prediction Sets and Coverage Guarantees

At prediction time, for a new game with model probability p and round group g, the prediction set at confidence level 1 - epsilon is constructed as:

```
C(x) = { y in {0, 1} : alpha(x, y) <= q_hat(g, epsilon) }
```

Expanding this:

- **Include team_a (y=1)** in the prediction set if: `(1 - p) <= q_hat(g, epsilon)`
- **Include team_b (y=0)** in the prediction set if: `p <= q_hat(g, epsilon)`

The resulting prediction set can be:

- **Singleton {1}**: Only team_a is included. The model is confident that team_a wins at this confidence level.
- **Singleton {0}**: Only team_b is included. The model is confident that team_b wins.
- **Both {0, 1}**: Both teams are included. The model considers this game genuinely uncertain at this confidence level.
- **Empty set**: In the degenerate case where neither team is included (which can occur with extreme probability values), the implementation defaults to the single most likely team.

The fundamental theoretical guarantee of conformal prediction is:

```
P(Y_{n+1} in C(X_{n+1})) >= 1 - epsilon
```

This holds under the sole assumption of exchangeability (the calibration and test samples are drawn from the same distribution), which is a much weaker assumption than the parametric distributional assumptions required by traditional confidence intervals. The round-conditional variant maintains this guarantee within each round group separately, at the cost of slightly wider sets in groups with fewer calibration samples.

### 4.5 Confidence Tiers: HIGH, MED, LOW

The prediction pipeline collapses the conformal prediction sets at two confidence levels (80% and 90%) into three human-readable confidence tiers:

| Tier | Condition | Interpretation |
|---|---|---|
| **HIGH** | 90% conformal set is a singleton | The model is very confident. Even at the stringent 90% level, only one team is plausible. These predictions are the most trustworthy. |
| **MED** | 80% set is a singleton, but 90% set contains both teams | The model leans one way, but acknowledges meaningful uncertainty. At 80% confidence the model commits, but at 90% it hedges. |
| **LOW** | 80% conformal set contains both teams | The model considers this game genuinely uncertain even at the lenient 80% level. These are essentially coin-flip games. |

---

## 5. Upset Calibration Methodology

### 5.1 Why Calibrate for Upsets?

**Simple explanation**: Models trained on historical data tend to be conservative -- they learn that the higher-seeded team usually wins, and so they predict the higher seed in almost every game. But in real March Madness tournaments, upsets happen at predictable rates: roughly 25--30% of games are upsets overall. If the model only predicts 10% upsets, your bracket will look unrealistically "chalky" and will be badly wrong in most pools. The upset calibration ensures that the model's upset rate matches what actually happens historically.

**Technical details**: Machine learning classifiers trained to minimize log-loss on imbalanced data (where the higher seed wins ~71% of the time) will, at prediction time, produce probability estimates that are reasonably well-calibrated in aggregate but tend to systematically underestimate the probability of the minority class (upsets) in specific regimes. This is particularly problematic for bracket generation, where the downstream task requires selecting a specific set of upsets -- not just assigning probabilities.

### 5.2 Historical Upset Base Rates

The `HISTORICAL_UPSET_RATES` dictionary encodes the empirical upset frequency for each seed differential observed in the 2015--2025 tournament data:

| Seed Matchup | seed_diff | Upset Rate |
|---|---|---|
| 1 vs 16 | -15 | 5% |
| 2 vs 15 | -13 | 10% |
| 3 vs 14 | -11 | 12% |
| 4 vs 13 | -9 | 24% |
| 5 vs 12 | -7 | 30% |
| 6 vs 11 | -5 | 48% |
| 7 vs 10 | -4 | 32% |
| 8 vs 9 | -3 | 33% |

These rates serve as Bayesian priors that anchor the model's predictions toward historically realistic upset frequencies.

### 5.3 The Calibration Function

The `_calibrate_with_upset_prior()` function implements a dynamic blending of the model's raw probability with the historical base rate. The blending is governed by three signals:

**1. Model confidence weighting**: The model's own confidence level determines how much weight the historical prior receives. When the model is highly confident (probability far from 0.5), the prior receives less weight; when the model is uncertain (probability near 0.5), the prior receives more weight.

```
model_confidence = |model_prob - 0.5| * 2       (ranges from 0 to 1)
```

**2. Specialist disagreement boost**: When multiple base models predict an upset (their probability for team_a is below 0.5), the historical upset prior receives additional weight. This reflects the intuition that when specialists independently agree on an upset, there is meaningful signal that should not be overridden by the meta-learner's tendency toward the higher seed.

```
disagree_boost = 0.15    if >= 2 specialists predict upset
disagree_boost = 0.08    if >= 1 specialist predicts upset
disagree_boost = 0.0     otherwise
```

**3. Effective weight computation and blending**:

```
effective_weight = base_weight * (1 - model_confidence^1.2) + disagree_boost
calibrated_prob = model_prob * (1 - effective_weight) + base_prob * effective_weight
```

The exponent 1.2 on `model_confidence` creates a slightly superlinear decay: the prior's influence diminishes faster-than-linearly as model confidence increases, but remains substantial in the uncertain regime (model_confidence near 0). The default `base_weight` is 0.35.

For example, consider a 5-vs-12 game where the model gives P(5-seed wins) = 0.62:
- `model_confidence = |0.62 - 0.5| * 2 = 0.24`
- `base_prob = 1.0 - 0.30 = 0.70` (historical P(higher seed wins) for 5-vs-12)
- If no specialist disagreement: `effective_weight = 0.35 * (1 - 0.24^1.2) = 0.35 * 0.80 = 0.28`
- `calibrated_prob = 0.62 * 0.72 + 0.70 * 0.28 = 0.45 + 0.20 = 0.643`

The calibration nudges the probability slightly toward the historical base rate, and in cases with specialist disagreement, can push marginal games across the 0.5 threshold into upset territory.

### 5.4 Global Upset Injection

Beyond per-game calibration, the `bracket.py` module implements a **global upset injection** mechanism to ensure that the overall number of upsets per round matches historical expectations. The expected upset counts per round (across all four regions combined) are:

| Round | Expected Upsets | Historical Rate |
|---|---|---|
| Round of 64 | 9 | ~28% of 32 games |
| Round of 32 | 3 | ~19% of 16 games |
| Sweet 16 | 3 | ~38% of 8 games |
| Elite 8 | 2 | ~50% of 4 games |

The overall target is approximately **29% upset rate** across the entire tournament, consistent with the empirical average from the 2015--2025 period.

The upset selection algorithm works as follows:

1. All games in a round are predicted by the model, yielding calibrated probabilities and per-game upset probabilities.
2. Games are ranked by their upset probability in descending order.
3. Games where a 1-seed faces a 16-seed are excluded from upset candidacy (this matchup type has only been upset once in tournament history).
4. Games where the model gives the upset less than 20% probability are excluded (the model is confident enough that forcing an upset would be unreasonable).
5. The top N candidates (where N = expected upsets for that round) are flipped to upset outcomes.

This mechanism ensures that the generated bracket exhibits a realistic upset profile rather than the systematically-too-chalky profile that a pure probability-maximizing approach would produce.

---

## 6. Bracket Prediction Pipeline

### 6.1 Bracket Definition and Data Sources

The 2026 bracket is defined in the `BRACKET_2026` dictionary, sourced from the ESPN Selection Sunday broadcast. It contains all 68 teams organized into four regions (East, West, South, Midwest), each with seeds 1 through 16. Some slots are marked as "TBD" pending First Four play-in results.

Team name resolution is handled by the `NAME_MAP` dictionary, which translates ESPN's team naming conventions to the TeamRankings naming conventions used in the statistical database. For example, "UConn" becomes "Connecticut", and "Michigan St." becomes "Michigan St". When direct mapping fails, the system falls back to the `TeamIndex` fuzzy matching system.

### 6.2 Round-by-Round Simulation

The bracket is simulated forward from the Round of 64 through the Championship:

1. **Round of 64**: All 32 games across all four regions are predicted simultaneously. Upsets are selected globally (not per-region) to match the expected count of 9.
2. **Round of 32**: Winners from R64 are paired according to bracket structure. Predictions and upset selection proceed with an expected count of 3.
3. **Sweet 16**: 8 games, expected 3 upsets.
4. **Elite 8**: 4 games, expected 2 upsets. Region champions are determined.
5. **Final Four**: East champion vs. Midwest champion, West champion vs. South champion (per the ESPN 2026 bracket structure).
6. **Championship**: The two Final Four winners face off.

For each game, the full stacking pipeline runs: feature construction from team stats, four specialist model predictions, meta-feature engineering, meta-learner inference, upset calibration with historical priors, and conformal prediction.

### 6.3 HTML Visualization

The `generate_html()` function produces a self-contained HTML file with:

- **Dark theme**: A deep navy/purple background (`#0a0a1a`) with high-contrast text, designed for comfortable viewing.
- **Four-region grid layout**: Regions are displayed in a 2x2 CSS grid, each containing collapsible round sections.
- **Confidence coloring**: Game outcomes are color-coded by confidence tier -- green for HIGH, yellow for MED, red for LOW.
- **Upset highlighting**: Upset picks are displayed with a red left border and a tinted background (`#2a1a1a`).
- **Model agreement display**: Each game shows how many of the four base models agree with the final prediction (e.g., "3/4").
- **Final Four and Championship**: Displayed in a full-width section spanning both grid columns, with the predicted champion prominently featured.
- **Legend**: A footer legend explains the confidence color coding.

---

## 7. Serialized Artifacts

The training pipeline produces the following serialized files, all saved to the configured `MODELS_DIR`:

| File | Format | Contents |
|---|---|---|
| `specialist_models.pkl` | Python pickle | Dictionary containing all four base models and the Model C scaler. Keys: `model_a_xgb`, `model_b_close`, `model_c_late`, `model_c_scaler`, `model_d_rf`. |
| `meta_learner.pkl` | Python pickle | The `StackingMetaLearner` instance (wraps a `StandardScaler` and `LogisticRegression`). |
| `conformal.pkl` | Python pickle | The `ConformalPredictor` instance with round-conditional thresholds for all confidence levels. |
| `stacking_metadata.json` | JSON | Feature lists (selected, curated, close, late, context), model names, CV accuracy, log-loss, MCC, baseline accuracy, training set size, and year list. |
| `metadata.json` | JSON | Legacy metadata from `train.py` (feature columns, model list, CV results). |
| `xgb_model.pkl` | Python pickle | Legacy XGBoost model from `train.py`. |
| `lgb_model.pkl` | Python pickle | Legacy LightGBM model from `train.py`. |
| `lr_model.pkl` | Python pickle | Legacy Logistic Regression model from `train.py`. |
| `scaler.pkl` | Python pickle | Legacy StandardScaler from `train.py`. |

---

## 8. Dependencies

The module requires the following Python packages:

| Package | Used By | Purpose |
|---|---|---|
| `numpy` | All files | Array operations, numerical computation |
| `polars` | All files | DataFrame operations, Parquet I/O |
| `scikit-learn` | `stacking.py`, `train.py`, `evaluate.py` | LogisticRegression, RandomForest, StandardScaler, metrics |
| `xgboost` | `stacking.py`, `train.py` | XGBoost gradient-boosted trees (optional, gracefully degrades) |
| `lightgbm` | `stacking.py`, `train.py` | LightGBM gradient-boosted trees (optional, gracefully degrades) |
| `config` | All files | Project-wide path configuration (MODELS_DIR, FEATURES_DIR, etc.) |
| `team_index` | `bracket.py` | Fuzzy team name resolution |

Both `xgboost` and `lightgbm` are imported with try/except guards and feature flags (`HAS_XGB`, `HAS_LGB`), so the system degrades gracefully if either library is unavailable, though prediction quality will be reduced.

---

## 9. Usage

**Training the stacking ensemble** (produces all serialized artifacts):

```bash
python -m models.stacking
```

**Running the legacy training pipeline** (superseded, retained for reference):

```bash
python -m models.train
```

**Evaluating the legacy model** (reads metadata.json from legacy train):

```bash
python -m models.evaluate
```

**Generating 2026 predictions** (requires stacking artifacts):

```bash
python -m models.predict
```

**Generating the full bracket with HTML visualization** (requires stacking artifacts and 2026 team stats):

```bash
python -m models.bracket
```

The standard workflow is: run `stacking.py` once to train and save all models, then run `bracket.py` to produce the full tournament bracket and HTML visualization.
