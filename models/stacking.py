"""Specialist stacking meta-learner + conformal prediction for tournament predictions.

Architecture:
    Model A (XGBoost): General-purpose, all selected features, trained on all data
    Model B (LightGBM): Close-game specialist, recency/rebounding/away features, trained on |seed_diff|<=7
    Model C (LogReg): Later-round/upset specialist, FT%/ball security/defense features, trained on round>=3
    Model D (Random Forest): Diversity model on curated features, trained on all data

    Meta-learner: LogReg on base predictions + disagreement signals + context features
    Smart routing: efficiency_gap + round_number determine specialist weighting
    Conformal prediction: round-conditional calibrated uncertainty
"""
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, matthews_corrcoef

import config

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def _load_selected_features() -> list[str] | None:
    path = config.FEATURES_DIR / "selected_features.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("selected_features", None)


def _get_curated_features() -> list[str]:
    return [
        "seed_diff", "seed_sum", "seed_pct_diff", "log_seed_ratio",
        "efficiency_gap", "ball_control_index", "shooting_index",
        "diff_offensive-efficiency_season", "diff_defensive-efficiency_season",
        "diff_assist--per--turnover-ratio_season", "diff_win-pct-all-games_season",
        "diff_average-scoring-margin_season", "diff_effective-field-goal-pct_season",
        "diff_turnovers-per-game_season", "diff_total-rebounding-percentage_season",
        "diff_assist--per--turnover-ratio_away", "diff_offensive-efficiency_away",
    ]


def _get_close_game_features() -> list[str]:
    """Features for close-game specialist (Model B).

    Focus: recency stats, rebounding, away performance gaps, ball control.
    These distinguish outcomes when seeds are similar.
    """
    return [
        "seed_pct_diff", "log_seed_ratio", "seed_diff", "seed_sum",
        "efficiency_gap", "ball_control_index", "shooting_index",
        # Recency (last 3 games)
        "diff_win-pct-all-games_last3", "diff_average-scoring-margin_last3",
        "diff_offensive-efficiency_last3", "diff_defensive-efficiency_last3",
        "diff_assist--per--turnover-ratio_last3",
        "diff_effective-field-goal-pct_last3",
        "diff_turnovers-per-game_last3",
        # Rebounding
        "diff_total-rebounding-percentage_season",
        "diff_offensive-rebounding-pct_season",
        "diff_defensive-rebounding-pct_season",
        "diff_total-rebounding-percentage_last3",
        # Away performance gaps
        "diff_awaygap_offensive-efficiency",
        "diff_awaygap_win-pct-all-games",
        "diff_awaygap_average-scoring-margin",
        "diff_awaygap_assist--per--turnover-ratio",
        # Ball control
        "diff_assist--per--turnover-ratio_season",
        "diff_steals-per-game_season",
        "diff_turnovers-per-game_season",
    ]


def _get_late_round_features() -> list[str]:
    """Features for later-round/upset specialist (Model C).

    Focus: FT%, ball security, defensive execution, composites.
    These matter most when talent levels are close (later rounds).
    """
    return [
        "seed_pct_diff", "log_seed_ratio", "seed_diff",
        "efficiency_gap", "ball_control_index", "shooting_index",
        # Free throw execution (critical in close late games)
        "diff_free-throw-pct_season", "diff_free-throw-rate_season",
        "diff_free-throw-pct_last3",
        # Defensive execution
        "diff_defensive-efficiency_season",
        "diff_defensive-efficiency_away",
        "diff_opponent-field-goal-pct_season",
        "diff_blocks-per-game_season",
        "diff_steals-per-game_season",
        # Ball security under pressure
        "diff_turnovers-per-game_season",
        "diff_turnovers-per-game_away",
        "diff_assist--per--turnover-ratio_season",
        "diff_assist--per--turnover-ratio_away",
        # Win pct proxies for clutch
        "diff_win-pct-all-games_season",
        "diff_win-pct-all-games_away",
        "diff_win-pct-close-games_season",
        "diff_average-scoring-margin_season",
        # Shooting efficiency
        "diff_effective-field-goal-pct_season",
        "diff_three-point-pct_season",
    ]


def _build_meta_features(base_preds: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Build meta-learner input from base predictions + context.

    Features:
    - Raw base model predictions (n_models)
    - Mean, std, spread, max_confidence (disagreement signals)
    - Context features (seed_diff, round, etc.)
    - Interaction: round_number * seed_diff_bucket
    """
    mean_pred = np.mean(base_preds, axis=1, keepdims=True)
    std_pred = np.std(base_preds, axis=1, keepdims=True)
    spread = (np.max(base_preds, axis=1, keepdims=True) -
              np.min(base_preds, axis=1, keepdims=True))
    max_conf = np.max(np.abs(base_preds - 0.5), axis=1, keepdims=True)

    # Interaction features from context
    # context cols: seed_diff(0), seed_sum(1), seed_pct_diff(2), log_seed_ratio(3),
    #               round_number(4), round_group(5), seed_diff_bucket(6)
    interactions = []
    if context.shape[1] >= 7:
        # round_number * seed_diff_bucket
        interactions.append(
            (context[:, 4:5] * context[:, 6:7])
        )
        # round_number * efficiency_gap proxy (std captures model disagreement ~= game difficulty)
        interactions.append(
            (context[:, 4:5] * std_pred)
        )

    parts = [
        base_preds,
        mean_pred,
        std_pred,
        spread,
        max_conf,
        context,
    ] + interactions

    return np.hstack(parts)


class SpecialistModels:
    """Train specialist base models and generate out-of-fold predictions."""

    def __init__(self, selected_features: list[str], curated_features: list[str],
                 close_features: list[str], late_features: list[str]):
        self.selected_features = selected_features
        self.curated_features = curated_features
        self.close_features = close_features
        self.late_features = late_features

    def _prepare_feature_matrices(self, df: pl.DataFrame) -> dict:
        """Prepare all feature matrices from a dataframe."""
        avail_sel = [c for c in self.selected_features if c in df.columns]
        avail_cur = [c for c in self.curated_features if c in df.columns]
        avail_close = [c for c in self.close_features if c in df.columns]
        avail_late = [c for c in self.late_features if c in df.columns]

        return {
            "X_sel": np.nan_to_num(df.select(avail_sel).to_numpy(), nan=0.0),
            "X_cur": np.nan_to_num(df.select(avail_cur).to_numpy(), nan=0.0),
            "X_close": np.nan_to_num(df.select(avail_close).to_numpy(), nan=0.0),
            "X_late": np.nan_to_num(df.select(avail_late).to_numpy(), nan=0.0),
            "cols_sel": avail_sel,
            "cols_cur": avail_cur,
            "cols_close": avail_close,
            "cols_late": avail_late,
        }

    def _train_specialists(self, train_df: pl.DataFrame, matrices: dict,
                           train_mask: np.ndarray, y_all: np.ndarray) -> dict:
        """Train all 4 specialist models on one fold."""
        models = {}
        y_train = y_all[train_mask]

        X_sel_train = matrices["X_sel"][train_mask]
        X_cur_train = matrices["X_cur"][train_mask]
        X_close_train = matrices["X_close"][train_mask]
        X_late_train = matrices["X_late"][train_mask]

        # Model A: XGBoost general-purpose on all selected features
        if HAS_XGB:
            m = xgb.XGBClassifier(
                max_depth=4, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            m.fit(X_sel_train, y_train)
            models["model_a_xgb"] = m

        # Model B: LightGBM close-game specialist
        # Train ONLY on close games (|seed_diff| <= 7) but predict on all
        if HAS_LGB:
            seed_diffs = train_df["seed_diff"].to_numpy() if "seed_diff" in train_df.columns else None
            if seed_diffs is not None:
                close_mask_local = np.abs(seed_diffs) <= 7
                X_close_sub = X_close_train[close_mask_local]
                y_close_sub = y_train[close_mask_local]

                if len(y_close_sub) >= 20:
                    m = lgb.LGBMClassifier(
                        max_depth=3, learning_rate=0.03, n_estimators=400,
                        subsample=0.8, colsample_bytree=0.7,
                        min_child_samples=15, reg_alpha=0.2, reg_lambda=1.5,
                        random_state=42, verbose=-1,
                    )
                    m.fit(X_close_sub, y_close_sub)
                    models["model_b_close"] = m

        # Model C: LogReg later-round/upset specialist
        # Train ONLY on later rounds (round >= 3) but predict on all
        round_nums = train_df["round_number"].to_numpy() if "round_number" in train_df.columns else None
        if round_nums is not None:
            late_mask_local = round_nums >= 3
            X_late_sub = X_late_train[late_mask_local]
            y_late_sub = y_train[late_mask_local]

            if len(y_late_sub) >= 15:
                scaler_c = StandardScaler()
                X_late_scaled = scaler_c.fit_transform(X_late_sub)
                m = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
                m.fit(X_late_scaled, y_late_sub)
                models["model_c_late"] = m
                models["model_c_scaler"] = scaler_c

        # Model D: Random Forest on curated features (diversity)
        m = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        m.fit(X_cur_train, y_train)
        models["model_d_rf"] = m

        return models

    def _predict_all(self, models: dict, matrices: dict,
                     mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Get predictions from all specialist models. Returns (n_samples, n_models)."""
        X_sel = matrices["X_sel"][mask]
        X_close = matrices["X_close"][mask]
        X_late = matrices["X_late"][mask]
        X_cur = matrices["X_cur"][mask]

        preds = []
        names = []

        if "model_a_xgb" in models:
            preds.append(models["model_a_xgb"].predict_proba(X_sel)[:, 1])
            names.append("xgb_general")

        if "model_b_close" in models:
            preds.append(models["model_b_close"].predict_proba(X_close)[:, 1])
            names.append("lgb_close")

        if "model_c_late" in models:
            X_late_scaled = models["model_c_scaler"].transform(X_late)
            preds.append(models["model_c_late"].predict_proba(X_late_scaled)[:, 1])
            names.append("lr_late")

        if "model_d_rf" in models:
            preds.append(models["model_d_rf"].predict_proba(X_cur)[:, 1])
            names.append("rf_diverse")

        return np.column_stack(preds) if preds else np.empty((X_sel.shape[0], 0)), names

    def generate_oof_predictions(self, df: pl.DataFrame, years: list[int]) -> dict:
        """Generate out-of-fold predictions using leave-one-year-out."""
        matrices = self._prepare_feature_matrices(df)
        y_all = df["winner"].to_numpy().astype(float)
        seasons = df["season"].to_numpy()

        # Context features for meta-learner
        context_cols = [
            "seed_diff", "seed_sum", "seed_pct_diff", "log_seed_ratio",
            "round_number", "round_group", "seed_diff_bucket",
        ]
        avail_ctx = [c for c in context_cols if c in df.columns]
        X_ctx_all = np.nan_to_num(df.select(avail_ctx).to_numpy(), nan=0.0)

        n_samples = len(df)
        oof_preds = None
        model_names = None
        fold_indices = []
        year_models = {}

        for test_year in years:
            train_mask = seasons != test_year
            test_mask = seasons == test_year
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            fold_indices.append((train_idx, test_idx))

            train_df = df.filter(pl.col("season") != test_year)
            models = self._train_specialists(train_df, matrices, train_mask, y_all)
            year_models[test_year] = models

            preds, mnames = self._predict_all(models, matrices, test_mask)

            if model_names is None:
                model_names = mnames
                oof_preds = np.full((n_samples, len(mnames)), np.nan)

            oof_preds[test_idx, :preds.shape[1]] = preds

        # Build meta features
        oof_meta = _build_meta_features(oof_preds, X_ctx_all)

        return {
            "oof_preds": oof_preds,
            "oof_meta_features": oof_meta,
            "y": y_all,
            "model_names": model_names,
            "fold_indices": fold_indices,
            "year_models": year_models,
            "matrices": matrices,
            "X_ctx_all": X_ctx_all,
            "context_cols": avail_ctx,
        }


class StackingMetaLearner:
    """Meta-learner that combines specialist predictions with smart routing."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X_meta: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_meta)
        self.model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_meta)
        return self.model.predict_proba(X_scaled)[:, 1]


class ConformalPredictor:
    """Conformal prediction with round-conditional nonconformity scores.

    Pools rounds into groups:
        Group 0: First Four
        Group 1: R64 + R32
        Group 2: Sweet 16 + Elite 8
        Group 3: Final Four + Championship
    """

    def __init__(self):
        self.thresholds: dict[int, dict[float, float]] = {}
        self.confidence_levels = [0.80, 0.85, 0.90, 0.95]

    def calibrate(self, probs: np.ndarray, y: np.ndarray, round_groups: np.ndarray):
        scores = np.where(y == 1, 1 - probs, probs)

        for rg in sorted(set(round_groups)):
            mask = round_groups == rg
            group_scores = np.sort(scores[mask])
            n = len(group_scores)

            self.thresholds[rg] = {}
            for conf_level in self.confidence_levels:
                q_level = min(conf_level, (np.ceil((n + 1) * conf_level)) / n) if n > 0 else 1.0
                q = np.quantile(group_scores, min(q_level, 1.0)) if n > 0 else 1.0
                self.thresholds[rg][conf_level] = q

            if n > 0:
                print(f"    Round group {rg}: {n} games, "
                      f"median score={np.median(group_scores):.3f}, "
                      f"90% threshold={self.thresholds[rg].get(0.90, 'N/A'):.3f}")

    def predict_sets(self, probs: np.ndarray, round_groups: np.ndarray,
                     confidence: float = 0.90) -> list[dict]:
        results = []
        for i in range(len(probs)):
            p = probs[i]
            rg = round_groups[i]

            if rg in self.thresholds and confidence in self.thresholds[rg]:
                thresh = self.thresholds[rg][confidence]
            else:
                all_thresholds = [self.thresholds[g][confidence]
                                  for g in self.thresholds if confidence in self.thresholds[g]]
                thresh = np.mean(all_thresholds) if all_thresholds else 0.5

            include_a = (1 - p) <= thresh
            include_b = p <= thresh

            pred_set = []
            if include_a:
                pred_set.append(1)
            if include_b:
                pred_set.append(0)
            if not pred_set:
                pred_set = [1 if p > 0.5 else 0]

            results.append({
                "prediction_set": pred_set,
                "set_size": len(pred_set),
                "prob_team_a": p,
                "confidence_level": confidence,
                "round_group": rg,
            })

        return results


def _evaluate_by_segment(y: np.ndarray, probs: np.ndarray,
                         df: pl.DataFrame, label: str):
    """Print accuracy breakdown by round and seed_diff_bucket."""
    preds_binary = (probs > 0.5).astype(int)

    # By round group
    if "round_group" in df.columns:
        rgs = df["round_group"].to_numpy()
        rg_names = {0: "First Four", 1: "Early (R64+R32)", 2: "Mid (S16+E8)", 3: "Late (F4+Champ)"}
        print(f"\n  {label} by round:")
        for rg in sorted(set(rgs)):
            mask = rgs == rg
            if mask.sum() == 0:
                continue
            acc = accuracy_score(y[mask], preds_binary[mask])
            n = mask.sum()
            print(f"    {rg_names.get(rg, f'RG{rg}')}: {acc:.1%} ({n} games)")

    # By seed diff bucket
    if "seed_diff_bucket" in df.columns:
        sdb = df["seed_diff_bucket"].to_numpy()
        sdb_names = {0: "Close (|diff|<=3)", 1: "Moderate (4-7)", 2: "Mismatch (8+)"}
        print(f"  {label} by matchup type:")
        for sb in sorted(set(sdb)):
            mask = sdb == sb
            if mask.sum() == 0:
                continue
            acc = accuracy_score(y[mask], preds_binary[mask])
            n = mask.sum()
            print(f"    {sdb_names.get(sb, f'SB{sb}')}: {acc:.1%} ({n} games)")


def run():
    """Full specialist stacking + conformal pipeline."""
    print("[Stacking] Running specialist stacking + conformal prediction...")

    # Load data
    features_path = config.FEATURES_DIR / "matchup_features.parquet"
    if not features_path.exists():
        print("  [ERROR] matchup_features.parquet not found")
        return

    df = pl.read_parquet(features_path)
    df = df.filter(pl.col("winner").is_not_null())
    print(f"  Loaded {len(df)} matchups")

    # Load selected features
    selected = _load_selected_features()
    if not selected:
        print("  [ERROR] No selected features found")
        return

    curated = [c for c in _get_curated_features() if c in df.columns]
    close_feats = [c for c in _get_close_game_features() if c in df.columns]
    late_feats = [c for c in _get_late_round_features() if c in df.columns]

    years = sorted(df["season"].unique().to_list())
    print(f"  Years: {years}")
    print(f"  Selected features (Model A): {len(selected)}")
    print(f"  Close-game features (Model B): {len(close_feats)}")
    print(f"  Late-round features (Model C): {len(late_feats)}")
    print(f"  Curated features (Model D): {len(curated)}")

    # ── Step 1: Generate out-of-fold specialist predictions ──────────────
    print("\n  Step 1: Training specialist models (leave-one-year-out)...")
    specialists = SpecialistModels(selected, curated, close_feats, late_feats)
    oof = specialists.generate_oof_predictions(df, years)

    model_names = oof["model_names"]
    y = oof["y"]
    print(f"\n  Specialist models: {model_names}")

    # Report per-model accuracy
    for i, name in enumerate(model_names):
        preds_i = oof["oof_preds"][:, i]
        valid = ~np.isnan(preds_i)
        acc = accuracy_score(y[valid], (preds_i[valid] > 0.5).astype(int))
        ll = log_loss(y[valid], preds_i[valid])
        mcc = matthews_corrcoef(y[valid], (preds_i[valid] > 0.5).astype(int))
        print(f"    {name}: acc={acc:.1%}, logloss={ll:.3f}, MCC={mcc:.3f}")

    # Correlation between models
    print("\n  Model correlation matrix:")
    for i, n1 in enumerate(model_names):
        corrs = []
        for j, n2 in enumerate(model_names):
            p1 = oof["oof_preds"][:, i]
            p2 = oof["oof_preds"][:, j]
            valid = ~(np.isnan(p1) | np.isnan(p2))
            r = np.corrcoef(p1[valid], p2[valid])[0, 1]
            corrs.append(f"{r:.3f}")
        print(f"    {n1:>15}: {' '.join(corrs)}")

    # ── Step 2: Train stacking meta-learner ──────────────────────────────
    print("\n  Step 2: Training stacking meta-learner...")
    meta = StackingMetaLearner()
    X_meta = oof["oof_meta_features"]
    meta.fit(X_meta, y)

    meta_probs = meta.predict_proba(X_meta)
    meta_acc = accuracy_score(y, (meta_probs > 0.5).astype(int))
    meta_ll = log_loss(y, meta_probs)
    meta_mcc = matthews_corrcoef(y, (meta_probs > 0.5).astype(int))
    baseline_acc = accuracy_score(y, np.ones(len(y)))

    print(f"    Meta-learner OOF: acc={meta_acc:.1%}, logloss={meta_ll:.3f}, MCC={meta_mcc:.3f}")
    print(f"    Baseline (higher seed): acc={baseline_acc:.1%}")
    print(f"    Improvement: {meta_acc - baseline_acc:+.1%}")

    # Compare to simple average baseline
    simple_avg = np.nanmean(oof["oof_preds"], axis=1)
    simple_acc = accuracy_score(y, (simple_avg > 0.5).astype(int))
    simple_ll = log_loss(y, np.clip(simple_avg, 0.01, 0.99))
    print(f"    Simple average: acc={simple_acc:.1%}, logloss={simple_ll:.3f}")
    print(f"    Meta vs simple: {meta_acc - simple_acc:+.1%}")

    # Per-year breakdown
    seasons = df["season"].to_numpy()
    print(f"\n  Per-year stacking results:")
    print(f"  {'Year':<6} {'Meta':>8} {'Simple':>8} {'Baseline':>10} {'LogLoss':>10} {'MCC':>8}")
    print(f"  {'-'*54}")
    for year in years:
        mask = seasons == year
        y_yr = y[mask]
        p_yr = meta_probs[mask]
        s_yr = simple_avg[mask]
        acc_yr = accuracy_score(y_yr, (p_yr > 0.5).astype(int))
        sacc_yr = accuracy_score(y_yr, (s_yr > 0.5).astype(int))
        ll_yr = log_loss(y_yr, p_yr)
        mcc_yr = matthews_corrcoef(y_yr, (p_yr > 0.5).astype(int))
        base_yr = accuracy_score(y_yr, np.ones(len(y_yr)))
        print(f"  {year:<6} {acc_yr:>8.1%} {sacc_yr:>8.1%} {base_yr:>10.1%} {ll_yr:>10.3f} {mcc_yr:>8.3f}")

    # Segment breakdown
    _evaluate_by_segment(y, meta_probs, df, "Meta-learner accuracy")

    # ── Step 3: Conformal prediction ─────────────────────────────────────
    print("\n  Step 3: Calibrating conformal prediction...")
    round_groups = df["round_group"].to_numpy() if "round_group" in df.columns else np.zeros(len(df))

    conformal = ConformalPredictor()
    conformal.calibrate(meta_probs, y, round_groups)

    print(f"\n  Prediction set analysis:")
    for conf in [0.80, 0.85, 0.90, 0.95]:
        sets = conformal.predict_sets(meta_probs, round_groups, confidence=conf)
        sizes = [s["set_size"] for s in sets]
        n_singleton = sum(1 for s in sizes if s == 1)
        n_both = sum(1 for s in sizes if s == 2)
        covered = sum(1 for j, s in enumerate(sets) if int(y[j]) in s["prediction_set"])
        coverage = covered / len(sets)
        print(f"    {conf:.0%} confidence: {n_singleton} confident ({n_singleton/len(sets):.0%}), "
              f"{n_both} uncertain ({n_both/len(sets):.0%}), "
              f"empirical coverage={coverage:.1%}")

    # ── Step 4: Train final models on all data ───────────────────────────
    print("\n  Step 4: Training final models on all data...")
    matrices = specialists._prepare_feature_matrices(df)
    all_mask = np.ones(len(df), dtype=bool)

    final_models = specialists._train_specialists(df, matrices, all_mask, y)

    models_dir = config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "specialist_models.pkl", "wb") as f:
        pickle.dump(final_models, f)
    with open(models_dir / "meta_learner.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(models_dir / "conformal.pkl", "wb") as f:
        pickle.dump(conformal, f)

    # Feature importance from Model A (XGBoost)
    if "model_a_xgb" in final_models:
        avail_sel = [c for c in selected if c in df.columns]
        importances = dict(zip(avail_sel, final_models["model_a_xgb"].feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:15]
        print("\n  XGBoost (Model A) top 15 features:")
        for feat, imp in top_features:
            print(f"    {imp:.4f}  {feat}")

    # Save metadata
    meta_info = {
        "model_names": model_names,
        "selected_features": [c for c in selected if c in df.columns],
        "curated_features": curated,
        "close_features": close_feats,
        "late_features": late_feats,
        "context_features": oof["context_cols"],
        "meta_accuracy": float(meta_acc),
        "meta_logloss": float(meta_ll),
        "meta_mcc": float(meta_mcc),
        "baseline_accuracy": float(baseline_acc),
        "simple_avg_accuracy": float(simple_acc),
        "n_training_games": len(df),
        "years": years,
    }
    with open(models_dir / "stacking_metadata.json", "w") as f:
        json.dump(meta_info, f, indent=2, default=str)

    print(f"\n  -> All models saved to {models_dir}")
    print("[Stacking] Done.")


if __name__ == "__main__":
    run()
