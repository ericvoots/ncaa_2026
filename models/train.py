"""Train ensemble model for tournament predictions."""
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

import config

# Try importing gradient boosting libraries
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


def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Get all feature columns (exclude metadata and target)."""
    exclude = {
        "team_a_id", "team_b_id", "team_a_name", "team_b_name",
        "team_a_seed", "team_b_seed", "team_a_score", "team_b_score",
        "winner", "bracket_round", "bracket_region", "season",
        "game_id", "date",
    }
    # Also exclude raw per-team stats (a_* and b_*) for the main model
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("a_") or c.startswith("b_"):
            continue
        feature_cols.append(c)
    return feature_cols


def _get_curated_features() -> list[str]:
    """Curated feature set for logistic regression."""
    return [
        "seed_diff",
        "seed_sum",
        "efficiency_gap",
        "ball_control_index",
        "shooting_index",
        "diff_offensive-efficiency_season",
        "diff_defensive-efficiency_season",
        "diff_assist--per--turnover-ratio_season",
        "diff_win-pct-all-games_season",
        "diff_average-scoring-margin_season",
        "diff_effective-field-goal-pct_season",
        "diff_turnovers-per-game_season",
        "diff_total-rebounding-percentage_season",
        "diff_assist--per--turnover-ratio_away",
        "diff_offensive-efficiency_away",
    ]


def _load_selected_features() -> list[str] | None:
    """Load feature selection results if available."""
    path = config.FEATURES_DIR / "selected_features.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("selected_features", None)


def train_loocv(df: pl.DataFrame):
    """Leave-one-year-out cross-validation + final model training."""
    # Use selected features if available, otherwise fall back to all
    selected = _load_selected_features()
    if selected:
        all_feature_cols = [c for c in selected if c in df.columns]
        print(f"  Using {len(all_feature_cols)} selected features (from feature selection)")
    else:
        all_feature_cols = _get_feature_cols(df)
        print(f"  [WARN] No selected_features.json found, using all {len(all_feature_cols)} features")

    curated_cols = [c for c in _get_curated_features() if c in df.columns]

    print(f"  Selected features: {len(all_feature_cols)}")
    print(f"  Curated features: {len(curated_cols)}")

    if len(all_feature_cols) == 0:
        print("  [ERROR] No features found!")
        return

    years = sorted(df["season"].unique().to_list())
    print(f"  Years for CV: {years}")

    # Results storage
    cv_results = []

    for test_year in years:
        train_df = df.filter(pl.col("season") != test_year)
        test_df = df.filter(pl.col("season") == test_year)

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Prepare matrices
        X_train_all = train_df.select(all_feature_cols).to_numpy()
        X_test_all = test_df.select(all_feature_cols).to_numpy()
        y_train = train_df["winner"].to_numpy().astype(float)
        y_test = test_df["winner"].to_numpy().astype(float)

        # Handle NaN - fill with 0 for now
        X_train_all = np.nan_to_num(X_train_all, nan=0.0)
        X_test_all = np.nan_to_num(X_test_all, nan=0.0)

        preds = {}

        # Model 1: XGBoost on all features
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            xgb_model.fit(X_train_all, y_train)
            preds["xgb"] = xgb_model.predict_proba(X_test_all)[:, 1]

        # Model 2: LightGBM on all features
        if HAS_LGB:
            lgb_model = lgb.LGBMClassifier(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            lgb_model.fit(X_train_all, y_train)
            preds["lgb"] = lgb_model.predict_proba(X_test_all)[:, 1]

        # Model 3: Logistic Regression on curated features
        if curated_cols:
            X_train_cur = train_df.select(curated_cols).to_numpy()
            X_test_cur = test_df.select(curated_cols).to_numpy()
            X_train_cur = np.nan_to_num(X_train_cur, nan=0.0)
            X_test_cur = np.nan_to_num(X_test_cur, nan=0.0)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_cur)
            X_test_scaled = scaler.transform(X_test_cur)

            lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            preds["lr"] = lr_model.predict_proba(X_test_scaled)[:, 1]

        if not preds:
            print(f"    {test_year}: No models trained!")
            continue

        # Ensemble: simple average
        ensemble_pred = np.mean(list(preds.values()), axis=0)

        # Metrics
        ll = log_loss(y_test, ensemble_pred)
        acc = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))

        # Baseline: always pick higher seed (team_a = higher seed, so predict 1)
        baseline_acc = accuracy_score(y_test, np.ones_like(y_test))

        cv_results.append({
            "year": test_year,
            "n_games": len(test_df),
            "log_loss": ll,
            "accuracy": acc,
            "baseline_acc": baseline_acc,
            "models": list(preds.keys()),
        })

        # Per-model metrics
        model_metrics = []
        for name, pred in preds.items():
            m_ll = log_loss(y_test, pred)
            m_acc = accuracy_score(y_test, (pred > 0.5).astype(int))
            model_metrics.append(f"{name}={m_acc:.1%}")

        print(f"    {test_year}: acc={acc:.1%} (baseline={baseline_acc:.1%}) "
              f"logloss={ll:.3f} [{', '.join(model_metrics)}]")

    # Summary
    if cv_results:
        avg_acc = np.mean([r["accuracy"] for r in cv_results])
        avg_ll = np.mean([r["log_loss"] for r in cv_results])
        avg_base = np.mean([r["baseline_acc"] for r in cv_results])
        print(f"\n  CV Summary:")
        print(f"    Avg accuracy: {avg_acc:.1%} (baseline: {avg_base:.1%})")
        print(f"    Avg log-loss: {avg_ll:.3f}")

    # ── Train final models on all data ─────────────────────────────────────
    print("\n  Training final models on all data...")
    X_all = df.select(all_feature_cols).to_numpy()
    X_all = np.nan_to_num(X_all, nan=0.0)
    y_all = df["winner"].to_numpy().astype(float)

    models_dir = config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    final_models = {}

    if HAS_XGB:
        xgb_final = xgb.XGBClassifier(
            max_depth=4, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_final.fit(X_all, y_all)
        final_models["xgb"] = xgb_final
        with open(models_dir / "xgb_model.pkl", "wb") as f:
            pickle.dump(xgb_final, f)

        # Feature importance
        importances = dict(zip(all_feature_cols, xgb_final.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:15]
        print("\n  XGBoost top 15 features:")
        for feat, imp in top_features:
            print(f"    {imp:.4f}  {feat}")

    if HAS_LGB:
        lgb_final = lgb.LGBMClassifier(
            max_depth=4, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        lgb_final.fit(X_all, y_all)
        final_models["lgb"] = lgb_final
        with open(models_dir / "lgb_model.pkl", "wb") as f:
            pickle.dump(lgb_final, f)

    if curated_cols:
        X_cur = df.select(curated_cols).to_numpy()
        X_cur = np.nan_to_num(X_cur, nan=0.0)
        scaler_final = StandardScaler()
        X_cur_scaled = scaler_final.fit_transform(X_cur)
        lr_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_final.fit(X_cur_scaled, y_all)
        final_models["lr"] = lr_final
        with open(models_dir / "lr_model.pkl", "wb") as f:
            pickle.dump(lr_final, f)
        with open(models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler_final, f)

    # Save metadata
    meta = {
        "all_feature_cols": all_feature_cols,
        "curated_cols": curated_cols,
        "models": list(final_models.keys()),
        "cv_results": cv_results,
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  -> Models saved to {models_dir}")


def run():
    """Main entry point."""
    print("[Train] Loading features and training models...")

    features_path = config.FEATURES_DIR / "matchup_features.parquet"
    if not features_path.exists():
        print("  [ERROR] matchup_features.parquet not found")
        return

    df = pl.read_parquet(features_path)
    print(f"  Loaded {len(df)} matchups, {len(df.columns)} columns")

    # Filter to rows with valid target
    df = df.filter(pl.col("winner").is_not_null())
    print(f"  After filtering: {len(df)} matchups")

    train_loocv(df)
    print("[Train] Done.")


if __name__ == "__main__":
    run()
