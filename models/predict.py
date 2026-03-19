"""Generate predictions for 2026 tournament using stacking + conformal."""
import json
import pickle

import numpy as np
import polars as pl

import config


def _load_stacking_artifacts():
    """Load all stacking model artifacts."""
    models_dir = config.MODELS_DIR
    meta_path = models_dir / "stacking_metadata.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    with open(models_dir / "specialist_models.pkl", "rb") as f:
        specialist_models = pickle.load(f)
    with open(models_dir / "meta_learner.pkl", "rb") as f:
        meta_learner = pickle.load(f)
    with open(models_dir / "conformal.pkl", "rb") as f:
        conformal = pickle.load(f)

    return {
        "meta": meta,
        "specialist_models": specialist_models,
        "meta_learner": meta_learner,
        "conformal": conformal,
    }


def _predict_with_stacking(df, artifacts):
    """Generate predictions for a set of games using stacking pipeline."""
    meta = artifacts["meta"]
    models = artifacts["specialist_models"]
    meta_learner = artifacts["meta_learner"]

    # Prepare feature matrices
    sel_cols = [c for c in meta["selected_features"] if c in df.columns]
    cur_cols = [c for c in meta["curated_features"] if c in df.columns]
    close_cols = [c for c in meta["close_features"] if c in df.columns]
    late_cols = [c for c in meta["late_features"] if c in df.columns]
    ctx_cols = [c for c in meta["context_features"] if c in df.columns]

    X_sel = np.nan_to_num(df.select(sel_cols).to_numpy(), nan=0.0)
    X_cur = np.nan_to_num(df.select(cur_cols).to_numpy(), nan=0.0)
    X_close = np.nan_to_num(df.select(close_cols).to_numpy(), nan=0.0)
    X_late = np.nan_to_num(df.select(late_cols).to_numpy(), nan=0.0)
    X_ctx = np.nan_to_num(df.select(ctx_cols).to_numpy(), nan=0.0)

    # Get base model predictions
    base_preds = []
    model_names = []

    if "model_a_xgb" in models:
        base_preds.append(models["model_a_xgb"].predict_proba(X_sel)[:, 1])
        model_names.append("xgb_general")

    if "model_b_close" in models:
        base_preds.append(models["model_b_close"].predict_proba(X_close)[:, 1])
        model_names.append("lgb_close")

    if "model_c_late" in models:
        X_late_scaled = models["model_c_scaler"].transform(X_late)
        base_preds.append(models["model_c_late"].predict_proba(X_late_scaled)[:, 1])
        model_names.append("lr_late")

    if "model_d_rf" in models:
        base_preds.append(models["model_d_rf"].predict_proba(X_cur)[:, 1])
        model_names.append("rf_diverse")

    base_preds = np.column_stack(base_preds)

    # Build meta features
    mean_pred = np.mean(base_preds, axis=1, keepdims=True)
    std_pred = np.std(base_preds, axis=1, keepdims=True)
    spread = np.max(base_preds, axis=1, keepdims=True) - np.min(base_preds, axis=1, keepdims=True)
    max_conf = np.max(np.abs(base_preds - 0.5), axis=1, keepdims=True)

    parts = [base_preds, mean_pred, std_pred, spread, max_conf, X_ctx]
    if X_ctx.shape[1] >= 7:
        parts.append(X_ctx[:, 4:5] * X_ctx[:, 6:7])  # round * seed_bucket
        parts.append(X_ctx[:, 4:5] * std_pred)         # round * disagreement

    X_meta = np.hstack(parts)

    # Meta-learner final probabilities
    probs = meta_learner.predict_proba(X_meta)

    # Conformal prediction sets
    conformal = artifacts["conformal"]
    round_groups = df["round_group"].to_numpy() if "round_group" in df.columns else np.zeros(len(df))

    conf_sets_90 = conformal.predict_sets(probs, round_groups, confidence=0.90)
    conf_sets_80 = conformal.predict_sets(probs, round_groups, confidence=0.80)

    return {
        "probs": probs,
        "base_preds": base_preds,
        "model_names": model_names,
        "conf_sets_90": conf_sets_90,
        "conf_sets_80": conf_sets_80,
    }


# Round display names
ROUND_NAMES = {
    0: "First Four",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


def run():
    """Generate 2026 bracket predictions."""
    print("[Predict] Generating 2026 tournament predictions...")

    artifacts = _load_stacking_artifacts()
    if not artifacts:
        print("  [ERROR] No stacking models found. Run train first.")
        return

    # Load features
    features_path = config.FEATURES_DIR / "matchup_features.parquet"
    if not features_path.exists():
        print("  [ERROR] matchup_features.parquet not found")
        return

    df = pl.read_parquet(features_path)

    # Split: train on 2015-2025, predict 2026
    predict_df = df.filter(pl.col("season") == 2026)
    if len(predict_df) == 0:
        print("  [ERROR] No 2026 data found")
        return

    print(f"  2026 games to predict: {len(predict_df)}")

    # Generate predictions
    results = _predict_with_stacking(predict_df, artifacts)
    probs = results["probs"]

    # Build output
    print(f"\n{'='*80}")
    print(f"  2026 MARCH MADNESS BRACKET PREDICTIONS")
    print(f"  Model: Specialist Stacking (4 models) + Conformal")
    print(f"  Trained on: {artifacts['meta']['n_training_games']} historical games")
    print(f"{'='*80}")

    # Get round numbers and sort games
    round_nums = predict_df["round_number"].to_list() if "round_number" in predict_df.columns else [0] * len(predict_df)

    correct = 0
    total = 0
    correct_by_round = {}
    total_by_round = {}
    upset_correct = 0
    upset_total = 0

    for rnd in sorted(set(round_nums)):
        rnd_mask = [i for i, r in enumerate(round_nums) if r == rnd]
        if not rnd_mask:
            continue

        rnd_name = ROUND_NAMES.get(rnd, f"Round {rnd}")
        print(f"\n  --- {rnd_name} ({len(rnd_mask)} games) ---")

        for idx in rnd_mask:
            row = predict_df.row(idx, named=True)
            prob = probs[idx]
            conf_90 = results["conf_sets_90"][idx]
            conf_80 = results["conf_sets_80"][idx]

            team_a = row["team_a_name"]
            team_b = row["team_b_name"]
            seed_a = row["team_a_seed"]
            seed_b = row["team_b_seed"]
            winner = row.get("winner")

            # Predicted winner
            if prob > 0.5:
                pred_name = team_a
                pred_seed = seed_a
                pred_prob = prob
            else:
                pred_name = team_b
                pred_seed = seed_b
                pred_prob = 1 - prob

            # Is this an upset prediction? (lower seed = higher number wins)
            is_upset_pred = (prob > 0.5 and seed_a > seed_b) or (prob <= 0.5 and seed_b > seed_a)

            # Confidence indicator
            if conf_90["set_size"] == 1:
                conf_str = "HIGH"
            elif conf_80["set_size"] == 2:
                conf_str = "LOW "
            else:
                conf_str = "MED "

            # Base model agreement
            base = results["base_preds"][idx]
            agree = sum(1 for b in base if (b > 0.5) == (prob > 0.5))
            agree_str = f"{agree}/{len(base)}"

            # Check against actual result
            marker = ""
            if winner is not None:
                actual = team_a if winner == 1 else team_b
                is_correct = pred_name == actual
                marker = " OK" if is_correct else " MISS"
                total += 1
                total_by_round[rnd] = total_by_round.get(rnd, 0) + 1
                if is_correct:
                    correct += 1
                    correct_by_round[rnd] = correct_by_round.get(rnd, 0) + 1
                # Track upsets
                actual_is_upset = (winner == 1 and seed_a > seed_b) or (winner == 0 and seed_b > seed_a)
                if actual_is_upset:
                    upset_total += 1
                    if is_correct:
                        upset_correct += 1

            upset_tag = " [UPSET]" if is_upset_pred else ""

            print(f"    ({seed_a:>2}) {team_a:<22} vs ({seed_b:>2}) {team_b:<22}"
                  f" -> ({pred_seed:>2}) {pred_name:<22} {pred_prob:>5.1%}"
                  f"  [{conf_str}] agree={agree_str}{upset_tag}{marker}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  PREDICTION SUMMARY")
    print(f"{'='*80}")

    if total > 0:
        print(f"  Overall accuracy: {correct}/{total} = {correct/total:.1%}")
        print(f"  Baseline (higher seed): ~70%")
        print()

        print(f"  By round:")
        for rnd in sorted(total_by_round.keys()):
            rnd_name = ROUND_NAMES.get(rnd, f"Round {rnd}")
            rc = correct_by_round.get(rnd, 0)
            rt = total_by_round[rnd]
            print(f"    {rnd_name:<20}: {rc}/{rt} = {rc/rt:.1%}")

        if upset_total > 0:
            print(f"\n  Upset detection: {upset_correct}/{upset_total} = {upset_correct/upset_total:.1%}")

    # Confidence breakdown
    # HIGH = singleton at 90% (very confident), MED = singleton at 80% but not 90%, LOW = uncertain at 80%
    n_high = sum(1 for s in results["conf_sets_90"] if s["set_size"] == 1)
    n_low = sum(1 for s in results["conf_sets_80"] if s["set_size"] == 2)
    n_med = len(probs) - n_high - n_low
    print(f"\n  Confidence distribution:")
    print(f"    HIGH (90% conformal singleton):  {n_high} games")
    print(f"    MED  (80% singleton, 90% both):  {n_med} games")
    print(f"    LOW  (80% conformal uncertain):  {n_low} games")

    # Save predictions
    pred_data = []
    for idx in range(len(predict_df)):
        row = predict_df.row(idx, named=True)
        prob = probs[idx]
        conf_90 = results["conf_sets_90"][idx]

        pred_name = row["team_a_name"] if prob > 0.5 else row["team_b_name"]
        pred_seed = row["team_a_seed"] if prob > 0.5 else row["team_b_seed"]

        pred_data.append({
            "round_number": round_nums[idx],
            "region": row.get("bracket_region", ""),
            "team_a_name": row["team_a_name"],
            "team_a_seed": row["team_a_seed"],
            "team_b_name": row["team_b_name"],
            "team_b_seed": row["team_b_seed"],
            "prob_team_a": float(prob),
            "predicted_winner": pred_name,
            "predicted_seed": int(pred_seed),
            "confidence": float(max(prob, 1 - prob)),
            "conformal_set_size_90": conf_90["set_size"],
            "actual_winner": row["team_a_name"] if row.get("winner") == 1 else (
                row["team_b_name"] if row.get("winner") == 0 else None),
        })

    out_df = pl.DataFrame(pred_data)
    out_path = config.DATA_DIR / "predictions_2026.parquet"
    out_df.write_parquet(out_path)

    # Also save as JSON for easy viewing
    json_path = config.DATA_DIR / "predictions_2026.json"
    with open(json_path, "w") as f:
        json.dump(pred_data, f, indent=2, default=str)

    print(f"\n  -> Predictions saved to {out_path}")
    print(f"  -> JSON saved to {json_path}")
    print("[Predict] Done.")


if __name__ == "__main__":
    run()
