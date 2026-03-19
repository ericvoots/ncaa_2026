"""Feature selection: Pearson dedup + MI + MCC relevance scoring."""
import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import matthews_corrcoef

import config


def _compute_pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix, handling NaN with pairwise complete."""
    # nan_to_num first for stability
    X_clean = np.nan_to_num(X, nan=0.0)
    # numpy corrcoef
    return np.corrcoef(X_clean, rowvar=False)


def _compute_mi_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute mutual information between each feature and binary target."""
    X_clean = np.nan_to_num(X, nan=0.0)
    mi = mutual_info_classif(X_clean, y, discrete_features=False, random_state=42, n_neighbors=5)
    return mi


def _compute_mcc_scores(X: np.ndarray, y: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Compute MCC between discretized features and binary target.

    Uses quartile binning, then computes MCC for each bin boundary.
    Takes the max MCC across bin boundaries for each feature.
    """
    X_clean = np.nan_to_num(X, nan=0.0)
    n_features = X_clean.shape[1]
    mcc_scores = np.zeros(n_features)

    for j in range(n_features):
        col = X_clean[:, j]
        best_mcc = 0.0

        # Try multiple quantile thresholds
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # e.g., [0.25, 0.5, 0.75] for quartiles
        for q in quantiles:
            threshold = np.quantile(col, q)
            col_binary = (col > threshold).astype(int)

            # MCC needs both classes present
            if len(np.unique(col_binary)) < 2:
                continue

            mcc = abs(matthews_corrcoef(y, col_binary))
            if mcc > best_mcc:
                best_mcc = mcc

        mcc_scores[j] = best_mcc

    return mcc_scores


def _deduplicate_correlated(
    feature_names: list[str],
    corr_matrix: np.ndarray,
    relevance_ranks: np.ndarray,
    threshold: float = 0.90,
) -> list[int]:
    """Remove one feature from each highly-correlated pair.

    Keeps the feature with the higher relevance rank (lower rank number = more relevant).
    Returns indices of features to KEEP.
    """
    n = len(feature_names)
    to_drop = set()

    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                # Drop the one with worse (higher) relevance rank
                if relevance_ranks[i] <= relevance_ranks[j]:
                    to_drop.add(j)
                else:
                    to_drop.add(i)
                    break  # i is dropped, move on

    keep = [i for i in range(n) if i not in to_drop]
    return keep


def select_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "winner",
    pearson_threshold: float = 0.90,
    drop_bottom_pct: float = 0.50,
    mcc_bins: int = 4,
    verbose: bool = True,
) -> list[str]:
    """Run full feature selection pipeline.

    1. Compute MI and MCC relevance scores for each feature vs target
    2. Rank features by combined MI + MCC rank
    3. Drop correlated pairs (Pearson > threshold), keeping higher-ranked feature
    4. Drop bottom % by combined relevance rank

    Returns list of selected feature names.
    """
    # Prepare data
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df.select(available_cols).to_numpy()
    y = df[target_col].to_numpy().astype(float)

    X = np.nan_to_num(X, nan=0.0)
    n_features = len(available_cols)

    if verbose:
        print(f"  Starting feature selection with {n_features} features...")

    # ── Step 1: Compute relevance scores ───────────────────────────────────
    if verbose:
        print("  Computing Mutual Information scores...")
    mi_scores = _compute_mi_scores(X, y)

    if verbose:
        print(f"  Computing MCC scores (n_bins={mcc_bins})...")
    mcc_scores = _compute_mcc_scores(X, y, n_bins=mcc_bins)

    # Rank each (lower rank = more relevant)
    mi_ranks = np.argsort(np.argsort(-mi_scores))  # descending
    mcc_ranks = np.argsort(np.argsort(-mcc_scores))  # descending

    # Combined rank: average of MI rank and MCC rank
    combined_ranks = (mi_ranks + mcc_ranks) / 2.0

    if verbose:
        # Show top 20 by combined rank
        top_idx = np.argsort(combined_ranks)[:20]
        print(f"\n  Top 20 features by combined MI+MCC rank:")
        print(f"  {'Feature':<55} {'MI':>8} {'MCC':>8} {'Rank':>8}")
        print(f"  {'-'*79}")
        for idx in top_idx:
            print(f"  {available_cols[idx]:<55} {mi_scores[idx]:>8.4f} "
                  f"{mcc_scores[idx]:>8.4f} {combined_ranks[idx]:>8.1f}")

    # ── Step 2: Pearson correlation deduplication ──────────────────────────
    if verbose:
        print(f"\n  Computing Pearson correlation matrix...")
    corr_matrix = _compute_pearson_matrix(X)

    if verbose:
        # Count pairs above threshold
        n_corr_pairs = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > pearson_threshold:
                    n_corr_pairs += 1
        print(f"  Found {n_corr_pairs} feature pairs with |r| > {pearson_threshold}")

    keep_idx = _deduplicate_correlated(
        available_cols, corr_matrix, combined_ranks, threshold=pearson_threshold
    )

    if verbose:
        n_dropped_corr = n_features - len(keep_idx)
        print(f"  Dropped {n_dropped_corr} correlated features, {len(keep_idx)} remaining")

    # ── Step 3: Drop bottom % by relevance ─────────────────────────────────
    # Among surviving features, rank by combined relevance and drop bottom half
    surviving_names = [available_cols[i] for i in keep_idx]
    surviving_ranks = combined_ranks[keep_idx]

    n_to_keep = max(10, int(len(surviving_names) * (1 - drop_bottom_pct)))
    top_surviving_idx = np.argsort(surviving_ranks)[:n_to_keep]
    selected = [surviving_names[i] for i in top_surviving_idx]

    if verbose:
        print(f"  After dropping bottom {drop_bottom_pct:.0%}: {len(selected)} features selected")
        print(f"\n  Final feature count: {len(selected)}")

        # Sanity check: ensure key features survived
        key_features = ["seed_diff", "efficiency_gap", "ball_control_index", "shooting_index"]
        for kf in key_features:
            status = "KEPT" if kf in selected else "DROPPED"
            print(f"    {kf}: {status}")

    return selected


def run():
    """Run feature selection and save results."""
    print("[FeatureSelection] Running feature selection pipeline...")

    features_path = config.FEATURES_DIR / "matchup_features.parquet"
    if not features_path.exists():
        print("  [ERROR] matchup_features.parquet not found")
        return

    df = pl.read_parquet(features_path)
    df = df.filter(pl.col("winner").is_not_null())

    # Get all engineered feature columns
    exclude = {
        "team_a_id", "team_b_id", "team_a_name", "team_b_name",
        "team_a_seed", "team_b_seed", "team_a_score", "team_b_score",
        "winner", "bracket_round", "bracket_region", "season",
        "game_id", "date",
    }
    feature_cols = [c for c in df.columns if c not in exclude
                    and not c.startswith("a_") and not c.startswith("b_")]

    selected = select_features(
        df, feature_cols,
        pearson_threshold=0.90,
        drop_bottom_pct=0.50,
        mcc_bins=4,
        verbose=True,
    )

    # Save selected feature list
    import json
    out_path = config.FEATURES_DIR / "selected_features.json"
    with open(out_path, "w") as f:
        json.dump({"selected_features": selected, "n_original": len(feature_cols)}, f, indent=2)

    print(f"\n  -> Saved {len(selected)} selected features to {out_path}")
    return selected


if __name__ == "__main__":
    run()
