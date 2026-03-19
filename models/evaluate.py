"""Evaluate model performance with detailed metrics."""
import json

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

import config


def run():
    """Print detailed evaluation from CV results."""
    print("[Evaluate] Model evaluation report...")

    meta_path = config.MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        print("  [ERROR] No metadata found. Run train first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    cv_results = meta.get("cv_results", [])
    if not cv_results:
        print("  [WARN] No CV results found")
        return

    print(f"\n  Leave-One-Year-Out Cross-Validation Results:")
    print(f"  {'Year':<6} {'Games':<6} {'Accuracy':<10} {'Baseline':<10} {'LogLoss':<10}")
    print(f"  {'-'*42}")

    for r in cv_results:
        print(f"  {r['year']:<6} {r['n_games']:<6} {r['accuracy']:<10.1%} "
              f"{r['baseline_acc']:<10.1%} {r['log_loss']:<10.3f}")

    avg_acc = np.mean([r["accuracy"] for r in cv_results])
    avg_ll = np.mean([r["log_loss"] for r in cv_results])
    avg_base = np.mean([r["baseline_acc"] for r in cv_results])

    print(f"  {'-'*42}")
    print(f"  {'AVG':<6} {'':6} {avg_acc:<10.1%} {avg_base:<10.1%} {avg_ll:<10.3f}")
    print(f"\n  Improvement over baseline: {avg_acc - avg_base:+.1%}")
    print(f"  Models used: {meta.get('models', [])}")
    print(f"  Total features: {len(meta.get('all_feature_cols', []))}")
    print(f"  Curated features: {len(meta.get('curated_cols', []))}")


if __name__ == "__main__":
    run()
