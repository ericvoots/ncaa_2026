"""Build matchup-level features from processed data."""
import re

import polars as pl
import numpy as np

import config


# ── Round normalization ────────────────────────────────────────────────────
# Maps messy bracket_round strings to clean round numbers
# First Four=0, R64=1, R32=2, Sweet16=3, Elite8=4, FinalFour=5, Championship=6
def _normalize_round(raw: str) -> int:
    """Convert messy round strings to integer 0-6."""
    s = raw.lower().replace("&#174;", "").replace("®", "").strip()

    if "first four" in s:
        return 0
    if "first round" in s or "second round" in s:
        # NCAA renamed: pre-2011 "First Round"=First Four, "Second Round"=R64
        # post-2011: "First Round"=R64, "Second Round"=R32
        # In our data (2015+), "Second Round" = R64 and "First Round" is sometimes R64 too
        # The numeric entries help disambiguate
        if "second" in s:
            return 1  # R64
        return 1  # R64
    if "third round" in s:
        return 2  # R32
    if "sweet 16" in s or "sweet sixteen" in s:
        return 3
    if "elite 8" in s or "elite eight" in s:
        return 4
    if "final four" in s:
        return 5
    if "championship" in s:
        return 6

    # Numeric fallback (some entries are just "1", "2", etc.)
    try:
        n = int(s)
        # NCAA API numeric: 1=R64, 2=R32, 3=Sweet16, 4=Elite8, 5=F4, 6=Championship
        return n
    except ValueError:
        return -1  # unknown


def _safe_diff(a, b):
    """Compute a - b, returning None if either is None."""
    if a is None or b is None:
        return None
    return a - b


def _safe_pct_diff(a, b):
    """Compute (a - b) / ((a + b) / 2), safe for zeros/nulls."""
    if a is None or b is None:
        return None
    avg = (a + b) / 2
    if avg == 0:
        return 0.0
    return (a - b) / avg


def run():
    """Build matchup features from tournament matchups + team stats."""
    print("[Features] Building matchup features...")

    matchups_path = config.PROCESSED_DIR / "tournament_matchups.parquet"
    stats_path = config.PROCESSED_DIR / "team_season_stats.parquet"
    conf_path = config.PROCESSED_DIR / "conference_strength.parquet"

    if not matchups_path.exists():
        print("  [ERROR] tournament_matchups.parquet not found")
        return
    if not stats_path.exists():
        print("  [ERROR] team_season_stats.parquet not found")
        return

    matchups = pl.read_parquet(matchups_path)
    stats = pl.read_parquet(stats_path)

    print(f"  Matchups: {len(matchups)} rows")
    print(f"  Stats: {len(stats)} rows, {len(stats.columns)} cols")

    # Get all stat columns (exclude metadata)
    meta_cols = {"team_id", "team_name", "season"}
    stat_cols = [c for c in stats.columns if c not in meta_cols]

    # Join team_a stats
    stats_a = stats.rename({c: f"a_{c}" for c in stat_cols})
    stats_a = stats_a.rename({"team_id": "team_a_id", "season": "season"})
    if "team_name" in stats_a.columns:
        stats_a = stats_a.drop("team_name")

    matchups = matchups.join(
        stats_a,
        on=["team_a_id", "season"],
        how="left",
    )

    # Join team_b stats
    stats_b = stats.rename({c: f"b_{c}" for c in stat_cols})
    stats_b = stats_b.rename({"team_id": "team_b_id", "season": "season"})
    if "team_name" in stats_b.columns:
        stats_b = stats_b.drop("team_name")

    matchups = matchups.join(
        stats_b,
        on=["team_b_id", "season"],
        how="left",
    )

    # ── Round features ─────────────────────────────────────────────────────
    round_numbers = [_normalize_round(r) for r in matchups["bracket_round"].to_list()]
    matchups = matchups.with_columns(pl.Series("round_number", round_numbers))

    # Round grouping for pooling (thin later rounds together)
    # 0=First Four, 1=R64+R32 early, 2=Sweet16+Elite8 mid, 3=F4+Championship late
    round_groups = []
    for r in round_numbers:
        if r <= 0:
            round_groups.append(0)  # First Four
        elif r <= 2:
            round_groups.append(1)  # Early rounds (R64 + R32)
        elif r <= 4:
            round_groups.append(2)  # Mid rounds (Sweet 16 + Elite 8)
        else:
            round_groups.append(3)  # Late rounds (Final Four + Championship)
    matchups = matchups.with_columns(pl.Series("round_group", round_groups))

    # ── Seed features ──────────────────────────────────────────────────────
    matchups = matchups.with_columns([
        (pl.col("team_a_seed") - pl.col("team_b_seed")).alias("seed_diff"),
        (pl.col("team_a_seed") + pl.col("team_b_seed")).alias("seed_sum"),
        (pl.col("team_a_seed") * pl.col("team_b_seed")).alias("seed_product"),
        # Percentage diff: recovers signal in later rounds where raw seed_diff fails
        ((pl.col("team_a_seed") - pl.col("team_b_seed")) /
         ((pl.col("team_a_seed") + pl.col("team_b_seed")) / 2).clip(lower_bound=0.001))
        .alias("seed_pct_diff"),
        # Log ratio: strong signal across all rounds
        (pl.col("team_a_seed").cast(pl.Float64).log() -
         pl.col("team_b_seed").cast(pl.Float64).log())
        .alias("log_seed_ratio"),
    ])

    # Seed diff bucketed (for stratification)
    matchups = matchups.with_columns(
        pl.when(pl.col("seed_diff").abs() <= 3).then(pl.lit(0))  # close matchup
        .when(pl.col("seed_diff").abs() <= 7).then(pl.lit(1))    # moderate mismatch
        .otherwise(pl.lit(2))                                      # heavy favorite
        .alias("seed_diff_bucket")
    )

    # ── Stat differentials ─────────────────────────────────────────────────
    # For each stat column, compute diff and pct_diff between team_a and team_b
    diff_exprs = []
    for col in stat_cols:
        a_col = f"a_{col}"
        b_col = f"b_{col}"
        if a_col in matchups.columns and b_col in matchups.columns:
            # Raw difference
            diff_exprs.append(
                (pl.col(a_col) - pl.col(b_col)).alias(f"diff_{col}")
            )
            # Percentage difference
            diff_exprs.append(
                ((pl.col(a_col) - pl.col(b_col)) /
                 ((pl.col(a_col) + pl.col(b_col)) / 2).clip(lower_bound=0.001))
                .alias(f"pctdiff_{col}")
            )

    if diff_exprs:
        matchups = matchups.with_columns(diff_exprs)

    # ── Away performance gap features ──────────────────────────────────────
    # For stats that have both _season and _away variants, compute the gap
    away_gap_exprs = []
    for col in stat_cols:
        if col.endswith("_season"):
            base = col.replace("_season", "")
            away_col = f"{base}_away"
            if away_col in stat_cols:
                a_season = f"a_{col}"
                a_away = f"a_{away_col}"
                b_season = f"b_{col}"
                b_away = f"b_{away_col}"

                if all(c in matchups.columns for c in [a_season, a_away, b_season, b_away]):
                    # How much worse each team is on the road
                    away_gap_exprs.append(
                        (pl.col(a_season) - pl.col(a_away)).alias(f"a_awaygap_{base}")
                    )
                    away_gap_exprs.append(
                        (pl.col(b_season) - pl.col(b_away)).alias(f"b_awaygap_{base}")
                    )
                    # Differential of away gaps
                    away_gap_exprs.append(
                        ((pl.col(a_season) - pl.col(a_away)) -
                         (pl.col(b_season) - pl.col(b_away))).alias(f"diff_awaygap_{base}")
                    )

    if away_gap_exprs:
        matchups = matchups.with_columns(away_gap_exprs)

    # ── Composite features ─────────────────────────────────────────────────
    composite_exprs = []

    # Efficiency gap: (off_eff_a - def_eff_a) - (off_eff_b - def_eff_b)
    oe_a = "a_offensive-efficiency_season"
    de_a = "a_defensive-efficiency_season"
    oe_b = "b_offensive-efficiency_season"
    de_b = "b_defensive-efficiency_season"
    if all(c in matchups.columns for c in [oe_a, de_a, oe_b, de_b]):
        composite_exprs.append(
            ((pl.col(oe_a) - pl.col(de_a)) - (pl.col(oe_b) - pl.col(de_b)))
            .alias("efficiency_gap")
        )

    # Ball control index: assist-to-TO ratio diff + steal diff - turnover diff
    ato_a = "a_assist--per--turnover-ratio_season"
    ato_b = "b_assist--per--turnover-ratio_season"
    stl_a = "a_steals-per-game_season"
    stl_b = "b_steals-per-game_season"
    tov_a = "a_turnovers-per-game_season"
    tov_b = "b_turnovers-per-game_season"
    if all(c in matchups.columns for c in [ato_a, ato_b]):
        ball_ctrl = (pl.col(ato_a) - pl.col(ato_b))
        if stl_a in matchups.columns and stl_b in matchups.columns:
            ball_ctrl = ball_ctrl + (pl.col(stl_a) - pl.col(stl_b)) * 0.5
        if tov_a in matchups.columns and tov_b in matchups.columns:
            ball_ctrl = ball_ctrl - (pl.col(tov_a) - pl.col(tov_b)) * 0.3
        composite_exprs.append(ball_ctrl.alias("ball_control_index"))

    # Shooting index: eFG% diff + 3P% diff + FT rate diff
    efg_a = "a_effective-field-goal-pct_season"
    efg_b = "b_effective-field-goal-pct_season"
    tp_a = "a_three-point-pct_season"
    tp_b = "b_three-point-pct_season"
    ftr_a = "a_free-throw-rate_season"
    ftr_b = "b_free-throw-rate_season"
    if all(c in matchups.columns for c in [efg_a, efg_b]):
        shooting = (pl.col(efg_a) - pl.col(efg_b))
        if tp_a in matchups.columns and tp_b in matchups.columns:
            shooting = shooting + (pl.col(tp_a) - pl.col(tp_b)) * 0.5
        if ftr_a in matchups.columns and ftr_b in matchups.columns:
            shooting = shooting + (pl.col(ftr_a) - pl.col(ftr_b)) * 0.3
        composite_exprs.append(shooting.alias("shooting_index"))

    if composite_exprs:
        matchups = matchups.with_columns(composite_exprs)

    # ── Conference strength features ───────────────────────────────────────
    if conf_path.exists():
        print("  Adding conference features...")
        # This would require knowing each team's conference - skip if mapping unavailable
        # Can be added later once we have team-to-conference mapping
        pass

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = config.FEATURES_DIR / "matchup_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matchups.write_parquet(out_path)

    # Summary
    feature_cols = [c for c in matchups.columns if c.startswith(("diff_", "pctdiff_", "a_awaygap_", "b_awaygap_", "diff_awaygap_", "seed_", "efficiency_", "ball_control_", "shooting_"))]
    print(f"\n  -> Saved {len(matchups)} matchups with {len(matchups.columns)} total cols")
    print(f"     {len(feature_cols)} engineered features")
    print(f"     Columns sample: {matchups.columns[:10]}")

    # Quick sanity check
    if "seed_diff" in matchups.columns and "winner" in matchups.columns:
        corr = matchups.select([
            pl.corr("seed_diff", "winner").alias("seed_diff_winner_corr")
        ])
        print(f"     seed_diff ↔ winner correlation: {corr[0, 0]:.3f}")


if __name__ == "__main__":
    run()
