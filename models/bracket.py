"""2026 NCAA Tournament Bracket Predictor + HTML Visualization.

Defines the bracket, predicts every game round-by-round,
and generates an interactive HTML bracket.
"""
import json
import pickle

import numpy as np
import polars as pl

import config
from team_index import TeamIndex

# ── 2026 Bracket Definition ──────────────────────────────────────────────
# Source: ESPN bracket (Selection Sunday 2026)
# First Four results/TBDs filled in where known
BRACKET_2026 = {
    "East": {
        1: "Duke", 2: "UConn", 3: "Michigan St.", 4: "Kansas",
        5: "St. John's", 6: "Louisville", 7: "UCLA", 8: "Ohio St.",
        9: "TCU", 10: "UCF", 11: "South Fla.", 12: "Northern Iowa",
        13: "California Baptist", 14: "North Dakota St.", 15: "Furman", 16: "Siena",
    },
    "West": {
        1: "Arizona", 2: "Purdue", 3: "Gonzaga", 4: "Arkansas",
        5: "Wisconsin", 6: "BYU", 7: "Miami (FL)", 8: "Villanova",
        9: "Utah St.", 10: "Missouri", 11: "TBD_West11",  # SMU/Miami OH winner
        12: "High Point", 13: "Hawaii", 14: "Kennesaw St.", 15: "Queens", 16: "Long Island",
    },
    "South": {
        1: "Florida", 2: "Houston", 3: "Illinois", 4: "Nebraska",
        5: "Vanderbilt", 6: "North Carolina", 7: "Saint Mary's", 8: "Clemson",
        9: "Iowa", 10: "Texas A&M", 11: "VCU",
        12: "McNeese", 13: "Troy", 14: "Penn", 15: "Idaho", 16: "TBD_South16",  # Lehigh/Prairie View winner
    },
    "Midwest": {
        1: "Michigan", 2: "Iowa St.", 3: "Virginia", 4: "Alabama",
        5: "Texas Tech", 6: "Tennessee", 7: "Kentucky", 8: "Georgia",
        9: "Saint Louis", 10: "Santa Clara", 11: "TBD_MW11",  # NC State/Texas winner
        12: "Akron", 13: "Hofstra", 14: "Wright St.", 15: "Tennessee St.", 16: "Howard",
    },
}

# First Four games (play-in)
FIRST_FOUR = [
    {"seed": 16, "region": "South", "team_a": "Lehigh", "team_b": "Prairie View", "slot": "TBD_South16"},
    {"seed": 16, "region": "Midwest", "team_a": "UMBC", "team_b": "Howard", "winner": "Howard", "slot": "TBD_MW16"},
    {"seed": 11, "region": "West", "team_a": "SMU", "team_b": "Miami (OH)", "slot": "TBD_West11"},
    {"seed": 11, "region": "Midwest", "team_a": "NC State", "team_b": "Texas", "slot": "TBD_MW11"},
]

# Standard bracket matchup order for R64 (seed pairs)
R64_MATCHUPS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

# Historical upset rates by seed_diff (from 2015-2025 tournament data)
# Upset = lower seed (team_b, higher number) wins
# Key: seed_diff (team_a_seed - team_b_seed, always negative since team_a is higher seed)
HISTORICAL_UPSET_RATES = {
    -15: 0.05,   # 1 vs 16
    -13: 0.10,   # 2 vs 15
    -11: 0.12,   # 3 vs 14
    -10: 0.20,   # - (rare matchup)
    -9:  0.24,   # 4 vs 13
    -8:  0.18,   # - (rare)
    -7:  0.30,   # 5 vs 12
    -6:  0.40,   # - (rare)
    -5:  0.48,   # 6 vs 11
    -4:  0.32,   # 7 vs 10
    -3:  0.33,   # 8 vs 9
    -2:  0.27,
    -1:  0.43,
    0:   0.36,
}


def _calibrate_with_upset_prior(model_prob: float, seed_diff: int,
                                 base_preds: list[float] = None,
                                 base_weight: float = 0.35) -> float:
    """Blend model probability with historical upset base rate.

    Three signals for upsets:
    1. Historical base rate for this seed matchup
    2. Model uncertainty (close to 50%)
    3. Specialist disagreement (close-game model vs general model)

    Args:
        model_prob: P(team_a wins) from meta-learner
        seed_diff: team_a_seed - team_b_seed (negative = team_a is higher seed)
        base_preds: list of base model predictions [xgb, lgb_close, lr_late, rf]
        base_weight: how much weight to give the base rate (0-1)
    """
    # Get historical P(team_a wins) for this seed diff
    upset_rate = HISTORICAL_UPSET_RATES.get(seed_diff, 0.30)
    base_prob = 1.0 - upset_rate  # P(higher seed wins) from history

    # Dynamic weighting: model gets more weight when it's confident
    model_confidence = abs(model_prob - 0.5) * 2  # 0 = uncertain, 1 = very confident

    # Boost upset prior weight when specialists disagree
    disagree_boost = 0.0
    if base_preds and len(base_preds) >= 2:
        # Check if the close-game specialist (index 1) disagrees with general (index 0)
        specialist_upset_votes = sum(1 for p in base_preds if p < 0.5)
        if specialist_upset_votes >= 2:
            disagree_boost = 0.15  # Significant disagreement
        elif specialist_upset_votes >= 1:
            disagree_boost = 0.08

    effective_weight = base_weight * (1 - model_confidence ** 1.2) + disagree_boost

    calibrated = model_prob * (1 - effective_weight) + base_prob * effective_weight
    return calibrated


# Name mapping: ESPN names -> TeamRankings names (for stat lookup)
NAME_MAP = {
    "UConn": "Connecticut",
    "Michigan St.": "Michigan St",
    "St. John's": "St Johns",
    "South Fla.": "South Florida",
    "Ohio St.": "Ohio St",
    "Iowa St.": "Iowa St",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Utah St.": "Utah St",
    "North Dakota St.": "North Dakota St",
    "California Baptist": "California Baptist",
    "Wright St.": "Wright St",
    "Tennessee St.": "Tennessee St",
    "Saint Mary's": "Saint Marys",
    "Texas A&M": "Texas A&M",
    "NC State": "NC State",
    "High Point": "High Point",
    "Kennesaw St.": "Kennesaw St",
    "Long Island": "LIU",
    "Northern Iowa": "Northern Iowa",
}


def _resolve_team_name(name: str, team_index: TeamIndex, stats_names: set) -> str | None:
    """Resolve a bracket team name to a team_id in our stats data."""
    # Try direct match first
    mapped = NAME_MAP.get(name, name)

    # Try exact match in stats
    if mapped in stats_names:
        return mapped

    # Try team index resolution
    tid = team_index.resolve(mapped)
    if tid:
        return tid

    # Try fuzzy on original name
    tid = team_index.resolve(name)
    if tid:
        return tid

    return None


def _build_matchup_features(team_a_stats: dict, team_b_stats: dict,
                            seed_a: int, seed_b: int,
                            round_number: int, stat_cols: list[str]) -> dict:
    """Build feature dict for a single matchup (team_a = higher seed)."""
    features = {}

    # Seed features
    features["seed_diff"] = seed_a - seed_b
    features["seed_sum"] = seed_a + seed_b
    features["seed_product"] = seed_a * seed_b
    features["seed_pct_diff"] = (seed_a - seed_b) / max((seed_a + seed_b) / 2, 0.001)
    features["log_seed_ratio"] = np.log(max(seed_a, 0.5)) - np.log(max(seed_b, 0.5))
    features["seed_diff_bucket"] = 0 if abs(seed_a - seed_b) <= 3 else (1 if abs(seed_a - seed_b) <= 7 else 2)
    features["round_number"] = round_number
    features["round_group"] = 0 if round_number <= 0 else (1 if round_number <= 2 else (2 if round_number <= 4 else 3))

    # Stat differentials
    for col in stat_cols:
        a_val = team_a_stats.get(col)
        b_val = team_b_stats.get(col)
        if a_val is not None and b_val is not None:
            features[f"diff_{col}"] = a_val - b_val
            avg = (a_val + b_val) / 2
            features[f"pctdiff_{col}"] = (a_val - b_val) / max(abs(avg), 0.001)

    # Away gaps
    for col in stat_cols:
        if col.endswith("_season"):
            base = col.replace("_season", "")
            away_col = f"{base}_away"
            if away_col in stat_cols:
                a_s = team_a_stats.get(col)
                a_a = team_a_stats.get(away_col)
                b_s = team_b_stats.get(col)
                b_a = team_b_stats.get(away_col)
                if all(v is not None for v in [a_s, a_a, b_s, b_a]):
                    features[f"a_awaygap_{base}"] = a_s - a_a
                    features[f"b_awaygap_{base}"] = b_s - b_a
                    features[f"diff_awaygap_{base}"] = (a_s - a_a) - (b_s - b_a)

    # Composites
    oe_a = team_a_stats.get("offensive-efficiency_season")
    de_a = team_a_stats.get("defensive-efficiency_season")
    oe_b = team_b_stats.get("offensive-efficiency_season")
    de_b = team_b_stats.get("defensive-efficiency_season")
    if all(v is not None for v in [oe_a, de_a, oe_b, de_b]):
        features["efficiency_gap"] = (oe_a - de_a) - (oe_b - de_b)

    ato_a = team_a_stats.get("assist--per--turnover-ratio_season")
    ato_b = team_b_stats.get("assist--per--turnover-ratio_season")
    if ato_a is not None and ato_b is not None:
        bc = ato_a - ato_b
        stl_a = team_a_stats.get("steals-per-game_season")
        stl_b = team_b_stats.get("steals-per-game_season")
        if stl_a is not None and stl_b is not None:
            bc += (stl_a - stl_b) * 0.5
        tov_a = team_a_stats.get("turnovers-per-game_season")
        tov_b = team_b_stats.get("turnovers-per-game_season")
        if tov_a is not None and tov_b is not None:
            bc -= (tov_a - tov_b) * 0.3
        features["ball_control_index"] = bc

    efg_a = team_a_stats.get("effective-field-goal-pct_season")
    efg_b = team_b_stats.get("effective-field-goal-pct_season")
    if efg_a is not None and efg_b is not None:
        si = efg_a - efg_b
        tp_a = team_a_stats.get("three-point-pct_season")
        tp_b = team_b_stats.get("three-point-pct_season")
        if tp_a is not None and tp_b is not None:
            si += (tp_a - tp_b) * 0.5
        ftr_a = team_a_stats.get("free-throw-rate_season")
        ftr_b = team_b_stats.get("free-throw-rate_season")
        if ftr_a is not None and ftr_b is not None:
            si += (ftr_a - ftr_b) * 0.3
        features["shooting_index"] = si

    return features


def _predict_game(features: dict, artifacts: dict) -> dict:
    """Run stacking prediction on a single game's features."""
    meta_info = artifacts["meta"]
    models = artifacts["specialist_models"]
    meta_learner = artifacts["meta_learner"]
    conformal = artifacts["conformal"]

    # Build feature vectors for each model
    sel_cols = meta_info["selected_features"]
    cur_cols = meta_info["curated_features"]
    close_cols = meta_info["close_features"]
    late_cols = meta_info["late_features"]
    ctx_cols = meta_info["context_features"]

    def _vec(cols):
        return np.array([[features.get(c, 0.0) for c in cols]])

    X_sel = _vec(sel_cols)
    X_cur = _vec(cur_cols)
    X_close = _vec(close_cols)
    X_late = _vec(late_cols)
    X_ctx = _vec(ctx_cols)

    # Base model predictions
    base_preds = []
    if "model_a_xgb" in models:
        base_preds.append(models["model_a_xgb"].predict_proba(X_sel)[:, 1])
    if "model_b_close" in models:
        base_preds.append(models["model_b_close"].predict_proba(X_close)[:, 1])
    if "model_c_late" in models:
        X_late_s = models["model_c_scaler"].transform(X_late)
        base_preds.append(models["model_c_late"].predict_proba(X_late_s)[:, 1])
    if "model_d_rf" in models:
        base_preds.append(models["model_d_rf"].predict_proba(X_cur)[:, 1])

    base_arr = np.column_stack(base_preds)

    # Meta features
    mean_p = np.mean(base_arr, axis=1, keepdims=True)
    std_p = np.std(base_arr, axis=1, keepdims=True)
    spread = np.max(base_arr, axis=1, keepdims=True) - np.min(base_arr, axis=1, keepdims=True)
    max_c = np.max(np.abs(base_arr - 0.5), axis=1, keepdims=True)

    parts = [base_arr, mean_p, std_p, spread, max_c, X_ctx]
    if X_ctx.shape[1] >= 7:
        parts.append(X_ctx[:, 4:5] * X_ctx[:, 6:7])
        parts.append(X_ctx[:, 4:5] * std_p)

    X_meta = np.hstack(parts)
    raw_prob = meta_learner.predict_proba(X_meta)[0]

    # Calibrate with historical upset base rates + specialist disagreement
    seed_diff = int(features.get("seed_diff", 0))
    prob = _calibrate_with_upset_prior(raw_prob, seed_diff,
                                        base_preds=[float(b) for b in base_arr[0]])

    # Conformal
    rg = int(features.get("round_group", 1))
    conf_90 = conformal.predict_sets(np.array([prob]), np.array([rg]), confidence=0.90)[0]
    conf_80 = conformal.predict_sets(np.array([prob]), np.array([rg]), confidence=0.80)[0]

    if conf_90["set_size"] == 1:
        confidence = "HIGH"
    elif conf_80["set_size"] == 2:
        confidence = "LOW"
    else:
        confidence = "MED"

    agree = sum(1 for b in base_arr[0] if (b > 0.5) == (prob > 0.5))

    return {
        "prob_team_a": float(prob),
        "raw_prob": float(raw_prob),
        "confidence": confidence,
        "agreement": f"{agree}/{len(base_arr[0])}",
        "base_preds": [float(b) for b in base_arr[0]],
    }


def load_artifacts():
    """Load all model artifacts."""
    models_dir = config.MODELS_DIR
    with open(models_dir / "stacking_metadata.json") as f:
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


def load_team_stats():
    """Load 2026 team stats as a dict of {team_name: {stat: value}}."""
    stats = pl.read_parquet(config.PROCESSED_DIR / "team_season_stats.parquet")
    s26 = stats.filter(pl.col("season") == 2026)

    meta_cols = {"team_id", "team_name", "season"}
    stat_cols = [c for c in s26.columns if c not in meta_cols]

    team_stats = {}
    for row in s26.iter_rows(named=True):
        name = row["team_name"]
        team_stats[name] = {c: row[c] for c in stat_cols if row[c] is not None}

    return team_stats, stat_cols


def predict_bracket():
    """Predict the full 2026 bracket round-by-round."""
    print("[Bracket] Loading models and data...")
    artifacts = load_artifacts()
    team_stats, stat_cols = load_team_stats()
    stats_names = set(team_stats.keys())

    # Build team index for name resolution
    index_path = config.RAW_DIR / "team_index.parquet"
    ti = TeamIndex(path=index_path)
    if index_path.exists():
        ti.load()

    # Resolve bracket names to stats names
    def resolve(name):
        mapped = NAME_MAP.get(name, name)
        if mapped in stats_names:
            return mapped
        resolved = _resolve_team_name(name, ti, stats_names)
        return resolved

    # Track resolved names and warn about missing
    resolved_map = {}
    missing = []
    for region, teams in BRACKET_2026.items():
        for seed, name in teams.items():
            if name.startswith("TBD"):
                continue
            r = resolve(name)
            if r:
                resolved_map[name] = r
            else:
                missing.append(f"  {region} ({seed}) {name}")

    if missing:
        print(f"  WARNING: Could not find stats for {len(missing)} teams:")
        for m in missing:
            print(m)

    print(f"  Resolved {len(resolved_map)}/{sum(len(t) for t in BRACKET_2026.values())} teams to stats")

    # ── Predict round by round ───────────────────────────────────────────
    bracket_results = {}  # {region: {round: [(seed_a, name_a, seed_b, name_b, prob, winner, conf)]}}
    region_winners = {}   # {region: (seed, name)}

    round_names = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
                   4: "Elite 8", 5: "Final Four", 6: "Championship"}

    # Expected upset count per round (across all 4 regions combined)
    # Based on historical data: R64 ~8.8, R32 ~2.8, S16 ~2.5, E8 ~1.5
    EXPECTED_UPSETS = {1: 9, 2: 3, 3: 3, 4: 2}

    def _predict_round(games_data, rnd, region_label=""):
        """Predict all games in a round, return list of results with raw upset probabilities."""
        round_results = []
        for g in games_data:
            seed_a, name_a, seed_b, name_b = g["seed_a"], g["name_a"], g["seed_b"], g["name_b"]

            if name_a.startswith("TBD") or name_a.startswith("[") or name_b.startswith("TBD") or name_b.startswith("["):
                tbd_name = name_a if (name_a.startswith("TBD") or name_a.startswith("[")) else name_b
                non_tbd = name_b if (name_a.startswith("TBD") or name_a.startswith("[")) else name_a
                non_tbd_seed = seed_b if (name_a.startswith("TBD") or name_a.startswith("[")) else seed_a
                round_results.append({
                    "seed_a": seed_a, "name_a": name_a,
                    "seed_b": seed_b, "name_b": name_b,
                    "prob": 0.5, "upset_prob": 0.0,
                    "chalk_winner": non_tbd, "chalk_seed": non_tbd_seed,
                    "upset_winner": tbd_name, "upset_seed": seed_b if non_tbd == name_a else seed_a,
                    "confidence": "TBD", "agreement": "", "is_tbd": True,
                })
                continue

            stats_a = team_stats.get(resolved_map.get(name_a, name_a), {})
            stats_b = team_stats.get(resolved_map.get(name_b, name_b), {})
            features = _build_matchup_features(stats_a, stats_b, seed_a, seed_b, rnd, stat_cols)
            result = _predict_game(features, artifacts)
            prob = result["prob_team_a"]

            # Upset probability = P(lower seed wins)
            upset_prob = 1.0 - prob  # team_b is lower seed (higher number)

            round_results.append({
                "seed_a": seed_a, "name_a": name_a,
                "seed_b": seed_b, "name_b": name_b,
                "prob": float(prob),
                "upset_prob": float(upset_prob),
                "chalk_winner": name_a, "chalk_seed": seed_a,
                "upset_winner": name_b, "upset_seed": seed_b,
                "confidence": result["confidence"],
                "agreement": result["agreement"],
                "base_preds": result.get("base_preds", []),
                "is_tbd": False,
            })
        return round_results

    def _select_upsets(round_results, expected_upsets):
        """Select which games to flip to upsets based on upset probability ranking.

        Rules:
        - Skip TBD games entirely
        - Never upset a 1-seed in R64 (1 vs 16, too rare)
        - Never force an upset where model gives <20% chance
        - Sort real games by upset_prob descending
        - Flip the top N where N = expected_upsets
        """
        # Collect candidates for upset (exclude TBD and extreme mismatches)
        candidates = []
        for i, g in enumerate(round_results):
            if g["is_tbd"]:
                continue
            sd = abs(g["seed_a"] - g["seed_b"])
            # Don't force 16-over-1 upsets
            if sd >= 15:
                continue
            # Only consider if model gives meaningful chance (>20%)
            if g["upset_prob"] > 0.20:
                candidates.append((i, g["upset_prob"]))

        # Sort by upset probability (highest first)
        candidates.sort(key=lambda x: -x[1])

        # Select top N upsets
        upset_indices = set()
        for idx, uprob in candidates[:expected_upsets]:
            upset_indices.add(idx)

        # Apply selections
        final_results = []
        for i, g in enumerate(round_results):
            g = dict(g)  # copy
            if g["is_tbd"]:
                # TBD: default to the non-TBD team (higher seed side)
                g["winner"] = g["chalk_winner"]
                g["winner_seed"] = g["chalk_seed"]
                g["is_upset"] = False
            elif i in upset_indices:
                g["winner"] = g["upset_winner"]
                g["winner_seed"] = g["upset_seed"]
                g["is_upset"] = True
            else:
                if g["prob"] > 0.5:
                    g["winner"] = g["chalk_winner"]
                    g["winner_seed"] = g["chalk_seed"]
                else:
                    g["winner"] = g["upset_winner"]
                    g["winner_seed"] = g["upset_seed"]
                g["is_upset"] = g["winner_seed"] > min(g["seed_a"], g["seed_b"])
            final_results.append(g)

        return final_results

    # Collect ALL R64 games across all regions first, then select upsets globally
    all_r64_games = []
    r64_region_map = []  # track which region each game belongs to

    for region_name, teams in BRACKET_2026.items():
        for seed_a, seed_b in R64_MATCHUPS:
            name_a = teams[seed_a]
            name_b = teams[seed_b]
            all_r64_games.append({"seed_a": seed_a, "name_a": name_a, "seed_b": seed_b, "name_b": name_b})
            r64_region_map.append(region_name)

    # Predict all R64 games
    r64_results = _predict_round(all_r64_games, rnd=1)

    # Select upsets globally across all regions
    r64_with_upsets = _select_upsets(r64_results, EXPECTED_UPSETS[1])

    # Distribute back to regions and print
    region_idx = {}
    for region_name in BRACKET_2026:
        region_idx[region_name] = 0

    current_teams_by_region = {}
    for region_name in BRACKET_2026:
        bracket_results[region_name] = {1: []}
        current_teams_by_region[region_name] = []

    for i, g in enumerate(r64_with_upsets):
        region_name = r64_region_map[i]
        bracket_results[region_name][1].append(g)
        current_teams_by_region[region_name].append((g["winner_seed"], g["winner"]))

    # Print R64 results
    for region_name in BRACKET_2026:
        print(f"\n  === {region_name.upper()} REGION ===")
        for g in bracket_results[region_name][1]:
            upset_tag = " ** UPSET **" if g.get("is_upset") else ""
            conf = g.get("confidence", "TBD")
            agree = g.get("agreement", "")
            prob_display = max(g["prob"], 1 - g["prob"])
            print(f"    ({g['seed_a']:>2}) {g['name_a']:<22} vs ({g['seed_b']:>2}) {g['name_b']:<22}"
                  f" -> ({g['winner_seed']:>2}) {g['winner']:<22} {prob_display:>5.1%}"
                  f" [{conf}]{upset_tag}")

    # Subsequent rounds (per region)
    for rnd in range(2, 5):
        rnd_name = round_names[rnd]

        # Collect all games this round across regions
        all_games = []
        game_region_map = []
        for region_name in BRACKET_2026:
            current_teams = current_teams_by_region[region_name]
            for j in range(0, len(current_teams), 2):
                if j + 1 >= len(current_teams):
                    all_games.append(None)
                    game_region_map.append(region_name)
                    continue
                seed_a, name_a = current_teams[j]
                seed_b, name_b = current_teams[j + 1]
                if seed_a > seed_b:
                    seed_a, name_a, seed_b, name_b = seed_b, name_b, seed_a, name_a
                all_games.append({"seed_a": seed_a, "name_a": name_a, "seed_b": seed_b, "name_b": name_b})
                game_region_map.append(region_name)

        # Predict and select upsets
        valid_games = [g for g in all_games if g is not None]
        rnd_results = _predict_round(valid_games, rnd=rnd)
        rnd_with_upsets = _select_upsets(rnd_results, EXPECTED_UPSETS.get(rnd, 1))

        # Distribute back
        for region_name in BRACKET_2026:
            bracket_results[region_name][rnd] = []
            current_teams_by_region[region_name] = []

        result_idx = 0
        for i, region_name in enumerate(game_region_map):
            if all_games[i] is None:
                continue
            g = rnd_with_upsets[result_idx]
            bracket_results[region_name][rnd].append(g)
            current_teams_by_region[region_name].append((g["winner_seed"], g["winner"]))
            result_idx += 1

        # Print
        for region_name in BRACKET_2026:
            games = bracket_results[region_name][rnd]
            if games:
                print(f"\n    -- {region_name} {rnd_name} --")
                for g in games:
                    upset_tag = " ** UPSET **" if g.get("is_upset") else ""
                    conf = g.get("confidence", "TBD")
                    prob_display = max(g["prob"], 1 - g["prob"])
                    print(f"    ({g['seed_a']:>2}) {g['name_a']:<22} vs ({g['seed_b']:>2}) {g['name_b']:<22}"
                          f" -> ({g['winner_seed']:>2}) {g['winner']:<22} {prob_display:>5.1%}"
                          f" [{conf}]{upset_tag}")

    # Region winners
    for region_name in BRACKET_2026:
        ct = current_teams_by_region[region_name]
        if ct:
            region_winners[region_name] = ct[0]
            print(f"\n    >>> {region_name} CHAMPION: ({ct[0][0]}) {ct[0][1]}")

    # ── Final Four ───────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  FINAL FOUR")
    print(f"  {'='*60}")

    # Per ESPN 2026 bracket: East/South on left side, West/Midwest on right side
    ff_matchups = [("East", "South"), ("West", "Midwest")]
    ff_winners = []
    bracket_results["Final Four"] = {5: []}

    for r1, r2 in ff_matchups:
        if r1 not in region_winners or r2 not in region_winners:
            continue
        seed_a, name_a = region_winners[r1]
        seed_b, name_b = region_winners[r2]

        if seed_a > seed_b:
            seed_a, name_a, seed_b, name_b = seed_b, name_b, seed_a, name_a

        stats_a = team_stats.get(resolved_map.get(name_a, name_a), {})
        stats_b = team_stats.get(resolved_map.get(name_b, name_b), {})
        features = _build_matchup_features(stats_a, stats_b, seed_a, seed_b, 5, stat_cols)
        result = _predict_game(features, artifacts)

        prob = result["prob_team_a"]
        if prob > 0.5:
            winner_name, winner_seed = name_a, seed_a
        else:
            winner_name, winner_seed = name_b, seed_b

        ff_winners.append((winner_seed, winner_name))
        bracket_results["Final Four"][5].append({
            "seed_a": seed_a, "name_a": name_a, "region_a": r1,
            "seed_b": seed_b, "name_b": name_b, "region_b": r2,
            "prob": float(prob), "winner": winner_name, "winner_seed": winner_seed,
            "confidence": result["confidence"],
            "agreement": result["agreement"],
        })

        print(f"  ({seed_a:>2}) {name_a:<22} ({r1}) vs ({seed_b:>2}) {name_b:<22} ({r2})"
              f" -> ({winner_seed:>2}) {winner_name:<22} {max(prob, 1-prob):>5.1%} [{result['confidence']}]")

    # ── Championship ─────────────────────────────────────────────────────
    if len(ff_winners) == 2:
        print(f"\n  {'='*60}")
        print(f"  CHAMPIONSHIP")
        print(f"  {'='*60}")

        seed_a, name_a = ff_winners[0]
        seed_b, name_b = ff_winners[1]
        if seed_a > seed_b:
            seed_a, name_a, seed_b, name_b = seed_b, name_b, seed_a, name_a

        stats_a = team_stats.get(resolved_map.get(name_a, name_a), {})
        stats_b = team_stats.get(resolved_map.get(name_b, name_b), {})
        features = _build_matchup_features(stats_a, stats_b, seed_a, seed_b, 6, stat_cols)
        result = _predict_game(features, artifacts)

        prob = result["prob_team_a"]
        if prob > 0.5:
            champ_name, champ_seed = name_a, seed_a
        else:
            champ_name, champ_seed = name_b, seed_b

        bracket_results["Final Four"][6] = [{
            "seed_a": seed_a, "name_a": name_a,
            "seed_b": seed_b, "name_b": name_b,
            "prob": float(prob), "winner": champ_name, "winner_seed": champ_seed,
            "confidence": result["confidence"],
            "agreement": result["agreement"],
        }]

        print(f"  ({seed_a:>2}) {name_a:<22} vs ({seed_b:>2}) {name_b:<22}"
              f" -> ({champ_seed:>2}) {champ_name:<22} {max(prob, 1-prob):>5.1%} [{result['confidence']}]")

        print(f"\n  {'*'*60}")
        print(f"  PREDICTED 2026 NATIONAL CHAMPION: ({champ_seed}) {champ_name}")
        print(f"  {'*'*60}")

    # Save results
    out_path = config.DATA_DIR / "bracket_2026.json"
    with open(out_path, "w") as f:
        json.dump(bracket_results, f, indent=2, default=str)
    print(f"\n  -> Bracket saved to {out_path}")

    return bracket_results, region_winners


def generate_html(bracket_results: dict = None):
    """Generate an HTML bracket visualization."""
    if bracket_results is None:
        path = config.DATA_DIR / "bracket_2026.json"
        with open(path) as f:
            bracket_results = json.load(f)

    regions = ["East", "West", "South", "Midwest"]

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2026 March Madness Bracket Predictions</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0a1a; color: #e0e0e0; padding: 20px; }
  h1 { text-align: center; color: #fff; margin-bottom: 5px; font-size: 28px; }
  .subtitle { text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }
  .bracket-container { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; max-width: 1400px; margin: 0 auto; }
  .region { background: #12122a; border-radius: 12px; padding: 20px; border: 1px solid #2a2a4a; }
  .region-title { font-size: 20px; font-weight: 700; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #3a3a5a; }
  .round { margin-bottom: 15px; }
  .round-title { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .game { display: flex; align-items: center; margin-bottom: 6px; padding: 6px 10px; border-radius: 6px; background: #1a1a35; font-size: 13px; }
  .game:hover { background: #22224a; }
  .seed { display: inline-block; width: 24px; text-align: center; font-weight: 700; color: #888; font-size: 12px; }
  .team { flex: 1; margin-left: 6px; }
  .team.winner { color: #4ade80; font-weight: 600; }
  .team.loser { color: #666; }
  .vs { color: #555; margin: 0 8px; font-size: 11px; }
  .prob { font-size: 12px; margin-left: 8px; min-width: 45px; text-align: right; }
  .conf-high { color: #4ade80; }
  .conf-med { color: #facc15; }
  .conf-low { color: #f87171; }
  .conf-tbd { color: #666; }
  .upset { background: #2a1a1a !important; border-left: 3px solid #f87171; }
  .final-four { grid-column: 1 / -1; background: linear-gradient(135deg, #1a1a3a, #2a1a2a); border: 1px solid #4a3a5a; }
  .champion { text-align: center; padding: 20px; font-size: 24px; font-weight: 700; color: #fbbf24; }
  .champion .seed-big { font-size: 16px; color: #888; }
  .legend { display: flex; justify-content: center; gap: 20px; margin-top: 20px; font-size: 12px; color: #888; }
  .legend span { display: flex; align-items: center; gap: 5px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .agree { font-size: 11px; color: #555; margin-left: 4px; }
</style>
</head>
<body>
<h1>2026 March Madness Bracket Predictions</h1>
<p class="subtitle">Specialist Stacking Ensemble (XGB + LGB-Close + LR-Late + RF) + Conformal Prediction</p>
<div class="bracket-container">
"""

    round_names = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8"}

    for region in regions:
        if region not in bracket_results:
            continue
        color = {"East": "#60a5fa", "West": "#f472b6", "South": "#4ade80", "Midwest": "#facc15"}[region]
        html += f'<div class="region">\n'
        html += f'<div class="region-title" style="color: {color}">{region.upper()} REGION</div>\n'

        for rnd in [1, 2, 3, 4]:
            games = bracket_results[region].get(str(rnd), bracket_results[region].get(rnd, []))
            if not games:
                continue
            html += f'<div class="round"><div class="round-title">{round_names[rnd]}</div>\n'

            for g in games:
                prob = g.get("prob", 0.5)
                winner = g.get("winner", "")
                conf = g.get("confidence", "MED")
                agree = g.get("agreement", "")
                is_upset = g["winner_seed"] > min(g["seed_a"], g["seed_b"])
                upset_cls = " upset" if is_upset else ""

                name_a = g["name_a"]
                name_b = g["name_b"]
                cls_a = "winner" if winner == name_a else "loser"
                cls_b = "winner" if winner == name_b else "loser"
                conf_cls = f"conf-{conf.lower()}"
                prob_display = f"{max(prob, 1-prob):.0%}"

                html += f'<div class="game{upset_cls}">'
                html += f'<span class="seed">{g["seed_a"]}</span>'
                html += f'<span class="team {cls_a}">{name_a}</span>'
                html += f'<span class="vs">vs</span>'
                html += f'<span class="seed">{g["seed_b"]}</span>'
                html += f'<span class="team {cls_b}">{name_b}</span>'
                html += f'<span class="prob {conf_cls}">{prob_display}</span>'
                if agree:
                    html += f'<span class="agree">{agree}</span>'
                html += f'</div>\n'

            html += '</div>\n'
        html += '</div>\n'

    # Final Four
    ff_data = bracket_results.get("Final Four", {})
    ff_games = ff_data.get("5", ff_data.get(5, []))
    champ_games = ff_data.get("6", ff_data.get(6, []))

    html += '<div class="region final-four">\n'
    html += '<div class="region-title" style="color: #c084fc">FINAL FOUR & CHAMPIONSHIP</div>\n'

    if ff_games:
        html += '<div class="round"><div class="round-title">Final Four</div>\n'
        for g in ff_games:
            prob = g.get("prob", 0.5)
            winner = g.get("winner", "")
            conf = g.get("confidence", "MED")
            agree = g.get("agreement", "")
            cls_a = "winner" if winner == g["name_a"] else "loser"
            cls_b = "winner" if winner == g["name_b"] else "loser"
            conf_cls = f"conf-{conf.lower()}"
            region_a = g.get("region_a", "")
            region_b = g.get("region_b", "")

            html += f'<div class="game">'
            html += f'<span class="seed">{g["seed_a"]}</span>'
            html += f'<span class="team {cls_a}">{g["name_a"]} <small style="color:#666">({region_a})</small></span>'
            html += f'<span class="vs">vs</span>'
            html += f'<span class="seed">{g["seed_b"]}</span>'
            html += f'<span class="team {cls_b}">{g["name_b"]} <small style="color:#666">({region_b})</small></span>'
            html += f'<span class="prob {conf_cls}">{max(prob, 1-prob):.0%}</span>'
            if agree:
                html += f'<span class="agree">{agree}</span>'
            html += f'</div>\n'
        html += '</div>\n'

    if champ_games:
        g = champ_games[0]
        prob = g.get("prob", 0.5)
        winner = g.get("winner", "")
        conf = g.get("confidence", "MED")
        conf_cls = f"conf-{conf.lower()}"

        html += '<div class="round"><div class="round-title">Championship</div>\n'
        cls_a = "winner" if winner == g["name_a"] else "loser"
        cls_b = "winner" if winner == g["name_b"] else "loser"
        html += f'<div class="game">'
        html += f'<span class="seed">{g["seed_a"]}</span>'
        html += f'<span class="team {cls_a}">{g["name_a"]}</span>'
        html += f'<span class="vs">vs</span>'
        html += f'<span class="seed">{g["seed_b"]}</span>'
        html += f'<span class="team {cls_b}">{g["name_b"]}</span>'
        html += f'<span class="prob {conf_cls}">{max(prob, 1-prob):.0%}</span>'
        html += f'</div>\n</div>\n'

        html += f'<div class="champion">'
        html += f'<div>PREDICTED CHAMPION</div>'
        html += f'<div style="font-size:32px; margin-top:10px;">({g["winner_seed"]}) {winner}</div>'
        html += f'<div class="{conf_cls}" style="font-size:14px; margin-top:5px;">'
        html += f'Confidence: {max(prob, 1-prob):.1%} [{conf}]</div>'
        html += f'</div>\n'

    html += '</div>\n'  # final-four

    html += """</div>
<div class="legend">
  <span><span class="legend-dot" style="background:#4ade80"></span> HIGH confidence</span>
  <span><span class="legend-dot" style="background:#facc15"></span> MED confidence</span>
  <span><span class="legend-dot" style="background:#f87171"></span> LOW confidence</span>
  <span><span class="legend-dot" style="background:#f87171; opacity:0.5"></span> Upset pick</span>
</div>
</body>
</html>"""

    out_path = config.DATA_DIR / "bracket_2026.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  -> HTML bracket saved to {out_path}")
    return out_path


def run():
    """Full bracket prediction + visualization."""
    results, winners = predict_bracket()
    html_path = generate_html(results)
    print(f"\n[Bracket] Done. Open {html_path} to view the bracket.")


if __name__ == "__main__":
    run()
