"""NCAAB March Madness Prediction Pipeline - Orchestrator."""
import argparse
import sys

from scrapers import ncaa_api, teamrankings, warrennolan
from etl import build_team_index, process_stats, process_conferences, process_tournament
from features import build_features, select_features
from models import train, evaluate, predict, stacking, bracket


STAGES = {
    "scrape": [
        ("NCAA API (tournament games + schools)", ncaa_api.run),
        ("TeamRankings (team stats)", teamrankings.run),
        ("Warren Nolan (conference rankings)", warrennolan.run),
    ],
    "index": [
        ("Build team index", build_team_index.run),
    ],
    "etl": [
        ("Process team stats", process_stats.run),
        ("Process conference rankings", process_conferences.run),
        ("Process tournament matchups", process_tournament.run),
    ],
    "features": [
        ("Build matchup features", build_features.run),
        ("Select features (Pearson + MI + MCC)", select_features.run),
    ],
    "train": [
        ("Train stacking ensemble + conformal", stacking.run),
    ],
    "evaluate": [
        ("Evaluate models", evaluate.run),
    ],
    "predict": [
        ("Generate predictions", predict.run),
    ],
    "bracket": [
        ("Predict 2026 bracket + HTML visualization", bracket.run),
    ],
}

# Full pipeline order
STAGE_ORDER = ["scrape", "index", "etl", "features", "train", "evaluate", "predict", "bracket"]


def run_stage(stage_name: str, years=None):
    """Run a single pipeline stage."""
    if stage_name not in STAGES:
        print(f"Unknown stage: {stage_name}")
        print(f"Available: {', '.join(STAGES.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name}")
    print(f"{'='*60}")

    for desc, func in STAGES[stage_name]:
        print(f"\n>> {desc}")
        try:
            # Pass years arg to scraper functions if they accept it
            if stage_name == "scrape" and years:
                func(years=years)
            else:
                func()
        except Exception as e:
            print(f"  [ERROR] {desc} failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="NCAAB March Madness Prediction Pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to process (e.g., '2023,2024,2025')",
    )
    args = parser.parse_args()

    years = None
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]

    if args.stage == "all":
        for stage in STAGE_ORDER:
            run_stage(stage, years=years)
    else:
        run_stage(args.stage, years=years)

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
