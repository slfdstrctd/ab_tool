import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click
import colorlog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DataLoader
from decision_engine import DecisionEngine
from statistical_tests import StatisticalTester
from visualization import Visualizer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
    )

    logger = colorlog.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


@click.command()
@click.option("--data-dir", "-d", default="data", help="Directory containing CSV files")
@click.option(
    "--experiment", "-e", help='Experiment name to analyze (e.g., "add_bttn_fix")'
)
@click.option(
    "--output-dir", "-o", default="outputs", help="Directory for output files"
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    default=["revenue_usd", "messages_count"],
    help="Metrics to analyze",
)
@click.option(
    "--significance-level", "-s", default=0.12, help="Significance level for tests"
)
@click.option(
    "--effect-sizes",
    "-es",
    multiple=True,
    type=float,
    default=[1.0, 2.0, 5.0, 10.0],
    help="Effect sizes to test (as percentages)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--all-experiments", is_flag=True, help="Analyze all available experiments"
)
def main(
    data_dir: str,
    experiment: Optional[str],
    output_dir: str,
    metrics: List[str],
    significance_level: float,
    effect_sizes: List[float],
    verbose: bool,
    all_experiments: bool,
):
    logger = setup_logging(verbose)
    logger.info("Starting A/B Experiment Tools")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        logger.info(f"Loading data from: {data_dir}")
        data_loader = DataLoader(data_dir)
        data = data_loader.load_all_data()

        quality_report = data_loader.get_data_quality_report()
        logger.info(f"Loaded {quality_report['total_users']} users")

        with open(output_path / "data_quality_report.json", "w") as f:
            json.dump(quality_report, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    if all_experiments:
        experiments_to_analyze = data_loader.get_all_experiments()
        logger.info(f"Found {len(experiments_to_analyze)} experiments to analyze")
    elif experiment:
        experiments_to_analyze = [experiment]
    else:
        available = data_loader.get_all_experiments()
        logger.info("Available experiments:")
        for exp in available:
            exp_name = exp.replace("exp_", "")
            details = quality_report["experiment_details"].get(exp, {})
            logger.info(
                f"  - {exp_name}: "
                f"{details.get('control', 0)} control, "
                f"{details.get('treatment', 0)} treatment users"
            )

        logger.error("Please specify an experiment with -e or use --all-experiments")
        sys.exit(1)

    stat_tester = StatisticalTester(significance_level)
    visualizer = Visualizer(output_dir)
    decision_engine = DecisionEngine(significance_threshold=significance_level)

    all_results = {}

    for exp_name in experiments_to_analyze:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Analyzing experiment: {exp_name}")
        logger.info(f"{'=' * 60}")

        try:
            # Create experiment-specific output directory
            exp_output_dir = output_path / exp_name
            exp_output_dir.mkdir(exist_ok=True)
            
            exp_data = data_loader.get_experiment_data(exp_name)
            control_data = exp_data["control"]
            treatment_data = exp_data["treatment"]

            logger.info(
                f"Control: {len(control_data)} users, "
                f"Treatment: {len(treatment_data)} users"
            )

            logger.info("Running statistical tests...")
            test_results = stat_tester.run_all_tests(
                control_data, treatment_data, ["revenue_usd", "messages_count"]
            )

            logger.info("Testing multiple effect sizes...")
            effect_size_results = {}
            for metric in ["revenue_usd", "messages_count"]:
                effect_size_results[metric] = stat_tester.test_multiple_effect_sizes(
                    control_data, treatment_data, metric, list(effect_sizes)
                )

            logger.info("Creating visualizations...")

            for metric in ["revenue_usd", "messages_count"]:
                if stat_tester.get_last_bootstrap_distribution():
                    bootstrap_dist, observed_diff = (
                        stat_tester.get_last_bootstrap_distribution()
                    )
                    bootstrap_result = test_results[metric][0]

                    visualizer.plot_bootstrap_distribution(
                        bootstrap_dist,
                        observed_diff,
                        metric,
                        bootstrap_result.p_value,
                        bootstrap_result.confidence_interval,
                        save_path=exp_output_dir / f"{exp_name}_bootstrap_{metric}.png",
                    )

            visualizer.plot_metric_comparison(
                control_data,
                treatment_data,
                ["revenue_usd", "messages_count"],
                save_path=exp_output_dir / f"{exp_name}_metric_comparison.png",
            )

            for metric, es_results in effect_size_results.items():
                visualizer.plot_effect_size_analysis(
                    es_results,
                    metric,
                    save_path=exp_output_dir / f"{exp_name}_effect_size_{metric}.png",
                )

            logger.info("Making automated decision...")
            decision_result = decision_engine.make_decision(
                test_results, quality_report, exp_name
            )

            logger.info(f"\n{'=' * 40}")
            logger.info(f"DECISION: {decision_result.decision.value}")
            logger.info(f"Confidence: {decision_result.confidence:.1f}%")
            logger.info(f"{'=' * 40}")

            logger.info("\nKey Findings:")
            for finding in decision_result.key_findings:
                logger.info(f"  • {finding}")

            if decision_result.warnings:
                logger.warning("\nWarnings:")
                for warning in decision_result.warnings:
                    logger.warning(f"  • {warning}")

            logger.info("\nRecommendations:")
            for rec in decision_result.recommendations:
                logger.info(f"  • {rec}")

            visualizer.create_interactive_dashboard(
                test_results,
                quality_report,
                exp_name,
                save_path=exp_output_dir / f"{exp_name}_dashboard.html",
            )

            visualizer.create_summary_report_plot(
                decision_result.decision.value,
                decision_result.confidence,
                decision_result.key_findings,
                decision_result.warnings,
                save_path=exp_output_dir / f"{exp_name}_summary.png",
            )

            results = {
                "experiment": exp_name,
                "decision": decision_result.decision.value,
                "confidence": decision_result.confidence,
                "reasoning": decision_result.reasoning,
                "key_findings": decision_result.key_findings,
                "warnings": decision_result.warnings,
                "recommendations": decision_result.recommendations,
                "test_results": {
                    metric: [
                        {
                            "test_name": result.test_name,
                            "p_value": float(result.p_value),
                            "effect_size": float(result.effect_size),
                            "relative_difference": float(result.relative_difference),
                            "is_significant": bool(result.is_significant),
                        }
                        for result in results
                    ]
                    for metric, results in test_results.items()
                },
                "timestamp": datetime.now().isoformat(),
            }

            with open(exp_output_dir / f"{exp_name}_results.json", "w") as f:
                json.dump(results, f, indent=2)

            all_results[exp_name] = results

        except Exception as e:
            logger.error(f"Failed to analyze {exp_name}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            continue

    if len(all_results) > 1:
        summary = {
            "total_experiments": len(all_results),
            "accepted": sum(
                1 for r in all_results.values() if r["decision"] == "ACCEPT"
            ),
            "rejected": sum(
                1 for r in all_results.values() if r["decision"] == "REJECT"
            ),
            "keep_running": sum(
                1 for r in all_results.values() if r["decision"] == "KEEP RUNNING"
            ),
            "experiments": all_results,
        }

        with open(output_path / "all_experiments_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'=' * 60}")
        logger.info("Summary of All Experiments:")
        logger.info(f"Accepted: {summary['accepted']}")
        logger.info(f"Rejected: {summary['rejected']}")
        logger.info(f"Keep Running: {summary['keep_running']}")

    logger.info(f"\nAnalysis complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
