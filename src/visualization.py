import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class Visualizer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_bootstrap_distribution(
        self,
        bootstrap_diffs: np.ndarray,
        observed_diff: float,
        metric_name: str,
        p_value: float,
        confidence_interval: Tuple[float, float],
        save_path: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))

        n, bins, patches = ax.hist(
            bootstrap_diffs,
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        ax.axvline(
            observed_diff,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Observed Difference: {observed_diff:.4f}",
        )

        ax.axvline(
            confidence_interval[0],
            color="green",
            linestyle=":",
            label=f"{(1 - p_value) * 100:.0f}% CI",
        )
        ax.axvline(confidence_interval[1], color="green", linestyle=":")

        ax.fill_betweenx(
            [0, max(n)],
            confidence_interval[0],
            confidence_interval[1],
            alpha=0.2,
            color="green",
        )

        ax.axvline(0, color="black", linestyle="-", alpha=0.5)

        ax.set_xlabel("Difference in Means")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Bootstrap Distribution for {metric_name}\np-value: {p_value:.4f}"
        )
        ax.legend()

        textstr = f"Mean: {np.mean(bootstrap_diffs):.4f}\n"
        textstr += f"Std: {np.std(bootstrap_diffs):.4f}\n"
        textstr += f"CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            save_path = self.output_dir / f"bootstrap_{metric_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()
        logger.info(f"Saved bootstrap distribution plot to {save_path}")

    def plot_metric_comparison(
        self,
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metrics: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            control_values = control_data[metric].values
            treatment_values = treatment_data[metric].values

            data = pd.DataFrame(
                {
                    "Group": ["Control"] * len(control_values)
                    + ["Treatment"] * len(treatment_values),
                    metric: np.concatenate([control_values, treatment_values]),
                }
            )

            sns.violinplot(data=data, x="Group", y=metric, ax=ax)

            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)

            ax.hlines(
                control_mean,
                -0.4,
                0.4,
                colors="red",
                linestyles="dashed",
                label=f"Control Mean: {control_mean:.2f}",
            )
            ax.hlines(
                treatment_mean,
                0.6,
                1.4,
                colors="red",
                linestyles="dashed",
                label=f"Treatment Mean: {treatment_mean:.2f}",
            )

            ax.text(
                0,
                ax.get_ylim()[0] * 0.95,
                f"n={len(control_values)}",
                ha="center",
                va="top",
            )
            ax.text(
                1,
                ax.get_ylim()[0] * 0.95,
                f"n={len(treatment_values)}",
                ha="center",
                va="top",
            )

            ax.set_title(f"{metric} Distribution")
            ax.legend()

        plt.suptitle("Metric Comparison: Control vs Treatment", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            save_path = self.output_dir / "metric_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()
        logger.info(f"Saved metric comparison plot to {save_path}")

    def create_interactive_dashboard(
        self,
        test_results: Dict[str, List],
        data_quality: Dict[str, Any],
        experiment_name: str,
        save_path: Optional[str] = None,
    ) -> None:
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "P-values by Test Type",
                "Effect Sizes",
                "Sample Sizes",
                "Relative Differences",
                "Confidence Intervals",
                "Data Quality",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        metrics = list(test_results.keys())
        test_types = []
        p_values = []
        effect_sizes = []
        relative_diffs = []
        ci_lowers = []
        ci_uppers = []

        for metric, results in test_results.items():
            for result in results:
                test_types.append(f"{metric} - {result.test_name}")
                p_values.append(result.p_value)
                effect_sizes.append(result.effect_size)
                relative_diffs.append(result.relative_difference)
                ci_lowers.append(result.confidence_interval[0])
                ci_uppers.append(result.confidence_interval[1])

        colors = ["green" if p < 0.12 else "red" for p in p_values]
        fig.add_trace(
            go.Bar(
                x=test_types,
                y=p_values,
                marker_color=colors,
                text=[f"{p:.4f}" for p in p_values],
                textposition="auto",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=0.12,
            line_dash="dash",
            line_color="black",
            annotation_text="α = 0.12",
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=test_types,
                y=effect_sizes,
                text=[f"{e:.3f}" for e in effect_sizes],
                textposition="auto",
            ),
            row=1,
            col=2,
        )

        if test_results:
            first_metric = list(test_results.keys())[0]
            first_result = test_results[first_metric][0]
            sample_data = pd.DataFrame(
                {
                    "Group": ["Control", "Treatment"],
                    "Sample Size": [
                        first_result.sample_size_control,
                        first_result.sample_size_treatment,
                    ],
                }
            )
            fig.add_trace(
                go.Bar(
                    x=sample_data["Group"],
                    y=sample_data["Sample Size"],
                    text=sample_data["Sample Size"],
                    textposition="auto",
                ),
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Bar(
                x=test_types,
                y=relative_diffs,
                text=[f"{r:.2f}%" for r in relative_diffs],
                textposition="auto",
            ),
            row=2,
            col=2,
        )

        for i, test_type in enumerate(test_types):
            fig.add_trace(
                go.Scatter(
                    x=[ci_lowers[i], ci_uppers[i]],
                    y=[test_type, test_type],
                    mode="lines+markers",
                    marker=dict(size=10),
                    line=dict(width=3),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

        quality_data = []
        quality_data.append(["Total Users", str(data_quality["total_users"])])
        quality_data.append(["Total Messages", str(data_quality["total_messages"])])
        quality_data.append(["Total Payments", str(data_quality["total_payments"])])

        if experiment_name in data_quality["experiment_details"]:
            exp_details = data_quality["experiment_details"][experiment_name]
            quality_data.append(["Control Users", str(exp_details["control"])])
            quality_data.append(["Treatment Users", str(exp_details["treatment"])])
            # quality_data.append(
            #     ["Not in Experiment", str(exp_details["not_in_experiment"])]
            # )

        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=list(zip(*quality_data))),
            ),
            row=3,
            col=2,
        )

        fig.update_layout(
            title=f"A/B Test Dashboard: {experiment_name}",
            height=1200,
            showlegend=False,
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text="P-value", row=1, col=1)
        fig.update_yaxes(title_text="Effect Size", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Relative Diff (%)", row=2, col=2)
        fig.update_xaxes(title_text="Confidence Interval", row=3, col=1)

        if save_path:
            fig.write_html(save_path)
        else:
            save_path = self.output_dir / f"dashboard_{experiment_name}.html"
            fig.write_html(str(save_path))

        logger.info(f"Saved interactive dashboard to {save_path}")

    def plot_effect_size_analysis(
        self,
        effect_size_results: Dict[float, Any],
        metric_name: str,
        save_path: Optional[str] = None,
    ) -> None:
        effect_sizes = sorted(effect_size_results.keys())
        p_values = [effect_size_results[es].p_value for es in effect_sizes]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(effect_sizes, p_values, "bo-", linewidth=2, markersize=8)

        ax.axhline(
            y=0.12, color="red", linestyle="--", label="Significance Threshold (α=0.12)"
        )

        ax.fill_between(
            effect_sizes, 0, 0.12, alpha=0.2, color="green", label="Significant Region"
        )

        ax.set_xlabel("Effect Size (%)")
        ax.set_ylabel("P-value")
        ax.set_title(f"Effect Size Analysis for {metric_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        for es, p_val in zip(effect_sizes, p_values):
            if p_val < 0.12:
                ax.annotate(
                    f"{es}%\np={p_val:.3f}",
                    xy=(es, p_val),
                    xytext=(es, p_val + 0.02),
                    ha="center",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            save_path = self.output_dir / f"effect_size_analysis_{metric_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()
        logger.info(f"Saved effect size analysis plot to {save_path}")

    def create_summary_report_plot(
        self,
        decision: str,
        confidence: float,
        key_findings: List[str],
        warnings: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")

        title_color = (
            "green"
            if decision == "ACCEPT"
            else "red"
            if decision == "REJECT"
            else "orange"
        )
        ax.text(
            0.5,
            0.95,
            f"\nExperiment Decision: {decision}",
            fontsize=24,
            fontweight="bold",
            ha="center",
            color=title_color,
            transform=ax.transAxes,
        )

        ax.text(
            0.5,
            0.88,
            f"Confidence Level: {confidence:.1f}%",
            fontsize=16,
            ha="center",
            transform=ax.transAxes,
        )

        y_pos = 0.75
        ax.text(
            0.1,
            y_pos,
            "Key Findings:",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )

        for finding in key_findings[:5]:
            y_pos -= 0.08
            ax.text(
                0.15,
                y_pos,
                f"• {finding}",
                fontsize=12,
                transform=ax.transAxes,
                wrap=True,
            )

        if warnings:
            y_pos -= 0.1
            ax.text(
                0.1,
                y_pos,
                "Warnings:",
                fontsize=14,
                fontweight="bold",
                color="orange",
                transform=ax.transAxes,
            )

            for warning in warnings[:3]:
                y_pos -= 0.08
                ax.text(
                    0.15,
                    y_pos,
                    f"⚠ {warning}",
                    fontsize=12,
                    color="orange",
                    transform=ax.transAxes,
                    wrap=True,
                )

        rect = plt.Rectangle(
            (0.05, 0.05),
            0.9,
            0.9,
            fill=False,
            edgecolor="black",
            linewidth=2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            save_path = self.output_dir / "experiment_summary.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()
        logger.info(f"Saved summary report plot to {save_path}")
