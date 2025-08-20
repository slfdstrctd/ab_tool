import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_metric_cards(metrics_data: Dict[str, Any]):
    cols = st.columns(len(metrics_data))

    for i, (label, value) in enumerate(metrics_data.items()):
        with cols[i]:
            if isinstance(value, (int, float)):
                if value >= 1000:
                    display_value = f"{value:,.0f}"
                else:
                    display_value = f"{value:.2f}"
            else:
                display_value = str(value)

            st.metric(label, display_value)


def create_decision_card(decision: str, confidence: float, reasoning: List[str]):
    decision_colors = {
        "ACCEPT": ("#28a745"),
        "REJECT": ("#dc3545"),
        "KEEP RUNNING": ("#ffc107"),
    }

    color = decision_colors.get(decision, ("#17a2b8"))

    st.markdown(
        f"""
    <div style="
        background-color: {"#d4edda" if decision == "ACCEPT" else "#f8d7da" if decision == "REJECT" else "#fff3cd"};
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid {color};
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h2 style="color: {color}; margin: 0 0 0.5rem 0;">
            Decision: {decision}
        </h2>
        <p style="font-size: 1.1rem; margin: 0 0 1rem 0;">
            <strong>Confidence:</strong> {confidence:.1f}%
        </p>
        <div style="font-size: 0.95rem;">
            <strong>Reasoning:</strong>
            <ul style="margin: 0.5rem 0 0 1rem;">
                {"".join([f"<li>{reason}</li>" for reason in reasoning])}
            </ul>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_statistical_results_table(test_results: Dict[str, List[Any]]):
    for metric, results in test_results.items():
        st.subheader(f"{metric} Results")

        result_data = []
        for result in results:
            result_data.append(
                {
                    "Test Method": result.test_name,
                    "P-value": f"{result.p_value:.4f}",
                    "Effect Size": f"{result.effect_size:.4f}",
                    "Relative Difference": f"{result.relative_difference:+.2f}%",
                    "Significant (α=0.12)": f"{result.is_significant}",
                    "Control Mean": f"{result.control_mean:.4f}",
                    "Treatment Mean": f"{result.treatment_mean:.4f}",
                }
            )

        df = pd.DataFrame(result_data)
        st.dataframe(df, use_container_width=True)


def create_sample_size_analysis(control_size: int, treatment_size: int):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Control", "Treatment"],
            y=[control_size, treatment_size],
            marker_color=["lightblue", "lightcoral"],
            text=[f"{control_size:,}", f"{treatment_size:,}"],
            textposition="auto",
        )
    )

    balanced_size = (control_size + treatment_size) / 2
    fig.add_hline(
        y=balanced_size,
        line_dash="dash",
        line_color="gray",
        annotation_text="Perfect Balance",
    )

    fig.update_layout(
        title="Sample Size Distribution", yaxis_title="Number of Users", height=400
    )

    return fig


def create_effect_size_radar_chart(test_results: Dict[str, List[Any]]):
    metrics = list(test_results.keys())

    effect_sizes = [results[0].effect_size for results in test_results.values()]
    p_values = [results[0].p_value for results in test_results.values()]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=effect_sizes,
            theta=metrics,
            fill="toself",
            name="Effect Size",
            line_color="blue",
        )
    )

    significance_threshold = [0.2] * len(metrics)
    fig.add_trace(
        go.Scatterpolar(
            r=significance_threshold,
            theta=metrics,
            fill="toself",
            name="Small Effect Threshold",
            line_color="red",
            line_dash="dash",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(effect_sizes), 0.5)])
        ),
        title="Effect Sizes by Metric",
        height=500,
    )

    return fig


def create_confidence_interval_plot(test_results: Dict[str, List[Any]]):
    fig = go.Figure()

    y_pos = 0
    colors = px.colors.qualitative.Set3

    for metric, results in test_results.items():
        for i, result in enumerate(results):
            color = colors[i % len(colors)]

            ci_lower, ci_upper = result.confidence_interval

            fig.add_trace(
                go.Scatter(
                    x=[ci_lower, ci_upper],
                    y=[y_pos, y_pos],
                    mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(size=8, color=color),
                    name=f"{metric} - {result.test_name}",
                    showlegend=True,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[result.absolute_difference],
                    y=[y_pos],
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="diamond"),
                    name=f"{metric} - Observed",
                    showlegend=False,
                )
            )

            y_pos += 1

    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)

    fig.update_layout(
        title="Confidence Intervals for All Tests",
        xaxis_title="Effect Size",
        yaxis=dict(
            tickmode="array",
            tickvals=list(
                range(len([r for results in test_results.values() for r in results]))
            ),
            ticktext=[
                f"{metric} - {result.test_name}"
                for metric, results in test_results.items()
                for result in results
            ],
        ),
        height=400 + len(test_results) * 100,
    )

    return fig


def create_power_analysis_plot(
    control_size: int, treatment_size: int, effect_sizes: List[float] = None
):
    if effect_sizes is None:
        effect_sizes = np.linspace(0.01, 0.5, 50)

    from scipy import stats

    powers = []

    for effect_size in effect_sizes:
        pooled_std = 1.0
        se_diff = pooled_std * np.sqrt(1 / control_size + 1 / treatment_size)
        t_critical = stats.t.ppf(0.94, control_size + treatment_size - 2)

        ncp = effect_size / se_diff
        power = 1 - stats.t.cdf(t_critical, control_size + treatment_size - 2, ncp)
        powers.append(power)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=effect_sizes,
            y=powers,
            mode="lines",
            line=dict(color="blue", width=3),
            name="Statistical Power",
        )
    )

    fig.add_hline(
        y=0.8, line_dash="dash", line_color="red", annotation_text="80% Power Threshold"
    )

    fig.update_layout(
        title=f"Statistical Power Analysis (N={control_size + treatment_size})",
        xaxis_title="Effect Size (Cohen's d)",
        yaxis_title="Statistical Power",
        yaxis=dict(range=[0, 1]),
        height=400,
    )

    return fig


def export_results_json(
    experiment_name: str, test_results: Dict, decision_result: Any
) -> str:
    export_data = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "decision": {
            "decision": decision_result.decision.value,
            "confidence": decision_result.confidence,
            "reasoning": decision_result.reasoning,
            "key_findings": decision_result.key_findings,
            "warnings": decision_result.warnings,
            "recommendations": decision_result.recommendations,
        },
        "statistical_results": {
            metric: [
                {
                    "test_name": result.test_name,
                    "p_value": float(result.p_value),
                    "effect_size": float(result.effect_size),
                    "relative_difference": float(result.relative_difference),
                    "confidence_interval": [
                        float(result.confidence_interval[0]),
                        float(result.confidence_interval[1]),
                    ],
                    "is_significant": bool(result.is_significant),
                    "control_mean": float(result.control_mean),
                    "treatment_mean": float(result.treatment_mean),
                    "sample_size_control": int(result.sample_size_control),
                    "sample_size_treatment": int(result.sample_size_treatment),
                }
                for result in results
            ]
            for metric, results in test_results.items()
        },
    }

    return json.dumps(export_data, indent=2)


def create_progress_tracker(steps: List[str], current_step: int):
    progress = current_step / len(steps)

    st.progress(progress)

    cols = st.columns(len(steps))

    for i, (step, col) in enumerate(zip(steps, cols)):
        with col:
            if i < current_step:
                st.success(f"{step}")
            elif i == current_step:
                st.info(f"{step}")
            else:
                st.write(f"{step}")


def render_experiment_timeline(experiments: List[str], results: Dict[str, Any]):
    st.subheader("Experiment Timeline")

    timeline_data = []
    for exp_name in experiments:
        if exp_name in results:
            result = results[exp_name]
            decision = result["decision_result"].decision.value
            confidence = result["decision_result"].confidence

            timeline_data.append(
                {
                    "Experiment": exp_name,
                    "Decision": decision,
                    "Confidence": confidence,
                    "Status": decision,
                }
            )

    if timeline_data:
        df = pd.DataFrame(timeline_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No experiment results to display")
