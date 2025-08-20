import os
import shutil
import sys
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DataLoader
from decision_engine import Decision, DecisionEngine
from statistical_tests import StatisticalTester

st.set_page_config(
    page_title="A/B Testing Tool", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)


class ABTestingApp:
    def __init__(self):
        self.data_loader = None
        self.data = None
        self.experiments = []

    def run(self):
        self.render_sidebar()

        if "data_loaded" not in st.session_state:
            st.session_state.data_loaded = False

        if not st.session_state.data_loaded:
            self.render_data_upload()
        else:
            self.render_main_dashboard()

    def render_sidebar(self):
        st.sidebar.title("Settings")

        st.sidebar.subheader("Analysis Parameters")

        significance_level = st.sidebar.slider(
            "Significance Level (α)",
            min_value=0.01,
            max_value=0.20,
            value=0.12,
            step=0.01,
            help="P-value threshold for statistical significance",
        )

        min_sample_size = st.sidebar.number_input(
            "Minimum Sample Size",
            min_value=50,
            max_value=1000,
            value=100,
            help="Minimum users required per group",
        )

        bootstrap_iterations = st.sidebar.number_input(
            "Bootstrap Iterations",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Number of bootstrap resampling iterations",
        )

        st.session_state.significance_level = significance_level
        st.session_state.min_sample_size = min_sample_size
        st.session_state.bootstrap_iterations = bootstrap_iterations

        st.sidebar.subheader("Effect Sizes to Test (%)")
        effect_sizes = st.sidebar.multiselect(
            "Select effect sizes",
            options=[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0],
            default=[1.0, 2.0, 5.0, 10.0],
            help="Effect sizes to test as percentages",
        )
        st.session_state.effect_sizes = effect_sizes

        st.sidebar.subheader("Metrics")
        metrics = st.sidebar.multiselect(
            "Select metrics to analyze",
            options=["revenue_usd", "messages_count"],
            default=["revenue_usd", "messages_count"],
            help="Metrics to include in analysis",
        )
        st.session_state.metrics = metrics

        if st.sidebar.button("Reset Data", type="secondary"):
            for key in list(st.session_state.keys()):
                if key.startswith("data_") or key in [
                    "experiments",
                    "analysis_results",
                ]:
                    del st.session_state[key]
            st.rerun()

    def render_data_upload(self):
        st.header("Data Upload")

        tab1, tab2 = st.tabs(["Use Sample Data", "Upload Files"])

        with tab1:
            st.subheader("Use Existing Sample Data")
            st.markdown("""
            Use the sample data already available in the system for demonstration.
            """)

            if os.path.exists("data"):
                st.success("Sample data found!")
                if st.button("Load Sample Data", type="primary"):
                    self.load_sample_data()
            else:
                st.error("Sample data not found. Please upload your own files.")

        with tab2:
            st.subheader("Upload CSV Files")
            st.markdown("""
            Upload your experiment data files. The system expects:
            - **Users files**: user_id, ts, and ampl_user_data (JSON)
            - **Messages files**: user_id and messages_count
            - **Payments files**: user_id and price_usd
            """)

            uploaded_files = st.file_uploader(
                "Choose CSV files",
                type=["csv"],
                accept_multiple_files=True,
                help="Upload all your CSV files (users, messages, payments)",
            )

            if uploaded_files:
                if st.button("Process Uploaded Files", type="primary"):
                    self.process_uploaded_files(uploaded_files)

    def process_uploaded_files(self, uploaded_files):
        try:
            temp_dir = tempfile.mkdtemp()

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Loading and processing data...")
            self.data_loader = DataLoader(temp_dir)
            self.data = self.data_loader.load_all_data()
            self.experiments = self.data_loader.get_all_experiments()

            st.session_state.data_loaded = True
            st.session_state.data = self.data
            st.session_state.experiments = self.experiments
            st.session_state.data_loader = self.data_loader

            shutil.rmtree(temp_dir)

            progress_bar.progress(1.0)
            status_text.text("Data loaded successfully!")

            st.success(
                f"Successfully loaded {len(self.data['users'])} users and found {len(self.experiments)} experiments!"
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")

    def load_sample_data(self):
        try:
            with st.spinner("Loading sample data..."):
                self.data_loader = DataLoader("data")
                self.data = self.data_loader.load_all_data()
                self.experiments = self.data_loader.get_all_experiments()

                st.session_state.data_loaded = True
                st.session_state.data = self.data
                st.session_state.experiments = self.experiments
                st.session_state.data_loader = self.data_loader

            st.success(
                f"Successfully loaded {len(self.data['users'])} users and found {len(self.experiments)} experiments!"
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

    def render_main_dashboard(self):
        self.data = st.session_state.data
        self.experiments = st.session_state.experiments
        self.data_loader = st.session_state.data_loader

        self.render_data_overview()

        self.render_experiment_analysis()

    def render_data_overview(self):
        st.header("Data Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Users", f"{len(self.data['users']):,}")

        with col2:
            st.metric("Message Records", f"{len(self.data['messages']):,}")

        with col3:
            st.metric("Payment Records", f"{len(self.data['payments']):,}")

        with col4:
            st.metric("Experiments Found", len(self.experiments))

        quality_report = self.data_loader.get_data_quality_report()

        st.subheader("Experiment Details")

        exp_data = []
        for exp_name, details in quality_report["experiment_details"].items():
            clean_name = exp_name.replace("exp_", "")
            exp_data.append(
                {
                    "Experiment": clean_name,
                    "Control Users": f"{details['control']:,}",
                    "Treatment Users": f"{details['treatment']:,}",
                    # "Not in Experiment": f"{details['not_in_experiment']:,}",
                    "Balance Ratio": f"{details['treatment'] / max(details['control'], 1):.2f}",
                }
            )

        if exp_data:
            df_exp = pd.DataFrame(exp_data)
            st.dataframe(df_exp, use_container_width=True)

    def render_experiment_analysis(self):
        st.header("Experiment Analysis")

        experiment_options = [exp.replace("exp_", "") for exp in self.experiments]

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_experiment = st.selectbox(
                "Select Experiment to Analyze",
                options=experiment_options,
                help="Choose which experiment to analyze",
            )

        with col2:
            analyze_all = st.checkbox(
                "Analyze All Experiments",
                help="Run analysis on all available experiments",
            )

        if st.button("Run Analysis", type="primary", use_container_width=True):
            if analyze_all:
                self.run_all_experiments_analysis()
            else:
                self.run_single_experiment_analysis(selected_experiment)

    def run_single_experiment_analysis(self, experiment_name):
        try:
            with st.spinner(f"Analyzing experiment: {experiment_name}..."):
                exp_data = self.data_loader.get_experiment_data(experiment_name)
                control_data = exp_data["control"]
                treatment_data = exp_data["treatment"]

                stat_tester = StatisticalTester(st.session_state.significance_level)
                stat_tester.bootstrap_iterations = st.session_state.bootstrap_iterations

                decision_engine = DecisionEngine(
                    significance_threshold=st.session_state.significance_level,
                    min_sample_size=st.session_state.min_sample_size,
                )

                test_results = stat_tester.run_all_tests(
                    control_data, treatment_data, st.session_state.metrics
                )

                quality_report = self.data_loader.get_data_quality_report()
                decision_result = decision_engine.make_decision(
                    test_results, quality_report, experiment_name
                )

                self.display_experiment_results(
                    experiment_name,
                    exp_data,
                    test_results,
                    decision_result,
                    stat_tester,
                )

        except Exception as e:
            st.error(f"Error analyzing experiment: {str(e)}")

    def run_all_experiments_analysis(self):
        st.subheader("Analyzing All Experiments")

        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, exp_name in enumerate(self.experiments):
            clean_name = exp_name.replace("exp_", "")
            status_text.text(f"Analyzing {clean_name}...")

            try:
                exp_data = self.data_loader.get_experiment_data(clean_name)
                control_data = exp_data["control"]
                treatment_data = exp_data["treatment"]

                stat_tester = StatisticalTester(st.session_state.significance_level)
                decision_engine = DecisionEngine(
                    significance_threshold=st.session_state.significance_level,
                    min_sample_size=st.session_state.min_sample_size,
                )

                test_results = stat_tester.run_all_tests(
                    control_data, treatment_data, st.session_state.metrics
                )

                quality_report = self.data_loader.get_data_quality_report()
                decision_result = decision_engine.make_decision(
                    test_results, quality_report, clean_name
                )

                results[clean_name] = {
                    "exp_data": exp_data,
                    "test_results": test_results,
                    "decision_result": decision_result,
                }

            except Exception as e:
                st.warning(f"Failed to analyze {clean_name}: {str(e)}")

            progress_bar.progress((i + 1) / len(self.experiments))

        status_text.text("Analysis complete!")

        self.display_all_experiments_summary(results)

        for exp_name, result_data in results.items():
            with st.expander(f"{exp_name} - Detailed Results"):
                self.display_experiment_results(
                    exp_name,
                    result_data["exp_data"],
                    result_data["test_results"],
                    result_data["decision_result"],
                    None,
                )

    def display_all_experiments_summary(self, results):
        st.subheader("All Experiments Summary")

        total_experiments = len(results)
        accepted = sum(
            1
            for r in results.values()
            if r["decision_result"].decision == Decision.ACCEPT
        )
        rejected = sum(
            1
            for r in results.values()
            if r["decision_result"].decision == Decision.REJECT
        )
        keep_running = sum(
            1
            for r in results.values()
            if r["decision_result"].decision == Decision.KEEP_RUNNING
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Experiments", total_experiments)

        with col2:
            st.metric("Accepted", accepted)

        with col3:
            st.metric("Rejected", rejected)

        with col4:
            st.metric("Keep Running", keep_running)

        summary_data = []
        for exp_name, result_data in results.items():
            decision = result_data["decision_result"]
            exp_data = result_data["exp_data"]

            summary_data.append(
                {
                    "Experiment": exp_name,
                    "Decision": decision.decision.value,
                    "Confidence": f"{decision.confidence:.1f}%",
                    "Control Users": f"{len(exp_data['control']):,}",
                    "Treatment Users": f"{len(exp_data['treatment']):,}",
                    "Key Finding": decision.key_findings[0]
                    if decision.key_findings
                    else "No significant findings",
                }
            )

        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)

    def display_experiment_results(
        self, experiment_name, exp_data, test_results, decision_result, stat_tester
    ):
        st.subheader(f"Results: {experiment_name}")

        decision = decision_result.decision

        message = f"**Decision: {decision.value}**\n\n**Confidence:** {decision_result.confidence:.1f}%"

        if decision == Decision.ACCEPT:
            st.success(message)
        elif decision == Decision.REJECT:
            st.error(message)
        else:
            st.warning(message)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Control Users", f"{len(exp_data['control']):,}")

        with col2:
            st.metric("Treatment Users", f"{len(exp_data['treatment']):,}")

        with col3:
            balance_ratio = len(exp_data["treatment"]) / max(
                len(exp_data["control"]), 1
            )
            st.metric("Balance Ratio", f"{balance_ratio:.2f}")

        st.subheader("Statistical Test Results")

        for metric, results in test_results.items():
            st.write(f"**{metric}**")

            result_data = []
            for result in results:
                result_data.append(
                    {
                        "Test": result.test_name,
                        "P-value": f"{result.p_value:.4f}",
                        "Effect Size": f"{result.effect_size:.4f}",
                        "Relative Diff": f"{result.relative_difference:+.2f}%",
                        "Significant": f"{result.is_significant}",
                    }
                )

            df_results = pd.DataFrame(result_data)
            st.dataframe(df_results, use_container_width=True)

        st.subheader("Visualizations")

        self.create_metric_comparison_plot(exp_data["control"], exp_data["treatment"])

        if stat_tester and stat_tester.get_last_bootstrap_distribution():
            self.create_bootstrap_plot(stat_tester, list(test_results.keys())[-1])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Key Findings")
            if decision_result.key_findings:
                for finding in decision_result.key_findings:
                    st.write(f"• {finding}")
            else:
                st.write("No significant findings detected.")

        with col2:
            st.subheader("Recommendations")
            for rec in decision_result.recommendations:
                st.write(f"• {rec}")

        if decision_result.warnings:
            st.subheader("Warnings")
            for warning in decision_result.warnings:
                st.warning(warning)

    def create_metric_comparison_plot(self, control_data, treatment_data):
        metrics = st.session_state.metrics

        fig = make_subplots(
            rows=1,
            cols=len(metrics),
            subplot_titles=metrics,
            specs=[[{"type": "box"}] * len(metrics)],
        )

        for i, metric in enumerate(metrics, 1):
            fig.add_trace(
                go.Box(
                    y=control_data[metric],
                    name="Control",
                    boxpoints="outliers",
                    marker_color="lightblue",
                    showlegend=i == 1,
                ),
                row=1,
                col=i,
            )

            fig.add_trace(
                go.Box(
                    y=treatment_data[metric],
                    name="Treatment",
                    boxpoints="outliers",
                    marker_color="lightcoral",
                    showlegend=i == 1,
                ),
                row=1,
                col=i,
            )

        fig.update_layout(title="Metric Comparison: Control vs Treatment", height=400)

        st.plotly_chart(fig, use_container_width=True)

    def create_bootstrap_plot(self, stat_tester, metric_name):
        bootstrap_data = stat_tester.get_last_bootstrap_distribution()
        if not bootstrap_data:
            return

        bootstrap_diffs, observed_diff = bootstrap_data

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=bootstrap_diffs,
                nbinsx=50,
                opacity=0.7,
                name="Bootstrap Distribution",
                marker_color="lightblue",
            )
        )

        fig.add_vline(
            x=observed_diff,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Observed: {observed_diff:.4f}",
        )

        fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.5)

        fig.update_layout(
            title=f"Bootstrap Distribution for {metric_name}",
            xaxis_title="Difference in Means",
            yaxis_title="Frequency",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)


def main():
    app = ABTestingApp()
    app.run()


if __name__ == "__main__":
    main()
