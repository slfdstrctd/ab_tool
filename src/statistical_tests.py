import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    test_name: str
    metric: str
    control_mean: float
    treatment_mean: float
    absolute_difference: float
    relative_difference: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    sample_size_control: int
    sample_size_treatment: int
    is_significant: bool
    significance_level: float


class StatisticalTester:
    def __init__(self, significance_level: float = 0.12):
        self.significance_level = significance_level
        self.bootstrap_iterations = 10000
        self.confidence_level = 1 - significance_level

    def run_all_tests(
        self,
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metrics: List[str],
    ) -> Dict[str, List[TestResult]]:
        results = {}

        for metric in metrics:
            logger.info(f"Running tests for metric: {metric}")
            metric_results = []

            control_values = control_data[metric].values
            treatment_values = treatment_data[metric].values

            bootstrap_result = self.bootstrap_test(
                control_values, treatment_values, metric
            )
            metric_results.append(bootstrap_result)

            ttest_result = self.t_test(control_values, treatment_values, metric)
            metric_results.append(ttest_result)

            mw_result = self.mann_whitney_test(control_values, treatment_values, metric)
            metric_results.append(mw_result)

            results[metric] = metric_results

        return results

    def bootstrap_test(
        self, control: np.ndarray, treatment: np.ndarray, metric_name: str
    ) -> TestResult:
        logger.debug(f"Running bootstrap test for {metric_name}")

        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        observed_diff = treatment_mean - control_mean

        bootstrap_diffs = []

        for _ in tqdm(
            range(self.bootstrap_iterations),
            desc=f"Bootstrap {metric_name}",
            leave=False,
        ):
            control_sample = np.random.choice(control, size=len(control), replace=True)
            treatment_sample = np.random.choice(
                treatment, size=len(treatment), replace=True
            )

            diff = np.mean(treatment_sample) - np.mean(control_sample)
            bootstrap_diffs.append(diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        if observed_diff >= 0:
            p_value = 2 * np.mean(bootstrap_diffs <= -abs(observed_diff))
        else:
            p_value = 2 * np.mean(bootstrap_diffs >= abs(observed_diff))

        ci_lower = np.percentile(bootstrap_diffs, (self.significance_level / 2) * 100)
        ci_upper = np.percentile(
            bootstrap_diffs, (1 - self.significance_level / 2) * 100
        )

        pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        self.last_bootstrap_distribution = bootstrap_diffs
        self.last_observed_diff = observed_diff

        return TestResult(
            test_name="Bootstrap",
            metric=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_difference=observed_diff,
            relative_difference=(observed_diff / control_mean * 100)
            if control_mean != 0
            else np.inf,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            sample_size_control=len(control),
            sample_size_treatment=len(treatment),
            is_significant=p_value < self.significance_level,
            significance_level=self.significance_level,
        )

    def t_test(
        self, control: np.ndarray, treatment: np.ndarray, metric_name: str
    ) -> TestResult:
        logger.debug(f"Running t-test for {metric_name}")

        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        observed_diff = treatment_mean - control_mean

        se_diff = np.sqrt(
            np.var(control) / len(control) + np.var(treatment) / len(treatment)
        )
        t_critical = stats.t.ppf(
            1 - self.significance_level / 2, len(control) + len(treatment) - 2
        )
        ci_lower = observed_diff - t_critical * se_diff
        ci_upper = observed_diff + t_critical * se_diff

        pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        return TestResult(
            test_name="T-Test",
            metric=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_difference=observed_diff,
            relative_difference=(observed_diff / control_mean * 100)
            if control_mean != 0
            else np.inf,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            sample_size_control=len(control),
            sample_size_treatment=len(treatment),
            is_significant=p_value < self.significance_level,
            significance_level=self.significance_level,
        )

    def mann_whitney_test(
        self, control: np.ndarray, treatment: np.ndarray, metric_name: str
    ) -> TestResult:
        logger.debug(f"Running Mann-Whitney U test for {metric_name}")

        u_stat, p_value = stats.mannwhitneyu(
            treatment, control, alternative="two-sided"
        )

        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        observed_diff = treatment_mean - control_mean

        n1, n2 = len(control), len(treatment)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        bootstrap_diffs = []
        for _ in range(1000):
            control_sample = np.random.choice(control, size=len(control), replace=True)
            treatment_sample = np.random.choice(
                treatment, size=len(treatment), replace=True
            )
            diff = np.mean(treatment_sample) - np.mean(control_sample)
            bootstrap_diffs.append(diff)

        ci_lower = np.percentile(bootstrap_diffs, (self.significance_level / 2) * 100)
        ci_upper = np.percentile(
            bootstrap_diffs, (1 - self.significance_level / 2) * 100
        )

        return TestResult(
            test_name="Mann-Whitney U",
            metric=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_difference=observed_diff,
            relative_difference=(observed_diff / control_mean * 100)
            if control_mean != 0
            else np.inf,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            sample_size_control=len(control),
            sample_size_treatment=len(treatment),
            is_significant=p_value < self.significance_level,
            significance_level=self.significance_level,
        )

    def test_multiple_effect_sizes(
        self,
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metric: str,
        effect_sizes: List[float],
    ) -> Dict[float, TestResult]:
        results = {}

        control_values = control_data[metric].values
        treatment_values = treatment_data[metric].values
        control_mean = np.mean(control_values)

        for effect_size in effect_sizes:
            logger.info(f"Testing for {effect_size}% effect size")

            target_mean = control_mean * (1 + effect_size / 100)
            current_mean = np.mean(treatment_values)

            if current_mean != 0:
                scaled_treatment = treatment_values * (target_mean / current_mean)
            else:
                scaled_treatment = treatment_values + target_mean

            result = self.bootstrap_test(
                control_values, scaled_treatment, f"{metric}_{effect_size}%"
            )
            results[effect_size] = result

        return results

    def check_sample_size_adequacy(
        self,
        control_size: int,
        treatment_size: int,
        effect_size: float = 0.2,
        power: float = 0.8,
    ) -> Dict[str, Any]:
        from statsmodels.stats.power import ttest_ind_solve_power

        required_n = ttest_ind_solve_power(
            effect_size=effect_size,
            alpha=self.significance_level,
            power=power,
            ratio=treatment_size / control_size if control_size > 0 else 1,
            alternative="two-sided",
        )

        min_current = min(control_size, treatment_size)
        is_adequate = min_current >= required_n

        return {
            "control_size": control_size,
            "treatment_size": treatment_size,
            "required_size_per_group": int(np.ceil(required_n)),
            "is_adequate": is_adequate,
            "power_achieved": ttest_ind_solve_power(
                effect_size=effect_size,
                nobs1=min_current,
                alpha=self.significance_level,
                ratio=treatment_size / control_size if control_size > 0 else 1,
                alternative="two-sided",
            )
            if min_current > 0
            else 0,
        }

    def get_last_bootstrap_distribution(self) -> Optional[Tuple[np.ndarray, float]]:
        if hasattr(self, "last_bootstrap_distribution"):
            return self.last_bootstrap_distribution, self.last_observed_diff
        return None
