import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class Decision(Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    KEEP_RUNNING = "KEEP RUNNING"


@dataclass
class DecisionResult:
    decision: Decision
    confidence: float
    reasoning: List[str]
    key_findings: List[str]
    warnings: List[str]
    recommendations: List[str]


class DecisionEngine:
    def __init__(
        self,
        significance_threshold: float = 0.12,
        min_sample_size: int = 100,
        min_effect_size: float = 0.01,
        balance_threshold: float = 0.4,
    ):
        self.significance_threshold = significance_threshold
        self.min_sample_size = min_sample_size
        self.min_effect_size = min_effect_size
        self.balance_threshold = balance_threshold

    def make_decision(
        self,
        test_results: Dict[str, List],
        data_quality: Dict[str, Any],
        experiment_name: str,
    ) -> DecisionResult:
        logger.info(f"Making decision for experiment: {experiment_name}")

        reasoning = []
        key_findings = []
        warnings = []
        recommendations = []

        quality_check = self._check_data_quality(data_quality, experiment_name)
        if not quality_check["is_valid"]:
            warnings.extend(quality_check["warnings"])
            if quality_check["is_critical"]:
                return DecisionResult(
                    decision=Decision.REJECT,
                    confidence=90.0,
                    reasoning=["Critical data quality issues detected"],
                    key_findings=[],
                    warnings=warnings,
                    recommendations=[
                        "Fix data quality issues before re-running experiment"
                    ],
                )

        sample_check = self._check_sample_sizes(data_quality, experiment_name)
        if not sample_check["is_adequate"]:
            warnings.append(sample_check["warning"])
            if sample_check["is_critical"]:
                return DecisionResult(
                    decision=Decision.KEEP_RUNNING,
                    confidence=80.0,
                    reasoning=["Insufficient sample size for reliable decision"],
                    key_findings=[],
                    warnings=warnings,
                    recommendations=[
                        f"Continue until at least {sample_check['required_size']} users per group"
                    ],
                )

        stat_analysis = self._analyze_statistical_results(test_results)

        consistency_check = self._check_consistency(stat_analysis)

        practical_check = self._check_practical_significance(stat_analysis)

        decision, confidence = self._make_final_decision(
            stat_analysis, consistency_check, practical_check
        )

        key_findings = self._compile_key_findings(stat_analysis, test_results)
        reasoning = self._compile_reasoning(
            stat_analysis, consistency_check, practical_check, decision
        )
        recommendations = self._compile_recommendations(
            decision, stat_analysis, consistency_check
        )

        if consistency_check["has_conflicts"]:
            warnings.append("Conflicting results between metrics detected")

        return DecisionResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            key_findings=key_findings,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _check_data_quality(
        self, data_quality: Dict[str, Any], experiment_name: str
    ) -> Dict[str, Any]:
        warnings = []
        is_critical = False

        exp_key = experiment_name
        if f"exp_{experiment_name}" in data_quality.get("experiment_details", {}):
            exp_key = f"exp_{experiment_name}"
        elif "exp_" + experiment_name not in data_quality.get("experiment_details", {}):
            warnings.append(
                f"Experiment 'exp_{experiment_name}' not found in data {data_quality}"
            )
            is_critical = True
            return {"is_valid": False, "warnings": warnings, "is_critical": is_critical}

        exp_details = data_quality["experiment_details"][exp_key]

        not_in_exp = exp_details.get("not_in_experiment", 0)
        total_in_exp = exp_details.get("control", 0) + exp_details.get("treatment", 0)

        if not_in_exp > total_in_exp * 0.1:
            warnings.append(f"{not_in_exp} users not assigned to experiment groups")

        if data_quality.get("total_messages", 0) == 0:
            warnings.append("No message data available")

        if data_quality.get("total_payments", 0) == 0:
            warnings.append("No payment data available")

        return {
            "is_valid": len(warnings) == 0 or not is_critical,
            "warnings": warnings,
            "is_critical": is_critical,
        }

    def _check_sample_sizes(
        self, data_quality: Dict[str, Any], experiment_name: str
    ) -> Dict[str, Any]:
        exp_key = experiment_name
        if f"exp_{experiment_name}" in data_quality.get("experiment_details", {}):
            exp_key = f"exp_{experiment_name}"
        exp_details = data_quality["experiment_details"][exp_key]
        control_size = exp_details.get("control", 0)
        treatment_size = exp_details.get("treatment", 0)

        min_size = min(control_size, treatment_size)

        if min_size < self.min_sample_size:
            return {
                "is_adequate": False,
                "is_critical": True,
                "warning": f"Sample size too small: {min_size} < {self.min_sample_size}",
                "required_size": self.min_sample_size,
            }

        if control_size > 0:
            imbalance_ratio = abs(treatment_size - control_size) / control_size
            if imbalance_ratio > self.balance_threshold:
                return {
                    "is_adequate": True,
                    "is_critical": False,
                    "warning": f"Groups are imbalanced: {imbalance_ratio:.1%} difference",
                    "required_size": min_size,
                }

        return {
            "is_adequate": True,
            "is_critical": False,
            "warning": None,
            "required_size": min_size,
        }

    def _analyze_statistical_results(
        self, test_results: Dict[str, List]
    ) -> Dict[str, Any]:
        analysis = {
            "significant_results": [],
            "non_significant_results": [],
            "positive_effects": [],
            "negative_effects": [],
            "effect_sizes": {},
            "best_p_value": 1.0,
            "metrics_summary": {},
        }

        for metric, results in test_results.items():
            metric_summary = {
                "significant_tests": 0,
                "all_positive": True,
                "all_negative": True,
                "max_effect_size": 0,
                "min_p_value": 1.0,
            }

            for result in results:
                if result.p_value < metric_summary["min_p_value"]:
                    metric_summary["min_p_value"] = result.p_value
                    analysis["best_p_value"] = min(
                        analysis["best_p_value"], result.p_value
                    )

                if result.is_significant:
                    analysis["significant_results"].append((metric, result))
                    metric_summary["significant_tests"] += 1
                else:
                    analysis["non_significant_results"].append((metric, result))

                if result.relative_difference > 0:
                    analysis["positive_effects"].append((metric, result))
                    metric_summary["all_negative"] = False
                elif result.relative_difference < 0:
                    analysis["negative_effects"].append((metric, result))
                    metric_summary["all_positive"] = False

                metric_summary["max_effect_size"] = max(
                    metric_summary["max_effect_size"], abs(result.effect_size)
                )

            analysis["metrics_summary"][metric] = metric_summary
            analysis["effect_sizes"][metric] = metric_summary["max_effect_size"]

        return analysis

    def _check_consistency(self, stat_analysis: Dict[str, Any]) -> Dict[str, Any]:
        consistency = {
            "has_conflicts": False,
            "conflict_details": [],
            "consistency_score": 0.0,
        }

        has_positive = len(stat_analysis["positive_effects"]) > 0
        has_negative = len(stat_analysis["negative_effects"]) > 0

        if has_positive and has_negative:
            consistency["has_conflicts"] = True
            consistency["conflict_details"].append(
                "Some metrics show positive effects while others show negative"
            )

        total_results = len(stat_analysis["significant_results"]) + len(
            stat_analysis["non_significant_results"]
        )

        if total_results > 0:
            if has_positive and not has_negative:
                consistency["consistency_score"] = (
                    len(stat_analysis["significant_results"]) / total_results
                )
            elif has_negative and not has_positive:
                consistency["consistency_score"] = (
                    -len(stat_analysis["significant_results"]) / total_results
                )
            else:
                consistency["consistency_score"] = 0.5

        return consistency

    def _check_practical_significance(
        self, stat_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        practical = {
            "has_practical_significance": False,
            "large_effects": [],
            "small_effects": [],
        }

        for metric, effect_size in stat_analysis["effect_sizes"].items():
            if effect_size > 0.5:
                practical["large_effects"].append(metric)
                practical["has_practical_significance"] = True
            elif effect_size > 0.2:
                practical["has_practical_significance"] = True
            elif effect_size < 0.1:
                practical["small_effects"].append(metric)

        return practical

    def _make_final_decision(
        self,
        stat_analysis: Dict[str, Any],
        consistency_check: Dict[str, Any],
        practical_check: Dict[str, Any],
    ) -> Tuple[Decision, float]:
        confidence = 50.0

        has_significant = len(stat_analysis["significant_results"]) > 0

        total_tests = len(stat_analysis["significant_results"]) + len(
            stat_analysis["non_significant_results"]
        )
        sig_ratio = (
            len(stat_analysis["significant_results"]) / total_tests
            if total_tests > 0
            else 0
        )

        if has_significant:
            confidence += sig_ratio * 30

            if not consistency_check["has_conflicts"]:
                confidence += 10

                if consistency_check["consistency_score"] > 0:
                    if practical_check["has_practical_significance"]:
                        confidence += 10
                        return Decision.ACCEPT, min(confidence, 95.0)
                    else:
                        if stat_analysis["best_p_value"] < 0.05:
                            return Decision.ACCEPT, min(confidence, 85.0)
                        else:
                            return Decision.KEEP_RUNNING, confidence

                elif consistency_check["consistency_score"] < 0:
                    return Decision.REJECT, min(confidence, 90.0)

            else:
                confidence -= 20

                pos_count = len(stat_analysis["positive_effects"])
                neg_count = len(stat_analysis["negative_effects"])

                if pos_count > neg_count * 2:
                    return Decision.ACCEPT, max(confidence, 60.0)
                elif neg_count > pos_count * 2:
                    return Decision.REJECT, max(confidence, 60.0)
                else:
                    return Decision.KEEP_RUNNING, max(confidence, 50.0)

        else:
            if total_tests >= 3:
                confidence = 70.0

                all_small = all(
                    es < 0.1 for es in stat_analysis["effect_sizes"].values()
                )
                if all_small:
                    return Decision.REJECT, confidence
                else:
                    return Decision.KEEP_RUNNING, confidence - 10
            else:
                return Decision.KEEP_RUNNING, 50.0

    def _compile_key_findings(
        self, stat_analysis: Dict[str, Any], test_results: Dict[str, List]
    ) -> List[str]:
        findings = []

        for metric, result in stat_analysis["significant_results"][:3]:
            findings.append(
                f"{metric}: {result.relative_difference:+.1f}% "
                f"(p={result.p_value:.3f}, {result.test_name})"
            )

        for metric, effect_size in stat_analysis["effect_sizes"].items():
            if effect_size > 0.5:
                findings.append(
                    f"Large effect size detected for {metric}: {effect_size:.2f}"
                )

        if stat_analysis["best_p_value"] < 1.0:
            findings.append(
                f"Strongest evidence: p={stat_analysis['best_p_value']:.4f}"
            )

        return findings

    def _compile_reasoning(
        self,
        stat_analysis: Dict[str, Any],
        consistency_check: Dict[str, Any],
        practical_check: Dict[str, Any],
        decision: Decision,
    ) -> List[str]:
        reasoning = []

        sig_count = len(stat_analysis["significant_results"])
        total_count = sig_count + len(stat_analysis["non_significant_results"])

        if sig_count > 0:
            reasoning.append(
                f"{sig_count}/{total_count} tests showed statistical significance"
            )
        else:
            reasoning.append("No tests showed statistical significance")

        if consistency_check["has_conflicts"]:
            reasoning.append("Results show conflicting directions across metrics")
        else:
            reasoning.append("Results are consistent across all metrics")

        if practical_check["has_practical_significance"]:
            reasoning.append("Effects are large enough to be practically meaningful")
        else:
            reasoning.append("Effect sizes are small")

        if decision == Decision.ACCEPT:
            reasoning.append("Evidence supports positive impact of the treatment")
        elif decision == Decision.REJECT:
            if len(stat_analysis["negative_effects"]) > 0:
                reasoning.append("Evidence suggests negative impact of the treatment")
            else:
                reasoning.append("No meaningful improvement detected")
        else:
            reasoning.append("More data needed for confident decision")

        return reasoning

    def _compile_recommendations(
        self,
        decision: Decision,
        stat_analysis: Dict[str, Any],
        consistency_check: Dict[str, Any],
    ) -> List[str]:
        recommendations = []

        if decision == Decision.ACCEPT:
            recommendations.append("Roll out the experiment to all users")
            recommendations.append("Monitor metrics post-rollout to confirm results")

        elif decision == Decision.REJECT:
            recommendations.append("Do not roll out this experiment")
            if len(stat_analysis["negative_effects"]) > 0:
                recommendations.append(
                    "Investigate why the treatment had negative effects"
                )
            recommendations.append(
                "Consider alternative approaches to achieve the goal"
            )

        else:
            recommendations.append("Continue collecting data")

            current_best_p = stat_analysis["best_p_value"]
            if 0.12 < current_best_p < 0.3:
                recommendations.append(
                    "Consider increasing sample size by 50% for more conclusive results"
                )
            elif current_best_p >= 0.3:
                recommendations.append(
                    "Consider stopping if no improvement after doubling sample size"
                )

            if consistency_check["has_conflicts"]:
                recommendations.append(
                    "Investigate conflicting metric results before making final decision"
                )

        return recommendations
