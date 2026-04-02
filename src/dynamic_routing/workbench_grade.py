"""
WorkBench-style grading without importing vendor/src/evals/utils.py (that module
pulls optional langchain_community). Implements outcome-centric is_correct and
exact side-effect string match aligned with WorkBench.
"""

from __future__ import annotations

from dynamic_routing.workbench_env import workbench_session


def get_function_name(action: str) -> str:
    """Extract module.function from a call string like email.delete_email.func(...)."""
    return ".".join(action.split("(")[0].split(".")[0:2])


def _execute_actions_and_reset_state(actions: list[str]):
    """
    Mirror WorkBench execute_actions_and_reset_state: reset, eval each action,
    snapshot dataframes, reset again.
    """
    with workbench_session():
        from src.tools import (
            analytics,
            calendar,
            customer_relationship_manager,
            email,
            project_management,
        )

        DOMAINS = [calendar, email, analytics, project_management, customer_relationship_manager]

        for domain in DOMAINS:
            domain.reset_state()

        for action in actions:
            try:
                eval(action)
            except Exception:
                continue

        new_calendar_state = calendar.CALENDAR_EVENTS.copy()
        new_email_state = email.EMAILS.copy()
        new_analytics_state = analytics.PLOTS_DATA.copy()
        new_project_management_state = project_management.PROJECT_TASKS.copy()
        new_crm_state = customer_relationship_manager.CRM_DATA.copy()

        for domain in DOMAINS:
            domain.reset_state()

        return (
            True,
            new_calendar_state,
            new_email_state,
            new_analytics_state,
            new_project_management_state,
            new_crm_state,
        )


def is_exact_match_side_effects(predicted_actions: list[str], ground_truth_actions: list[str]) -> bool:
    """Same logic as WorkBench is_exact_match (sorted side-effect call strings)."""
    with workbench_session():
        from src.tools.toolkits import tools_with_side_effects

        tools_with_side_effects_names = [str(function.name) for function in tools_with_side_effects]
    predicted_se = [
        action
        for action in predicted_actions
        if get_function_name(action) in tools_with_side_effects_names
    ]
    predicted_se = sorted(a.lower() for a in predicted_se)
    ground_truth_se = sorted(a.lower() for a in ground_truth_actions)
    return predicted_se == ground_truth_se


def is_outcome_correct(predicted_actions: list[str], ground_truth_actions: list[str], agent_error: str) -> bool:
    """
    WorkBench outcome-centric check: execute predicted vs gold action lists on fresh
    state and compare resulting DataFrames.
    """
    if agent_error:
        return False
    (
        _,
        pred_cal,
        pred_email,
        pred_analytics,
        pred_pm,
        pred_crm,
    ) = _execute_actions_and_reset_state(predicted_actions)
    (
        _,
        gt_cal,
        gt_email,
        gt_analytics,
        gt_pm,
        gt_crm,
    ) = _execute_actions_and_reset_state(ground_truth_actions)

    def convert_strs_to_lowercase(df):
        fields_not_to_convert = ["status", "list_name", "board"]
        for col in df.columns:
            if col not in fields_not_to_convert:
                df[col] = df[col].str.lower()
        return df

    pred_cal = convert_strs_to_lowercase(pred_cal)
    pred_email = convert_strs_to_lowercase(pred_email)
    pred_analytics = convert_strs_to_lowercase(pred_analytics)
    pred_pm = convert_strs_to_lowercase(pred_pm)
    pred_crm = convert_strs_to_lowercase(pred_crm)

    gt_cal = convert_strs_to_lowercase(gt_cal.copy())
    gt_email = convert_strs_to_lowercase(gt_email.copy())
    gt_analytics = convert_strs_to_lowercase(gt_analytics.copy())
    gt_pm = convert_strs_to_lowercase(gt_pm.copy())
    gt_crm = convert_strs_to_lowercase(gt_crm.copy())

    return (
        pred_cal.equals(gt_cal)
        and pred_email.equals(gt_email)
        and pred_analytics.equals(gt_analytics)
        and pred_pm.equals(gt_pm)
        and pred_crm.equals(gt_crm)
    )


def workbench_accuracy_score(
    predicted_calls: list[str],
    gold_calls: list[str],
    agent_error: str,
) -> tuple[float, bool, bool]:
    """
    Returns (score, outcome_ok, exact_side_effect_match).
    Primary score is outcome-based (WorkBench paper); exact match is diagnostic.
    """
    outcome = is_outcome_correct(predicted_calls, gold_calls, agent_error)
    exact = is_exact_match_side_effects(predicted_calls, gold_calls)
    return (1.0 if outcome else 0.0, outcome, exact)
