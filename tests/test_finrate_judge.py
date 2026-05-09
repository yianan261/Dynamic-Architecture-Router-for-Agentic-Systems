"""Tests for structured Fin-RATE answer judging."""

from dynamic_routing.finrate_runner import judge_finrate_answer, judge_finrate_answer_detail


def test_finrate_local_judge_detail(monkeypatch) -> None:
    monkeypatch.setenv("FINRATE_JUDGE_BACKEND", "local")
    monkeypatch.delenv("FINRATE_USE_GPT_JUDGE", raising=False)

    detail = judge_finrate_answer_detail(
        "What net revenue did ExampleCo report?",
        "ExampleCo reported net revenue of 50 million USD.",
        "ExampleCo reported net revenue of 50 million USD.",
    )

    assert detail["score"] in (0.0, 0.5, 1.0)
    assert detail["score"] == 1.0
    assert detail["judge_backend"] == "local"
    assert detail["judge_model"] == "local_overlap"
    assert detail["reason"]


def test_finrate_gpt_falls_back_without_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("FINRATE_JUDGE_BACKEND", "gpt")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    detail = judge_finrate_answer_detail(
        "What net revenue did ExampleCo report?",
        "ExampleCo reported net revenue of 50 million USD.",
        "ExampleCo reported net revenue of 50 million USD.",
    )

    assert detail["score"] == 1.0
    assert detail["judge_backend"] == "gpt_fallback_local"
    assert detail["judge_model"] == "local_overlap"


def test_finrate_legacy_gpt_flag_maps_to_gpt_fallback(monkeypatch) -> None:
    monkeypatch.delenv("FINRATE_JUDGE_BACKEND", raising=False)
    monkeypatch.setenv("FINRATE_USE_GPT_JUDGE", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    score, backend = judge_finrate_answer(
        "What net revenue did ExampleCo report?",
        "ExampleCo reported net revenue of 50 million USD.",
        "ExampleCo reported net revenue of 50 million USD.",
    )

    assert score == 1.0
    assert backend == "gpt_fallback_local"
