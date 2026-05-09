"""Tests for BrowseComp fixture scoring modes."""

from dynamic_routing.browsecomp_runner import judge_browsecomp_answer


def test_browsecomp_local_judge_exact(monkeypatch) -> None:
    monkeypatch.setenv("BROWSECOMP_JUDGE_BACKEND", "local")

    score, backend = judge_browsecomp_answer(
        "What was Acme revenue?",
        "120 million USD",
        "120 million USD",
    )

    assert score == 1.0
    assert backend == "local_exact"


def test_browsecomp_local_judge_contains_gold(monkeypatch) -> None:
    monkeypatch.setenv("BROWSECOMP_JUDGE_BACKEND", "local")

    score, backend = judge_browsecomp_answer(
        "What was Acme revenue?",
        "The filing says Acme reported 120 million USD in revenue.",
        "120 million USD",
    )

    assert score == 1.0
    assert backend == "local_contains_gold"


def test_browsecomp_local_judge_overlap_partial(monkeypatch) -> None:
    monkeypatch.setenv("BROWSECOMP_JUDGE_BACKEND", "local")

    score, backend = judge_browsecomp_answer(
        "Who became CEO?",
        "Jane became CEO after the merger.",
        "Jane Rivera",
    )

    assert 0.0 < score < 1.0
    assert backend == "local_overlap"


def test_browsecomp_auto_uses_decisive_local_before_llm(monkeypatch) -> None:
    monkeypatch.setenv("BROWSECOMP_JUDGE_BACKEND", "auto")
    monkeypatch.setattr(
        "dynamic_routing.browsecomp_runner._llm_judge_score",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")),
    )

    score, backend = judge_browsecomp_answer(
        "Where is Acme headquartered?",
        "Acme Corp is headquartered in Austin Texas.",
        "Austin, Texas",
    )

    assert score == 1.0
    assert backend == "local_contains_gold"


def test_browsecomp_auto_falls_back_when_uncertain_llm_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("BROWSECOMP_JUDGE_BACKEND", "auto")
    monkeypatch.setattr("dynamic_routing.browsecomp_runner._llm_judge_score", lambda *_args: None)

    score, backend = judge_browsecomp_answer(
        "Who became CEO?",
        "Jane became CEO after the merger.",
        "Jane Rivera",
    )

    assert 0.0 < score < 1.0
    assert backend == "local_overlap_uncertain_no_llm"


def test_browsecomp_llm_mode_falls_back_to_local(monkeypatch) -> None:
    monkeypatch.setenv("BROWSECOMP_JUDGE_BACKEND", "llm")
    monkeypatch.setattr("dynamic_routing.browsecomp_runner._llm_judge_score", lambda *_args: None)

    score, backend = judge_browsecomp_answer(
        "What was Acme revenue?",
        "120 million USD",
        "120 million USD",
    )

    assert score == 1.0
    assert backend == "local_exact_fallback"
