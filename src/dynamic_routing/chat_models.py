"""
Multi-backend chat models for workers (ReAct) and router (structured metadata).

Environment
-----------
LLM_BACKEND (default ``vllm``)
    ``vllm`` — OpenAI-compatible server (``VLLM_WORKER_URL``, ``VLLM_WORKER_MODEL``).
    ``openai`` — ``https://api.openai.com/v1`` (or ``OPENAI_BASE_URL``) with
    ``OPENAI_API_KEY``. Model: ``OPENAI_WORKER_MODEL`` (default ``gpt-5.4-mini``).
    ``google`` — Gemini via ``langchain-google-genai``. Set ``GOOGLE_API_KEY``
    (or ``GEMINI_API_KEY``). Model: ``GOOGLE_WORKER_MODEL`` (default
    ``gemini-3.1-flash-lite-preview``).

ROUTER_LLM_BACKEND (optional)
    Overrides backend for ``get_router_chat_model`` only (same values as above).
    Router defaults: ``OPENAI_ROUTER_MODEL`` / ``GOOGLE_ROUTER_MODEL`` / ``VLLM_MODEL``.

Model IDs are documented at:
  https://developers.openai.com/api/docs/models/gpt-5.4-mini
  https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr


def llm_backend() -> str:
    return os.environ.get("LLM_BACKEND", "vllm").strip().lower()


def router_llm_backend() -> str:
    explicit = os.environ.get("ROUTER_LLM_BACKEND", "").strip().lower()
    return explicit if explicit else llm_backend()


def _openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def get_worker_chat_model(*, temperature: float = 0.1) -> BaseChatModel:
    """LLM for ReAct agents (SAS, CMAS workers, WorkBench SAS/CMAS)."""
    from langchain_openai import ChatOpenAI

    backend = llm_backend()
    if backend == "openai":
        return ChatOpenAI(
            model=os.environ.get("OPENAI_WORKER_MODEL", "gpt-5.4-mini"),
            api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
            base_url=_openai_base_url(),
            temperature=temperature,
        )
    if backend == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GOOGLE_WORKER_MODEL", "gemini-3.1-flash-lite-preview"),
            google_api_key=key or None,
            temperature=temperature,
        )
    return ChatOpenAI(
        model=os.environ.get("VLLM_WORKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "EMPTY")),
        base_url=os.environ.get("VLLM_WORKER_URL", "http://localhost:8001/v1").rstrip("/"),
        temperature=temperature,
    )


def get_router_chat_model(*, max_tokens: int = 150, temperature: float = 0.1) -> BaseChatModel:
    """LLM for routing metadata (structured JSON) or keyword-only if disabled elsewhere."""
    from langchain_openai import ChatOpenAI

    backend = router_llm_backend()
    if backend == "openai":
        model = os.environ.get("OPENAI_ROUTER_MODEL") or os.environ.get(
            "OPENAI_WORKER_MODEL", "gpt-5.4-mini"
        )
        return ChatOpenAI(
            model=model,
            api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
            base_url=_openai_base_url(),
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
    if backend == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.environ.get("GOOGLE_ROUTER_MODEL") or os.environ.get(
            "GOOGLE_WORKER_MODEL", "gemini-3.1-flash-lite-preview"
        )
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    return ChatOpenAI(
        model=os.environ.get("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "EMPTY")),
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/"),
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
