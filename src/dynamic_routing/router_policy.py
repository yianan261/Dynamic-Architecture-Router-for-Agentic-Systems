"""
Optional learned router: multinomial logistic regression on tabular features.

Set ``ROUTER_LEARNED_MODEL_PATH`` to a ``joblib`` bundle produced by
``scripts/train_router_from_regret.py`` (dict with keys ``model``, ``feature_names``,
``classes``). If unset or load fails, ``learned_route_destination`` returns ``None``
and threshold routing is used.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from dynamic_routing.state import RouterState

GraphNode = Literal["single_agent_node", "centralized_mas_node", "decentralized_mas_node"]

_DISPLAY_TO_NODE: dict[str, GraphNode] = {
    "Single-Agent System": "single_agent_node",
    "Centralized MAS": "centralized_mas_node",
    "Decentralized MAS": "decentralized_mas_node",
}

_NODE_ORDER: list[GraphNode] = [
    "single_agent_node",
    "centralized_mas_node",
    "decentralized_mas_node",
]

_MODEL_CACHE: Any = None
_MODEL_PATH: str | None = None


def _feature_vector(state: RouterState) -> list[float]:
    def _f(key: str, default: float = 0.0) -> float:
        v = state.get(key)  # type: ignore[arg-type]
        if v is None:
            return default
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return [
        _f("estimated_sequential_depth", 3.0),
        _f("parallelization_factor", 0.3),
        _f("estimated_tool_count", 4.0),
        _f("num_subgoals", 1.0),
        _f("entity_count", 1.0),
        _f("constraint_tightness", 2.0),
        _f("open_endedness", 2.0),
        _f("aggregation_required", 0.0),
        _f("expected_retrieval_fanout", 1.0),
        _f("domain_span", 1.0),
        _f("expected_context_expansion", 2.0),
        _f("final_synthesis_complexity", 2.0),
        _f("cross_branch_dependency", 1.0),
        _f("communication_load_estimate", 1.0),
    ]


def _load_bundle():
    global _MODEL_CACHE, _MODEL_PATH
    path = os.environ.get("ROUTER_LEARNED_MODEL_PATH", "").strip()
    if not path:
        _MODEL_CACHE = None
        _MODEL_PATH = None
        return None
    if _MODEL_CACHE is not None and _MODEL_PATH == path:
        return _MODEL_CACHE
    try:
        import joblib
    except ImportError:
        logging.warning("router_policy: joblib not installed; cannot load learned router")
        _MODEL_CACHE = None
        _MODEL_PATH = None
        return None
    try:
        bundle = joblib.load(path)
    except Exception as e:
        logging.warning("router_policy: failed to load %s: %s", path, str(e)[:120])
        _MODEL_CACHE = None
        _MODEL_PATH = None
        return None
    _MODEL_CACHE = bundle
    _MODEL_PATH = path
    return bundle


def learned_route_destination(state: RouterState) -> GraphNode | None:
    """Return a graph node name if a learned model confidently applies; else ``None``."""
    bundle = _load_bundle()
    if not bundle or not isinstance(bundle, dict):
        return None
    model = bundle.get("model")
    feats = bundle.get("feature_names")
    if model is None or not feats:
        return None
    x = _feature_vector(state)
    try:
        import numpy as np

        X = np.array([x], dtype=float)
    except ImportError:
        X = [x]

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf_thr = float(os.environ.get("ROUTER_LEARNED_MIN_CONF", "0") or 0.0)
            if conf_thr > 0 and float(max(proba)) < conf_thr:
                return None
        pred = model.predict(X)[0]
    except Exception as e:
        logging.warning("router_policy: predict failed: %s", str(e)[:120])
        return None

    classes = bundle.get("classes") or _NODE_ORDER
    if isinstance(pred, str):
        disp = pred
    else:
        try:
            idx = int(pred)
            disp = str(classes[idx])
        except (IndexError, TypeError, ValueError):
            disp = str(pred)
    node = _DISPLAY_TO_NODE.get(disp)
    if node is None:
        node = pred if pred in _NODE_ORDER else None  # type: ignore[assignment]
    return node


def predict_learned_display_architecture(state: RouterState) -> str | None:
    """Human-readable architecture for benchmark JSON, or None to fall back."""
    node = learned_route_destination(state)
    if node is None:
        return None
    inv = {v: k for k, v in _DISPLAY_TO_NODE.items()}
    return inv.get(node)
