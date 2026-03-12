"""
Dynamic Architecture Router for Agentic Systems.

A predictive meta-agent that routes tasks to optimal architectural topologies
(Single-Agent, Centralized MAS, or Decentralized MAS) to minimize coordination tax.
"""

from dynamic_routing.router import app as router_app

__all__ = ["router_app"]
