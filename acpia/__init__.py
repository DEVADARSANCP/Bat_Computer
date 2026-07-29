"""
ACPIA — Agentic Child Protection Investigation Assistant.

An agentic investigation platform: an orchestrating LLM reasons over a case,
invokes specialised forensic agents through an MCP tool layer, accumulates
structured claims into a knowledge graph, and recommends the next investigative
action — while keeping a human in the loop and every assertion traceable to the
evidence it came from.
"""

__version__ = "0.1.0"

from .schema import (
    AgentResult,
    Claim,
    ClaimKind,
    Entity,
    EntityType,
    Evidence,
    EvidenceKind,
    EvidenceRef,
    RelationType,
    Relationship,
    RiskLevel,
)
from .store import Store

__all__ = [
    "__version__",
    "AgentResult",
    "Claim",
    "ClaimKind",
    "Entity",
    "EntityType",
    "Evidence",
    "EvidenceKind",
    "EvidenceRef",
    "RelationType",
    "Relationship",
    "RiskLevel",
    "Store",
]
