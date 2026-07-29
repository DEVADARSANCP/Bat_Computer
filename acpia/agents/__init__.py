"""
Agent package.

Importing this module registers every available agent into the shared
`registry`. The orchestrator and the MCP server both read from that registry,
so adding a capability to ACPIA is exactly two steps: write the subclass, add
an import here.
"""

from .base import Agent, AgentContext, AgentRegistry, registry
from .metadata_agent import MetadataAgent

__all__ = [
    "Agent",
    "AgentContext",
    "AgentRegistry",
    "registry",
    "MetadataAgent",
]
