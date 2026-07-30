"""
Agent package.

Importing this module registers every available agent into the shared
`registry`. The orchestrator and the MCP server both read from that registry,
so adding a capability to ACPIA is exactly two steps: write the subclass, add
an import here.
"""

from .base import Agent, AgentContext, AgentRegistry, registry
from .anpr_agent import AnprAgent
from .appearance_agent import AppearanceAgent
from .chat_agent import ChatAgent
from .face_agent import FaceAgent
from .metadata_agent import MetadataAgent
from .ocr_agent import OcrAgent
from .video_agent import VideoAgent
from .vision_agent import VisionAgent

__all__ = [
    "Agent",
    "AgentContext",
    "AgentRegistry",
    "registry",
    "AnprAgent",
    "AppearanceAgent",
    "ChatAgent",
    "FaceAgent",
    "MetadataAgent",
    "OcrAgent",
    "VideoAgent",
    "VisionAgent",
]
