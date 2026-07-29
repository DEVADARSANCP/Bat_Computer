"""
MCP server — exposes ACPIA's forensic capabilities as standard MCP tools.

Every agent and every case tool is published over MCP stdio, so any MCP client
(Claude Desktop, an IDE, another agent) can drive an investigation using exactly
the same capabilities the internal planner uses. There is one implementation of
each tool, in `acpia.tools`; this module is only the protocol wrapper.

Run directly:

    python -m acpia.mcp_server --case CASE-001 --db acpia.db

Note on stdout: MCP speaks JSON-RPC over stdout, so nothing else may be written
there. All diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .agents import registry
from .store import Store
from .tools import ToolLayer


def build_server(tool_layer: ToolLayer) -> Server:
    server = Server("acpia")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec.get("inputSchema") or {"type": "object", "properties": {}},
            )
            for spec in tool_layer.list_tools()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        # Agents are synchronous and CPU/GPU-bound, so run them off the event
        # loop to keep the protocol responsive during long video work.
        result = await asyncio.to_thread(tool_layer.call, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server


async def _serve(case_id: str, db_path: str) -> None:
    store = Store(db_path)
    if not store.case_exists(case_id):
        store.create_case(case_id, name=f"Case {case_id}")

    tool_layer = ToolLayer(store, registry, case_id)
    server = build_server(tool_layer)

    print(
        f"[acpia] MCP server ready — case={case_id} db={db_path} "
        f"tools={len(tool_layer.list_tools())}",
        file=sys.stderr,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description="ACPIA MCP server")
    parser.add_argument("--case", default="CASE-001", help="Case id to operate on.")
    parser.add_argument("--db", default="acpia.db", help="Path to the ACPIA database.")
    args = parser.parse_args()

    try:
        asyncio.run(_serve(args.case, args.db))
    except KeyboardInterrupt:
        print("[acpia] shutting down", file=sys.stderr)


if __name__ == "__main__":
    main()
