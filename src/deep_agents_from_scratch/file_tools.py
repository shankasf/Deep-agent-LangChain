"""
File management tools for Deep Agents from Scratch.

This module provides tools for managing a virtual file system in the agent state,
enabling context offloading and information organization.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .state import DeepAgentState


@tool
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""
    return list(state.get("files", {}).keys())


@tool
def read_file(file_path: str, state: Annotated[DeepAgentState, InjectedState],
              offset: int = 0, limit: int = 2000) -> str:
    """Read file content from virtual filesystem with optional offset and limit."""
    files = state.get("files", {})
    if file_path not in files:
        return f"Error: File '{file_path}' not found"
    
    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"
    
    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))
    
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
    
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")
    
    return "\n".join(result_lines)


@tool
def write_file(file_path: str, content: str, state: Annotated[DeepAgentState, InjectedState],
               tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Write content to a file in the virtual filesystem."""
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)]
        }
    )
