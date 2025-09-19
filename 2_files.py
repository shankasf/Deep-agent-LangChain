#!/usr/bin/env python3
"""
Deep Agents from Scratch - Lesson 2: Context Offloading with Virtual File System

This script demonstrates how to implement a virtual file system stored in agent state
for context offloading. It includes:
- File operations: ls(), read_file(), write_file()
- Context management through information persistence
- Enabling agent "memory" across conversation turns
- Reducing token usage by storing detailed information in files

Based on notebook: 2_files.ipynb
"""

import os
from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()


def clean_content(content):
    """Clean content to remove problematic Unicode characters for Windows console."""
    import re
    # Remove common problematic Unicode characters
    content = re.sub(r'[\u23f0-\u23ff]', '', content)  # Clock symbols
    content = re.sub(r'[\u2190-\u2199]', '', content)  # Arrow symbols
    content = re.sub(r'[\u2600-\u26ff]', '', content)  # Miscellaneous symbols
    content = re.sub(r'[\u2700-\u27bf]', '', content)  # Dingbats
    content = re.sub(r'[\u1f300-\u1f9ff]', '', content)  # Emoji and pictographs
    return content

def format_messages(messages):
    """Format and display a list of messages with Rich formatting."""
    for m in messages:
        msg_type = m.__class__.__name__.replace("Message", "")
        content = clean_content(str(m.content))

        if msg_type == "Human":
            console.print(Panel(content, title="Human", border_style="blue"))
        elif msg_type == "Ai":
            console.print(Panel(content, title="Assistant", border_style="green"))
        elif msg_type == "Tool":
            console.print(Panel(content, title="Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"{msg_type}", border_style="white"))


# State definition
class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows."""
    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence."""
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


from langgraph.prebuilt.chat_agent_executor import AgentState


class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system."""
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


# File tool descriptions
LS_DESCRIPTION = """List all files in the virtual filesystem stored in agent state.

Shows what files currently exist in agent memory. Use this to orient yourself before other file operations and maintain awareness of your file organization.

No parameters required - simply call ls() to see all available files."""

READ_FILE_DESCRIPTION = """Read content from a file in the virtual filesystem with optional pagination.

This tool returns file content with line numbers (like `cat -n`) and supports reading large files in chunks to avoid context overflow.

Parameters:
- file_path (required): Path to the file you want to read
- offset (optional, default=0): Line number to start reading from  
- limit (optional, default=2000): Maximum number of lines to read

Essential before making any edits to understand existing content. Always read a file before editing it."""

WRITE_FILE_DESCRIPTION = """Create a new file or completely overwrite an existing file in the virtual filesystem.

This tool creates new files or replaces entire file contents. Use for initial file creation or complete rewrites. Files are stored persistently in agent state.

Parameters:
- file_path (required): Path where the file should be created/overwritten
- content (required): The complete content to write to the file

Important: This replaces the entire file content. Use edit_file for partial modifications."""

FILE_USAGE_INSTRUCTIONS = """You have access to a virtual file system to help you retain and save context.

## Workflow Process
1. **Orient**: Use ls() to see existing files before starting work
2. **Save**: Use write_file() to store the user's request so that we can keep it for later 
3. **Research**: Proceed with research. The search tool will write files.  
4. **Read**: Once you are satisfied with the collected sources, read the files and use them to answer the user's question directly.
"""


# File Tools
@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""
    return list(state.get("files", {}).keys())


@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content from virtual filesystem with optional offset and limit.

    Args:
        file_path: Path to the file to read
        state: Agent state containing virtual filesystem (injected in tool node)
        offset: Line number to start reading from (default: 0)
        limit: Maximum number of lines to read (default: 2000)

    Returns:
        Formatted file content with line numbers, or error message if file not found
    """
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


@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write content to a file in the virtual filesystem.

    Args:
        file_path: Path where the file should be created/updated
        content: Content to write to the file
        state: Agent state containing virtual filesystem (injected in tool node)
        tool_call_id: Tool call identifier for message response (injected in tool node)

    Returns:
        Command to update agent state with new file content
    """
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


# Mock search tool for demonstration
@tool(parse_docstring=True)
def web_search(query: str):
    """Search the web for information on a specific topic.

    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
               information you're looking for.

    Returns:
        Search results from search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    return """The Model Context Protocol (MCP) is an open standard protocol developed 
by Anthropic to enable seamless integration between AI models and external systems like 
tools, databases, and other services. It acts as a standardized communication layer, 
allowing AI models to access and utilize data from various sources in a consistent and 
efficient manner. Essentially, MCP simplifies the process of connecting AI assistants 
to external services by providing a unified language for data exchange."""


def main():
    """Main function to demonstrate virtual file system."""
    console.print(Panel(
        "Deep Agents from Scratch - Lesson 2: Context Offloading with Virtual File System",
        title="Starting File System Demo",
        border_style="green"
    ))
    
    # Create agent with file system tools
    console.print("\n[bold]1. Creating agent with virtual file system tools...[/bold]")
    
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    tools = [ls, read_file, write_file, web_search]

    # Add mock research instructions
    SIMPLE_RESEARCH_INSTRUCTIONS = """IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question."""

    # Full prompt
    INSTRUCTIONS = (
        FILE_USAGE_INSTRUCTIONS + "\n\n" + "=" * 80 + "\n\n" + SIMPLE_RESEARCH_INSTRUCTIONS
    )

    # Create agent
    agent = create_react_agent(
        model, tools, prompt=INSTRUCTIONS, state_schema=DeepAgentState
    )

    console.print("Agent created successfully!")
    
    # Test file system
    console.print("\n[bold]2. Testing virtual file system with research task...[/bold]")
    
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Give me an overview of Model Context Protocol (MCP)."
        }],
        "files": {},
    })
    
    console.print("Agent execution result:")
    format_messages(result["messages"])
    
    # Show files created
    console.print(f"\n[bold]Files created in virtual filesystem:[/bold]")
    for filename, content in result.get("files", {}).items():
        console.print(f"File {filename}: {len(content)} characters")
        # Show first few lines of content
        lines = content.split('\n')[:3]
        for line in lines:
            console.print(f"   {line}")
        if len(content.split('\n')) > 3:
            console.print("   ...")
    
    console.print(Panel(
        "Virtual file system demonstration completed successfully!",
        title="Demo Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
