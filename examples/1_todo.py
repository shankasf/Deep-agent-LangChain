#!/usr/bin/env python3
"""
Deep Agents from Scratch - Lesson 1: Task Planning with TODO Lists

This script demonstrates how to implement structured task planning using TODO lists.
It includes:
- Task tracking with status management (pending/in_progress/completed)
- Progress monitoring and context management
- The write_todos() tool for organizing complex multi-step workflows
- Best practices for maintaining focus and preventing task drift

Based on notebook: 1_todo.ipynb
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
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """
    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence.

    Used as a reducer function for the files field in agent state,
    allowing incremental updates to the virtual file system.

    Args:
        left: Left side dictionary (existing files)
        right: Right side dictionary (new/updated files)

    Returns:
        Merged dictionary with right values overriding left values
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


from langgraph.prebuilt.chat_agent_executor import AgentState


class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    """
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


# Tool descriptions
WRITE_TODOS_DESCRIPTION = """Create and manage structured task lists for tracking progress through complex workflows.

## When to Use
- Multi-step or non-trivial tasks requiring coordination
- When user provides multiple tasks or explicitly requests todo list  
- Avoid for single, trivial actions

## Structure
- Maintain one list containing multiple todo objects (content, status, id)
- Use clear, actionable content descriptions
- Status must be: pending, in_progress, or completed

## Best Practices  
- Only one in_progress task at a time
- Mark completed immediately when task is fully done
- Always send the full updated list when making changes
- Prune irrelevant items to keep list focused

## Progress Updates
- Call TodoWrite again to change task status or edit content
- Reflect real-time progress; don't batch completions  
- If blocked, keep in_progress and add new task describing blocker

## Parameters
- todos: List of TODO items with content and status fields

## Returns
Updates agent state with new todo list."""

TODO_USAGE_INSTRUCTIONS = """Based upon the user's request:
1. Use the write_todos tool to create TODO at the start of a user request, per the tool description.
2. After you accomplish a TODO, use the read_todos to read the TODOs in order to remind yourself of the plan. 
3. Reflect on what you've done and the TODO.
4. Mark you task as completed, and proceed to the next TODO.
5. Continue this process until you have completed all TODOs.

IMPORTANT: Always create a research plan of TODOs and conduct research following the above guidelines for ANY user request.
IMPORTANT: Aim to batch research tasks into a *single TODO* in order to minimize the number of TODOs you have to keep track of.
"""


# TODO Tools
@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Formatted string representation of the current TODO list
    """
    todos = state.get("todos", [])
    if not todos:
        return "No todos currently in the list."

    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

    return result.strip()


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
    """Main function to demonstrate TODO list management."""
    console.print(Panel(
        "Deep Agents from Scratch - Lesson 1: Task Planning with TODO Lists",
        title="Starting TODO Demo",
        border_style="green"
    ))
    
    # Create agent with TODO tools
    console.print("\n[bold]1. Creating agent with TODO management tools...[/bold]")
    
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    tools = [write_todos, web_search, read_todos]

    # Add mock research instructions
    SIMPLE_RESEARCH_INSTRUCTIONS = """IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question."""

    # Create agent
    agent = create_react_agent(
        model,
        tools,
        prompt=TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SIMPLE_RESEARCH_INSTRUCTIONS,
        state_schema=DeepAgentState,
    )

    console.print("Agent created successfully!")
    
    # Test TODO management
    console.print("\n[bold]2. Testing TODO list management with research task...[/bold]")
    
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Give me a short summary of the Model Context Protocol (MCP)."
        }],
        "todos": [],
    })
    
    console.print("Agent execution result:")
    format_messages(result["messages"])
    
    # Show final state
    console.print(f"\n[bold]Final TODO list:[/bold] {result.get('todos', [])}")
    
    console.print(Panel(
        "TODO list management demonstration completed successfully!",
        title="Demo Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
