"""
TODO management tools for Deep Agents from Scratch.

This module provides tools for creating, reading, and updating TODO lists
in the agent state for task planning and workflow management.
"""

from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .state import DeepAgentState, Todo


@tool
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Create or update the agent's TODO list for task planning and tracking."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)]
        }
    )


@tool
def read_todos(state: Annotated[DeepAgentState, InjectedState]) -> str:
    """Read the current TODO list from agent state."""
    todos = state.get("todos", [])
    if not todos:
        return "No todos found. Use write_todos to create a task list."
    
    todo_list = []
    for i, todo in enumerate(todos, 1):
        status_icon = "⏳" if todo["status"] == "pending" else "🔄" if todo["status"] == "in_progress" else "✅"
        todo_list.append(f"{i}. {status_icon} {todo['content']} ({todo['status']})")
    
    return "Current TODO List:\n" + "\n".join(todo_list)
