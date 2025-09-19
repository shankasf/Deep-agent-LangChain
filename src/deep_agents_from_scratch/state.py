"""
State management for Deep Agents from Scratch.

This module defines the core state classes and reducer functions used throughout
the agent system for managing persistent state across agent executions.
"""

from typing import Annotated, NotRequired, TypedDict

from langgraph.types import AgentState


class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows."""
    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """Safely merge file dictionaries with right-side precedence."""
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


class DeepAgentState(AgentState):
    """Extended agent state with custom fields for complex workflows."""
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
