"""
Task delegation tools for Deep Agents from Scratch.

This module provides tools for creating and managing sub-agents with isolated
contexts for specialized task execution.
"""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command

from .state import DeepAgentState


class SubAgent(TypedDict):
    """Configuration for a specialized sub-agent."""
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents."""
    # Create agent registry
    agents = {}
    
    # Build tool name mapping for selective tool assignment
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_
    
    # Create specialized sub-agents based on configurations
    for _agent in subagents:
        if "tools" in _agent:
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            _tools = tools
        agents[_agent["name"]] = create_react_agent(
            model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
        )
    
    other_agents_string = ", ".join([f"`{k}`" for k in agents])
    
    @tool(description=f"Delegate a task to a specialized sub-agent with isolated context. Available sub-agents: {other_agents_string}")
    def task(description: str, subagent_type: str,
             state: Annotated[DeepAgentState, InjectedState],
             tool_call_id: Annotated[str, InjectedToolCallId]):
        """Delegate a task to a specialized sub-agent with isolated context."""
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
        
        sub_agent = agents[subagent_type]
        
        # Create isolated context with only the task description
        state["messages"] = [{"role": "user", "content": description}]
        
        # Execute the sub-agent in isolation
        result = sub_agent.invoke(state)
        
        # Return results to parent agent via Command state update
        return Command(
            update={
                "files": result.get("files", {}),
                "messages": [
                    ToolMessage(result["messages"][-1].content, tool_call_id=tool_call_id)
                ],
            }
        )
    
    return task
