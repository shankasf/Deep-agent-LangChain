#!/usr/bin/env python3
"""
Deep Agents from Scratch - Lesson 3: Context Isolation with Sub-agents

This script demonstrates how to implement context isolation through sub-agent delegation.
It includes:
- Creating specialized sub-agents with focused tool sets
- Context isolation to prevent confusion and task interference
- The task() delegation tool and agent registry patterns
- Parallel execution capabilities for independent research streams

Based on notebook: 3_subagents.ipynb
"""

import os
from datetime import datetime
from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
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


# Sub-agent configuration
class SubAgent(TypedDict):
    """Configuration for a specialized sub-agent."""
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


# Task delegation tool
TASK_DESCRIPTION_PREFIX = """Delegate a task to a specialized sub-agent with isolated context. Available agents for delegation are:
{other_agents}"""

SUBAGENT_USAGE_INSTRUCTIONS = """You can delegate tasks to sub-agents.

<Task>
Your role is to coordinate research by delegating specific research tasks to sub-agents.
</Task>

<Available Tools>
1. **task(description, subagent_type)**: Delegate research tasks to specialized sub-agents
   - description: Clear, specific research question or task
   - subagent_type: Type of agent to use (e.g., "research-agent")
2. **think_tool(reflection)**: Reflect on the results of each delegated task and plan next steps.
   - reflection: Your detailed reflection on the results of the task and next steps.

**PARALLEL RESEARCH**: When you identify multiple independent research directions, make multiple **task** tool calls in a single response to enable parallel execution. Use at most {max_concurrent_research_units} parallel agents per iteration.
</Available Tools>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards focused research** - Use single agent for simple questions, multiple only when clearly beneficial or when you have multiple independent research directions based on the user's request.
- **Stop when adequate** - Don't over-research; stop when you have sufficient information
- **Limit iterations** - Stop after {max_researcher_iterations} task delegations if you haven't found adequate sources
</Hard Limits>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: "List the top 10 coffee shops in San Francisco" → Use 1 sub-agent, store in `findings_coffee_shops.md`

**Comparisons** can use a sub-agent for each element of the comparison:
- *Example*: "Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety" → Use 3 sub-agents
- Store findings in separate files: `findings_openai_safety.md`, `findings_anthropic_safety.md`, `findings_deepmind_safety.md`

**Multi-faceted research** can use parallel agents for different aspects:
- *Example*: "Research renewable energy: costs, environmental impact, and adoption rates" → Use 3 sub-agents
- Organize findings by aspect in separate files

**Important Reminders:**
- Each **task** call creates a dedicated research agent with isolated context
- Sub-agents can't see each other's work - provide complete standalone instructions
- Use clear, specific language - avoid acronyms or abbreviations in task descriptions
</Scaling Rules>"""


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context clash and confusion in complex multi-step tasks.

    Args:
        tools: List of available tools that can be assigned to sub-agents
        subagents: List of specialized sub-agent configurations
        model: The language model to use for all agents
        state_schema: The state schema (typically DeepAgentState)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """
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
            # Use specific tools if specified
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            # Default to all tools
            _tools = tools
        agents[_agent["name"]] = create_react_agent(
            model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
        )

    # Generate description of available sub-agents for the tool description
    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Delegate a task to a specialized sub-agent with isolated context.

        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.
        """
        # Validate requested agent type exists
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"

        # Get the requested sub-agent
        sub_agent = agents[subagent_type]

        # Create isolated context with only the task description
        # This is the key to context isolation - no parent history
        state["messages"] = [{"role": "user", "content": description}]

        # Execute the sub-agent in isolation
        result = sub_agent.invoke(state)

        # Return results to parent agent via Command state update
        return Command(
            update={
                "files": result.get("files", {}),  # Merge any file changes
                "messages": [
                    # Sub-agent result becomes a ToolMessage in parent context
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return task


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
    """Main function to demonstrate sub-agent delegation."""
    console.print(Panel(
        "Deep Agents from Scratch - Lesson 3: Context Isolation with Sub-agents",
        title="Starting Sub-agent Demo",
        border_style="green"
    ))
    
    # Create agent with sub-agent delegation
    console.print("\n[bold]1. Creating agent with sub-agent delegation capabilities...[/bold]")
    
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    
    # Limits
    max_concurrent_research_units = 3
    max_researcher_iterations = 3

    # Mock research instructions
    SIMPLE_RESEARCH_INSTRUCTIONS = """You are a researcher. Research the topic provided to you. IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the provided topic."""

    # Create research sub-agent
    research_sub_agent = {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
        "prompt": SIMPLE_RESEARCH_INSTRUCTIONS,
        "tools": ["web_search"],
    }

    # Tools for sub-agent
    sub_agent_tools = [web_search]

    # Create task tool to delegate tasks to sub-agents
    task_tool = _create_task_tool(
        sub_agent_tools, [research_sub_agent], model, DeepAgentState
    )

    # Tools
    delegation_tools = [task_tool]

    # Create agent with system prompt
    agent = create_react_agent(
        model,
        delegation_tools,
        prompt=SUBAGENT_USAGE_INSTRUCTIONS.format(
            max_concurrent_research_units=max_concurrent_research_units,
            max_researcher_iterations=max_researcher_iterations,
            date=datetime.now().strftime("%a %b %d, %Y"),
        ),
        state_schema=DeepAgentState,
    )

    console.print("Agent with sub-agent delegation created successfully!")
    
    # Test sub-agent delegation
    console.print("\n[bold]2. Testing sub-agent delegation with research task...[/bold]")
    
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Give me an overview of Model Context Protocol (MCP)."
        }],
    })
    
    console.print("Agent execution result:")
    format_messages(result["messages"])
    
    # Show files created by sub-agent
    console.print(f"\n[bold]Files created by sub-agent:[/bold]")
    for filename, content in result.get("files", {}).items():
        console.print(f"File {filename}: {len(content)} characters")
    
    console.print(Panel(
        "Sub-agent delegation demonstration completed successfully!",
        title="Demo Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
