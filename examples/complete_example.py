#!/usr/bin/env python3
"""
Deep Agents from Scratch - Complete Example

This script demonstrates all the capabilities of the deep_agents_from_scratch package
with the exact same logic as the original notebooks, but organized in a clean package structure.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from src.deep_agents_from_scratch import (
    DeepAgentState,
    write_todos,
    read_todos,
    ls,
    read_file,
    write_file,
    tavily_search,
    think_tool,
    _create_task_tool
)

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


def lesson_0_basic_agent():
    """Lesson 0: Basic ReAct Agent with calculator tool."""
    console.print(Panel("Lesson 0: Basic ReAct Agent", title="Deep Agents", border_style="cyan"))
    
    # Basic calculator tool (preserving original logic)
    @tool
    def calculator(operation: Literal["add","subtract","multiply","divide"],
                   a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Define a two-input calculator tool."""
        if operation == 'divide' and b == 0:
            return {"error": "Division by zero is not allowed."}
        
        if operation == 'add': result = a + b
        elif operation == 'subtract': result = a - b
        elif operation == 'multiply': result = a * b
        elif operation == 'divide': result = a / b
        else: result = "unknown operation"
        return result

    # State-aware calculator (preserving original logic)
    @tool
    def calculator_wstate(
        operation: Literal["add","subtract","multiply","divide"],
        a: Union[int, float], b: Union[int, float],
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Calculator with state tracking for operation history."""
        if operation == 'add': result = a + b
        elif operation == 'subtract': result = a - b
        elif operation == 'multiply': result = a * b
        elif operation == 'divide': result = a / b
        else: result = "unknown operation"
        
        ops = [f"({operation}, {a}, {b}),"]
        
        return Command(
            update={
                "ops": ops,
                "messages": [ToolMessage(f"{result}", tool_call_id=tool_call_id)]
            }
        )

    # Initialize model and create agent
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    agent = create_react_agent(model, [calculator, calculator_wstate], state_schema=DeepAgentState)
    
    # Test the agent
    result = agent.invoke({"messages": [("user", "What is 3.1 * 4.2?")]})
    format_messages(result["messages"])


def lesson_1_planning_agent():
    """Lesson 1: Planning Agent with TODO management."""
    console.print(Panel("Lesson 1: Planning Agent", title="Deep Agents", border_style="cyan"))
    
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    agent = create_react_agent(model, [write_todos, read_todos], state_schema=DeepAgentState)
    
    # Test TODO management
    result = agent.invoke({
        "messages": [("user", "Create a TODO list for researching AI safety. Include: 1) Search for recent papers, 2) Analyze key findings, 3) Write summary report")]
    })
    format_messages(result["messages"])


def lesson_2_context_agent():
    """Lesson 2: Context Management Agent with file system."""
    console.print(Panel("Lesson 2: Context Management Agent", title="Deep Agents", border_style="cyan"))
    
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    agent = create_react_agent(model, [ls, read_file, write_file], state_schema=DeepAgentState)
    
    # Test file management
    result = agent.invoke({
        "messages": [("user", "Write a file called 'research_notes.txt' with content about AI safety research, then list all files")]
    })
    format_messages(result["messages"])


def lesson_3_delegation_agent():
    """Lesson 3: Delegation Agent with sub-agents."""
    console.print(Panel("Lesson 3: Delegation Agent", title="Deep Agents", border_style="cyan"))
    
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    
    # Define sub-agents (preserving original logic)
    subagents = [
        {
            "name": "researcher",
            "description": "Specialized in research and information gathering",
            "prompt": "You are a research specialist. Focus on gathering and analyzing information.",
            "tools": ["tavily_search", "write_file", "read_file"]
        },
        {
            "name": "analyst", 
            "description": "Specialized in data analysis and synthesis",
            "prompt": "You are an analysis specialist. Focus on analyzing data and drawing insights.",
            "tools": ["read_file", "write_file", "think_tool"]
        }
    ]
    
    # Create task delegation tool
    task_tool = _create_task_tool([tavily_search, write_file, read_file, think_tool], subagents, model, DeepAgentState)
    
    agent = create_react_agent(model, [task_tool], state_schema=DeepAgentState)
    
    # Test delegation
    result = agent.invoke({
        "messages": [("user", "Use the researcher sub-agent to search for information about machine learning safety")]
    })
    format_messages(result["messages"])


def lesson_4_research_agent():
    """Lesson 4: Complete Research Agent with all capabilities."""
    console.print(Panel("Lesson 4: Complete Research Agent", title="Deep Agents", border_style="cyan"))
    
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    
    # All tools for complete research agent
    tools = [write_todos, read_todos, ls, read_file, write_file, tavily_search, think_tool]
    
    agent = create_react_agent(
        model, 
        tools, 
        prompt="You are a comprehensive research assistant with access to web search, file management, and strategic thinking tools.",
        state_schema=DeepAgentState
    )
    
    # Test complete research workflow
    result = agent.invoke({
        "messages": [("user", "Research the latest developments in AI safety. Create a TODO list, search for recent information, and write a comprehensive report.")]
    })
    format_messages(result["messages"])


def main():
    """Run all lessons to demonstrate the complete agent system."""
    console.print(Panel("Deep Agents from Scratch - Complete Example", title="Deep Agents", border_style="bold cyan"))
    
    try:
        lesson_0_basic_agent()
        console.print("\n" + "="*80 + "\n")
        
        lesson_1_planning_agent()
        console.print("\n" + "="*80 + "\n")
        
        lesson_2_context_agent()
        console.print("\n" + "="*80 + "\n")
        
        lesson_3_delegation_agent()
        console.print("\n" + "="*80 + "\n")
        
        lesson_4_research_agent()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure you have set up your .env file with API keys[/yellow]")


if __name__ == "__main__":
    main()
