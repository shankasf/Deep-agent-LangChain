#!/usr/bin/env python3
"""
Deep Agents from Scratch - Lesson 0: Create React Agent

This script demonstrates how to create a basic ReAct agent using LangGraph's
create_react_agent abstraction. It includes:
- Basic calculator tool
- Custom state management
- State injection and updates
- Hooks and structured responses

Based on notebook: 0_create_agent.ipynb
"""

import os
from typing import Annotated, List, Literal, Union

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


@tool
def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: Union[int, float],
    b: Union[int, float],
) -> Union[int, float]:
    """Define a two-input calculator tool.

    Arg:
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
        a (float or int): The first number.
        b (float or int): The second number.
        
    Returns:
        result (float or int): the result of the operation
    Example
        Divide: result   = a / b
        Subtract: result = a - b
    """
    if operation == 'divide' and b == 0:
        return {"error": "Division by zero is not allowed."}

    # Perform calculation
    if operation == 'add':
        result = a + b
    elif operation == 'subtract':
        result = a - b
    elif operation == 'multiply':
        result = a * b
    elif operation == 'divide':
        result = a / b
    else: 
        result = "unknown operation"
    return result


def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right


def main():
    """Main function to demonstrate basic ReAct agent creation."""
    console.print(Panel(
        "Deep Agents from Scratch - Lesson 0: Create React Agent",
        title="Starting Agent Demo",
        border_style="green"
    ))
    
    # Create basic agent with calculator tool
    console.print("\n[bold]1. Creating basic ReAct agent with calculator tool...[/bold]")
    
    SYSTEM_PROMPT = "You are a helpful arithmetic assistant who is an expert at using a calculator."
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", temperature=0.0)
    tools = [calculator]

    # Create agent
    agent = create_react_agent(
        model,
        tools,
        prompt=SYSTEM_PROMPT,
    ).with_config({"recursion_limit": 20})

    console.print(f"Agent type: {type(agent)}")
    
    # Test basic calculator
    console.print("\n[bold]2. Testing basic calculator...[/bold]")
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "What is 3.1 * 4.2?"}]
    })
    
    console.print("Result:")
    format_messages(result1["messages"])
    
    # Create agent with custom state
    console.print("\n[bold]3. Creating agent with custom state for operation tracking...[/bold]")
    
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from typing_extensions import TypedDict
    
    class CalcState(AgentState):
        """Graph State with operation tracking."""
        ops: Annotated[List[str], reduce_list]

    @tool
    def calculator_wstate(
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: Union[int, float],
        b: Union[int, float],
        state: Annotated[CalcState, InjectedState],   # not sent to LLM
        tool_call_id: Annotated[str, InjectedToolCallId] # not sent to LLM
    ) -> Union[int, float]:
        """Define a two-input calculator tool with state tracking.

        Arg:
            operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
            a (float or int): The first number.
            b (float or int): The second number.
            
        Returns:
            result (float or int): the result of the operation
        Example
            Divide: result   = a / b
            Subtract: result = a - b
        """
        if operation == 'divide' and b == 0:
            return {"error": "Division by zero is not allowed."}

        # Perform calculation
        if operation == 'add':
            result = a + b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'divide':
            result = a / b
        else: 
            result = "unknown operation"
        
        ops = [f"({operation}, {a}, {b}),"]
        return Command(
            update={
                "ops": ops,
                "messages": [
                    ToolMessage(f"{result}", tool_call_id=tool_call_id)
                ],
            }
        )

    # Create agent with state tracking
    agent_with_state = create_react_agent(
        model,
        [calculator_wstate],
        prompt=SYSTEM_PROMPT,
        state_schema=CalcState,
    ).with_config({"recursion_limit": 20})

    # Test state tracking
    console.print("\n[bold]4. Testing calculator with state tracking...[/bold]")
    result2 = agent_with_state.invoke({
        "messages": [{"role": "user", "content": "What is 3.1 * 4.2?"}]
    })
    
    console.print("Result with state:")
    format_messages(result2["messages"])
    console.print(f"Operations tracked: {result2.get('ops', [])}")
    
    # Test multiple operations
    console.print("\n[bold]5. Testing multiple operations...[/bold]")
    result3 = agent_with_state.invoke({
        "messages": [{"role": "user", "content": "What is 3.1 * 4.2 + 5.5 * 6.5?"}]
    })
    
    console.print("Multiple operations result:")
    format_messages(result3["messages"])
    console.print(f"All operations tracked: {result3.get('ops', [])}")
    
    console.print(Panel(
        "Basic ReAct agent demonstration completed successfully!",
        title="Demo Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()