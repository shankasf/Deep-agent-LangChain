#!/usr/bin/env python3
"""
Deep Agents from Scratch - Main Entry Point

This script demonstrates how to use the deep_agents_from_scratch package
to create and run various types of AI agents.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from src.deep_agents_from_scratch import (
    DeepAgentState,
    write_todos,
    read_todos,
    ls,
    read_file,
    write_file,
    tavily_search,
    think_tool
)

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate agent usage."""
    # Initialize the model
    model = init_chat_model(
        model="anthropic:claude-sonnet-4-20250514",
        temperature=0.0
    )
    
    # Create a basic agent with all tools
    tools = [
        write_todos,
        read_todos,
        ls,
        read_file,
        write_file,
        tavily_search,
        think_tool
    ]
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="You are a helpful research assistant with access to various tools.",
        state_schema=DeepAgentState
    )
    
    print("Deep Agents from Scratch - Main Entry Point")
    print("=" * 50)
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    print("\nAgent created successfully!")
    print("Use the agent by calling agent.invoke() with your messages.")

if __name__ == "__main__":
    main()
