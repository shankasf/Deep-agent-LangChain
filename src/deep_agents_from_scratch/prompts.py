"""
Prompt templates and system instructions for Deep Agents from Scratch.

This module contains all the prompt templates and system instructions used
throughout the agent system for consistent behavior and communication.
"""

# System prompts for different agent types
BASIC_AGENT_PROMPT = """You are a helpful assistant with access to tools. Use the tools to help answer questions and complete tasks."""

PLANNING_AGENT_PROMPT = """You are a helpful assistant with access to tools for task planning and management. You can create, read, and update TODO lists to help organize complex workflows."""

CONTEXT_AGENT_PROMPT = """You are a helpful assistant with access to tools for file management and context offloading. You can read, write, and list files to manage information efficiently."""

DELEGATION_AGENT_PROMPT = """You are a helpful assistant with access to tools for task delegation and sub-agent management. You can create specialized sub-agents for specific tasks."""

RESEARCH_AGENT_PROMPT = """You are a helpful research assistant with access to tools for web search, content analysis, and strategic thinking. You can search for information, analyze content, and provide comprehensive research reports."""

# Tool descriptions
CALCULATOR_DESCRIPTION = """Define a two-input calculator tool."""
TODO_DESCRIPTION = """Create or update the agent's TODO list for task planning and tracking."""
FILE_DESCRIPTION = """Manage files in the virtual filesystem for context offloading."""
SEARCH_DESCRIPTION = """Search web and save detailed results to files while returning minimal context."""
THINK_DESCRIPTION = """Tool for strategic reflection on research progress and decision-making."""

# Task delegation descriptions
TASK_DESCRIPTION_PREFIX = """Delegate a task to a specialized sub-agent with isolated context. Available sub-agents: {other_agents}"""

# Research workflow instructions
RESEARCH_WORKFLOW = """
1. Create a comprehensive TODO list for the research task
2. Search for relevant information using web search tools
3. Store findings in the virtual file system
4. Analyze and synthesize information
5. Generate a comprehensive research report
6. Update TODO list with completed tasks
"""
