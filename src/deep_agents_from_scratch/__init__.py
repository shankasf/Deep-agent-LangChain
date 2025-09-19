"""
Deep Agents from Scratch - Advanced AI Agent System

A comprehensive framework for building sophisticated AI agents using LangGraph and LangChain.
This package provides the core components for creating ReAct agents, planning agents,
context management agents, delegation agents, and production research agents.
"""

__version__ = "1.0.0"
__author__ = "Deep Agents Project"
__description__ = "Advanced AI Agent System with LangGraph and LangChain"

from .state import DeepAgentState, file_reducer
from .prompts import *
from .todo_tools import write_todos, read_todos
from .file_tools import ls, read_file, write_file
from .task_tool import _create_task_tool
from .research_tools import tavily_search, think_tool

__all__ = [
    "DeepAgentState",
    "file_reducer", 
    "write_todos",
    "read_todos",
    "ls",
    "read_file", 
    "write_file",
    "_create_task_tool",
    "tavily_search",
    "think_tool"
]
