"""
Research tools for Deep Agents from Scratch.

This module provides tools for web search, content analysis, and strategic thinking
for research-oriented agents.
"""

import os
from datetime import datetime
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .state import DeepAgentState


def get_today_str():
    """Get today's date as a formatted string."""
    return datetime.now().strftime("%a %b %d, %Y")


def run_tavily_search(query: str, max_results: int = 1, topic: str = "general", include_raw_content: bool = True):
    """Run Tavily search with the given parameters."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        return client.search(query=query, max_results=max_results, topic=topic, include_raw_content=include_raw_content)
    except ImportError:
        # Mock data for demonstration
        return {
            "results": [
                {
                    "title": f"Mock Search Result for: {query}",
                    "url": "https://example.com",
                    "content": f"This is mock content for the query: {query}",
                    "raw_content": f"Raw content for {query} - this would be the full webpage content in a real search."
                }
            ]
        }


def process_search_results(search_results):
    """Process and format search results."""
    processed_results = []
    for i, result in enumerate(search_results.get("results", []), 1):
        filename = f"search_result_{i}.md"
        summary = result.get("content", "No summary available")
        raw_content = result.get("raw_content", "")
        
        processed_results.append({
            "filename": filename,
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "summary": summary,
            "raw_content": raw_content
        })
    
    return processed_results


@tool
def tavily_search(query: str, state: Annotated[DeepAgentState, InjectedState],
                  tool_call_id: Annotated[str, InjectedToolCallId],
                  max_results: Annotated[int, InjectedToolArg] = 1,
                  topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general") -> Command:
    """Search web and save detailed results to files while returning minimal context."""
    # Execute search
    search_results = run_tavily_search(query, max_results=max_results, topic=topic, include_raw_content=True)
    
    # Process and summarize results
    processed_results = process_search_results(search_results)
    
    # Save each result to a file and prepare summary
    files = state.get("files", {})
    saved_files = []
    summaries = []
    
    for i, result in enumerate(processed_results):
        filename = result['filename']
        file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""
        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")
    
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""
    
    return Command(
        update={
            "files": files,
            "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)]
        }
    )


@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making."""
    return f"Reflection recorded: {reflection}"
