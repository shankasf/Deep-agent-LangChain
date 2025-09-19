# Installation Guide

## Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)
- API keys for Anthropic, OpenAI, and optionally Tavily

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/shankasf/Deep-agent-LangChain.git
   cd Deep-agent-LangChain
   ```

2. **Create virtual environment**
   ```bash
   python -m venv deep_agent_env
   # Windows:
   deep_agent_env\Scripts\activate
   # Linux/Mac:
   source deep_agent_env/bin/activate
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

4. **Configure environment**
   ```bash
   cp docs/env_example.txt .env
   # Edit .env with your API keys
   ```

## API Keys Required

- `ANTHROPIC_API_KEY`: Required for Claude models
- `OPENAI_API_KEY`: Optional, for summarization
- `TAVILY_API_KEY`: Optional, for web search
- `LANGSMITH_API_KEY`: Optional, for tracing

## Quick Start

```python
from src.deep_agents_from_scratch import DeepAgentState, write_todos, read_todos
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# Initialize model
model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")

# Create agent
agent = create_react_agent(model, [write_todos, read_todos], state_schema=DeepAgentState)

# Use agent
result = agent.invoke({"messages": [("user", "Create a TODO list for my project")]})
```
