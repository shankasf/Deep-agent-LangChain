# Deep Agents from Scratch - Python Scripts

This repository contains Python scripts converted from the original Jupyter notebooks, demonstrating how to build advanced AI agents using LangGraph. The scripts maintain all the original logic while providing a more streamlined execution experience.

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or later
- Required API keys (see Environment Setup below)

### Installation

1. **Clone or download the scripts** to your local machine

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env with your API keys
   nano .env  # or use your preferred editor
   ```

### Environment Setup

Create a `.env` file with the following variables:

```env
# Required for model usage
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For real web search (lessons 3-4)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: For evaluation and tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

**Note:** The scripts include mock implementations for web search, so `TAVILY_API_KEY` is optional for basic functionality.

## 📚 Running the Scripts

### Option 1: Run All Lessons

Execute all lessons in sequence:

```bash
python run_all_examples.py
```

### Option 2: Run Individual Lessons

Run specific lessons:

```bash
# Lesson 0: Basic ReAct Agent
python 0_create_agent.py

# Lesson 1: Task Planning with TODO Lists  
python 1_todo.py

# Lesson 2: Context Offloading with Virtual File System
python 2_files.py

# Lesson 3: Context Isolation with Sub-agents
python 3_subagents.py

# Lesson 4: Complete Research Agent
python 4_full_agent.py
```

### Option 3: Use the Runner with Options

```bash
# Run only lesson 2
python run_all_examples.py --lesson 2

# Skip environment setup check
python run_all_examples.py --skip-setup
```

## 📖 Lesson Overview

### Lesson 0: Create React Agent (`0_create_agent.py`)
- **What you'll learn:** Basic ReAct agent creation with LangGraph
- **Key concepts:** Tool integration, state management, state injection
- **Tools:** Calculator tool with state tracking
- **Duration:** ~2-3 minutes

### Lesson 1: Task Planning (`1_todo.py`)
- **What you'll learn:** Structured task planning using TODO lists
- **Key concepts:** Task tracking, progress monitoring, context management
- **Tools:** `write_todos()`, `read_todos()`, mock web search
- **Duration:** ~3-4 minutes

### Lesson 2: Virtual File System (`2_files.py`)
- **What you'll learn:** Context offloading through virtual file system
- **Key concepts:** File operations, context persistence, token efficiency
- **Tools:** `ls()`, `read_file()`, `write_file()`, mock web search
- **Duration:** ~3-4 minutes

### Lesson 3: Sub-agent Delegation (`3_subagents.py`)
- **What you'll learn:** Context isolation through sub-agent delegation
- **Key concepts:** Specialized agents, context isolation, parallel execution
- **Tools:** `task()` delegation tool, agent registry
- **Duration:** ~4-5 minutes

### Lesson 4: Complete Research Agent (`4_full_agent.py`)
- **What you'll learn:** Integration of all techniques into production-ready agent
- **Key concepts:** Real web search, content summarization, strategic thinking
- **Tools:** All previous tools plus `tavily_search()`, `think_tool()`
- **Duration:** ~5-6 minutes

## 🔧 Technical Details

### Architecture

Each script demonstrates progressive agent development:

1. **Basic Agent** → Tool integration and state management
2. **TODO Management** → Task planning and progress tracking  
3. **File System** → Context offloading and information persistence
4. **Sub-agents** → Context isolation and specialized delegation
5. **Complete Agent** → Production-ready research capabilities

### Key Features

- **Mock Implementations:** Web search and external APIs are mocked for reliable execution
- **Rich Output:** Beautiful console output with colored panels and progress indicators
- **Error Handling:** Comprehensive error handling and user-friendly messages
- **Modular Design:** Each lesson builds upon previous concepts
- **State Management:** Proper state injection and updates using LangGraph patterns

### Dependencies

- `langgraph>=0.6.4` - Core agent framework
- `langchain>=0.3.0` - LLM integration
- `rich>=14.0.0` - Beautiful console output
- `pydantic>=2.0.0` - Data validation
- `python-dotenv>=1.0.0` - Environment variable management

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: Missing required environment variable
   ```
   **Solution:** Ensure your `.env` file contains the required API keys

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain'
   ```
   **Solution:** Install dependencies with `pip install -r requirements.txt`

3. **Timeout Errors**
   ```
   Lesson X timed out after 5 minutes
   ```
   **Solution:** Check your internet connection and API key validity

4. **Permission Errors**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution:** Ensure you have write permissions in the current directory

### Getting Help

- Check the console output for detailed error messages
- Verify your API keys are correct and have sufficient credits
- Ensure you're using Python 3.11 or later
- Try running individual lessons to isolate issues

## 🎯 Learning Outcomes

After completing all lessons, you'll understand:

- How to build ReAct agents with LangGraph
- Task planning and progress tracking techniques
- Context offloading strategies for long-running agents
- Sub-agent delegation patterns for complex workflows
- Production-ready agent architecture patterns
- Best practices for agent state management
- Context engineering for AI agents

## 📝 Notes

- The scripts use mock implementations for external APIs to ensure reliable execution
- All original notebook logic has been preserved in the conversion
- The scripts are designed to run independently without requiring the original notebook environment
- Each script includes comprehensive error handling and user feedback

## 🤝 Contributing

If you find issues or want to improve the scripts:

1. Check the original notebooks for reference
2. Ensure all logic is preserved during modifications
3. Test thoroughly with different API configurations
4. Maintain the educational progression of the lessons

---

**Happy Learning!** 🚀

These scripts provide a hands-on way to learn advanced agent development techniques. Each lesson builds upon the previous one, creating a comprehensive learning path from basic agents to production-ready research systems.
