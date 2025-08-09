# AdaptiveAgent

Self-improving ReAct agent that iteratively optimizes its system prompt through evaluation and feedback.

## Overview

AdaptiveAgent runs a ReAct agent, evaluates its performance across three dimensions (response quality, reasoning, prompt effectiveness), then generates feedback to improve the system prompt. This cycle repeats until the agent meets performance criteria or reaches max attempts.

## Installation

```bash
poetry install
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
# or create .env file with OPENAI_API_KEY=your-key-here
```

## Usage

### Basic Setup

```python
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from main import AdaptiveAgent, get_location_info

model = ChatOpenAI(model="gpt-4")
tools = [get_location_info]  # Your tools here
agent = AdaptiveAgent(model, tools, MemorySaver())
```

### Tune Method

```python
result = await agent.tune(
    initial_prompt="You are a helpful assistant",
    task="Compare weather in NYC vs SF",
    eval_system_prompt="Evaluation criteria...",
    expected_react_final_response="Expected output format...",
    max_attempts=3,
    config={"configurable": {"thread_id": "session-1"}}
)
```

**Parameters:**
- `initial_prompt`: Starting system prompt
- `task`: Task for the agent to perform
- `eval_system_prompt`: Criteria/guidebook for how the system prompt should be evaluated
- `expected_react_final_response`: Expected output format/content
- `max_attempts`: Maximum optimization iterations (default: 1)
- `config`: LangGraph configuration with thread_id for persistence

**Returns:** Dictionary with scores, prompts, and final conversation

### Run Example

```bash
python main.py
```
