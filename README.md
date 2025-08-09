# AdaptiveAgent
An Extension to create_react_agent from LangGraph

## What it does
- **Adaptive tuning**: Iteratively runs a ReAct agent, evaluates Response / Chain-of-Thought / Prompt, then optimizes the system prompt.
- **Rich node visualization**: Color-coded node execution via a minimal decorator.

## Files
```
AdaptiveAgent/
  README.md        Minimal docs and usage
  main.py          ReAct run → evaluate → feedback → optimize loop with Rich visualization
  pyproject.toml   Dependencies
  LICENSE          License
```

## Usage
- Set required keys in `.env` as needed: `OPENAI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`.

### Console run
```bash
python main.py
```

## Notes
- Output is printed with Rich in `main.py`.
- Node execution is styled with a lightweight decorator for better traceability.
- Node internals avoid panels; only decorator start/end lines appear per node.
- Node input type hints simplified to avoid runtime introspection issues.
