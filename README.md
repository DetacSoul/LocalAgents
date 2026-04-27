# 🤖 LocalAgents

A collection of AI agent systems built with [LangGraph](https://github.com/langchain-ai/langgraph) and local inference via [Ollama](https://ollama.com). Each subdirectory is a self-contained agent project.

## Agents

| Agent | Description | Status |
|-------|-------------|--------|
| [`resume_agent/`](resume_agent/) | Generates tailored resume bullets and cover letters for a target company/role | ✅ Working |
| [`research_system/`](research_system/) | Automated research and synthesis pipeline | 🚧 In Progress |
| [`token_manager/`](token_manager/) | Token usage tracking and management | 🚧 In Progress |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A model pulled: `ollama pull qwen2.5:14b`

### Setup

```bash
git clone https://github.com/DetacSoul/LocalAgents.git
cd LocalAgents

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running an Agent

Each agent has its own directory. Navigate to it and run:

```bash
cd resume_agent
python main.py
```

See each agent's own `README.md` for specific instructions.

### Personal Profile (for agents that need it)

Some agents (like `resume_agent`) use a personal profile. Set it up once at the repo root:

```bash
cp personal_profile.example.py personal_profile.py
# Edit personal_profile.py with your info — it's gitignored
```

## Project Structure

```
LocalAgents/
├── shared/              # Reusable utilities (retry logic, config loader)
├── resume_agent/        # Each agent in its own directory
├── research_system/
├── token_manager/
├── personal_profile.example.py
└── requirements.txt
```

## Built With

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration
- [Ollama](https://ollama.com) — Local LLM inference
- [LangChain](https://github.com/langchain-ai/langchain) — LLM abstractions

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
