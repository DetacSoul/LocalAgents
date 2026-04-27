# PROJECT SPEC — LocalAgents

> **This file is the single source of truth.**  
> Paste the relevant section into whichever LLM (Claude, Grok, Gemini) you're working with.  
> Update it after every meaningful session.

---

## Goal

A collection of local AI agent systems built with LangGraph and Ollama for exploration, learning, and portfolio demonstration.

## Repo Structure

```
LocalAgents/
├── README.md                        # Repo-level overview and agent index
├── PROJECT_SPEC.md                  # ← You are here
├── DECISIONS.md                     # Session log across LLM collaborators
├── requirements.txt                 # Shared dependencies
├── personal_profile.example.py      # Profile template (safe to commit)
├── personal_profile.py              # Your real data (GITIGNORED)
├── .gitignore
├── LICENSE
│
├── shared/                          # Reusable utilities across all agents
│   ├── __init__.py
│   └── llm_helpers.py               # Retry logic, config loader, LLM factory
│
├── resume_agent/                    # Agent 1: Resume & cover letter writer
│   ├── README.md                    # Agent-specific docs
│   ├── config.yaml                  # Agent-specific settings
│   ├── main.py                      # Entrypoint
│   └── outputs/                     # Generated results (GITIGNORED)
│
├── research_system/                 # Agent 2: Research system
│   ├── README.md
│   ├── config.yaml
│   ├── main.py
│   └── outputs/
│
└── token_manager/                   # Agent 3: Token manager
    ├── README.md
    ├── config.yaml
    ├── main.py
    └── outputs/
```

## Agent Index

| Agent | Status | Description |
|-------|--------|-------------|
| `resume_agent/` | v2 working | Generates tailored resume bullets + cover letter |
| `research_system/` | In progress | Research automation system |
| `token_manager/` | In progress | Token management utility |

## Architecture Decisions

| Decision | Rationale | Date | Decided With |
|----------|-----------|------|--------------|
| Monorepo over separate repos | Easier to manage, single portfolio link | 2026-04-26 | Claude |
| LangGraph for orchestration | Explicit state graph, easy to debug | 2026-04-26 | Grok |
| Ollama for local inference | Free, private, no API keys needed | 2026-04-26 | — |
| Shared utilities in `shared/` | DRY — retry logic reused across agents | 2026-04-26 | Claude |

## Open Questions

1. Should each agent have its own venv, or one shared environment?
2. Consistent model across agents or different models per use case?
3. Add CI/linting with CodeRabbit for all agents?

## Task Queue

1. **Next:** Reorganize existing files into subdirectory structure
2. Add README for each agent
3. Add agent-level config.yaml to each agent
4. Build out research_system nodes
