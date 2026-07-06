# Multiple Agent System (MAS)

A learning project that evolves a Python and LangGraph coding assistant into a production-oriented multi-agent system.

## Current status

- Versions 1–5: graph fundamentals, state, routing, memory, and multi-agent concepts are reported complete.
- Version 6: tool calling is in progress.
- The planner/coder/reviewer graph compiles, and local tool execution is verified.
- Live Azure OpenAI execution requires a valid endpoint and API key pair.

See [AI_ENGINEERING_MASTER_ROADMAP.md](AI_ENGINEERING_MASTER_ROADMAP.md) for the detailed roadmap and verification notes.

## Setup

Use Python 3.12 because the currently installed LangChain stack warns about Python 3.14 compatibility.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` with your own credentials. The file is ignored by Git and must never be committed.

## Run

```powershell
python tool_calling_fundamentals.py
python coding_agent_v6.py
```

The fundamentals file keeps LLM-backed lessons commented out by default so they can be enabled one at a time.

## Security note

The included file tools are educational. They are not a production sandbox. Run generated code and file-writing agents only in an isolated environment.
