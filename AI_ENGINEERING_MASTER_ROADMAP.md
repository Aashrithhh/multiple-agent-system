# AI Engineering Master Roadmap

## End Goal

Become an AI Engineer who can design, build, deploy, debug, and explain production-grade AI agents. The learning project is one continuously evolving application: a **Production Coding Agent**.

For every new concept:

1. Explain it simply.
2. Explain why it exists and how it works internally.
3. Show realistic company use cases.
4. Implement it from scratch with Python and LangGraph.
5. Discuss common mistakes and interview questions.
6. Refactor the Production Coding Agent to include it.

## Verified Progress — July 6, 2026

### Reported as completed

- Version 1 — Graph Fundamentals
- Version 2 — State Management
- Version 3 — Conditional Routing
- Version 4 — Memory
- Version 5 — Multi-Agent Systems

The current MAS directory does not contain the Version 1–5 source files, tests, notes, or commit history, so these topics cannot be independently verified from this workspace. Their status is retained as **reported completed**.

### Currently in progress: Version 6 — Tool Calling

Evidence in this directory:

- `tool_calling_fundamentals.py` covers tool schemas, binding tools, the manual execution cycle, `ToolNode`, and a complete tool loop.
- `coding_agent_v6.py` defines planner, coder, and reviewer roles; executable tools; tool nodes; and conditional routing.
- `requirements.txt` lists the LangGraph, LangChain, OpenAI, and dotenv dependencies.

Verification results:

- Both Python files pass syntax compilation.
- The basic `add_numbers` tool executes successfully.
- Full tool-calling behavior is **not yet verified or complete**:
  - Azure OpenAI and the `gpt-4.1-mini` deployment are configured through an ignored local `.env` file, but Azure rejects the supplied key/endpoint pair with HTTP 401.
  - The standalone `ToolNode` lesson now runs through a compiled graph so LangGraph 1.x can inject runtime context.
  - The missing `end_or_revise` node was added so the Version 6 graph can compile.
  - The environment uses Python 3.14, and LangChain emits a warning that its Pydantic v1 compatibility layer does not support Python 3.14 or newer.
  - The files contain mojibake characters such as `â†’` and `ðŸ...`, indicating an encoding conversion problem.
  - No automated test suite exists in the workspace.

**Verified status:** Versions 1–5 are reported complete but lack local evidence; Version 6 has been started and is not complete.

## Project Evolution

### Version 1 — Basic Graph

Learn nodes, edges, `START`, and `END`. Build a simple Coding Assistant.

### Version 2 — State Management

Add shared graph state, message history, and conversation flow. Evolve it into a state-aware Coding Agent.

### Version 3 — Conditional Routing

Add dynamic routing, decision-making, and branching logic. Evolve it into a Smart Coding Agent.

### Version 4 — Memory

Add conversation memory, persistent state, and context awareness. Evolve it into a Coding Agent that remembers prior interactions.

### Version 5 — Multi-Agent System

Split responsibilities across Planner, Coding, and Reviewer agents. Evolve it into a collaborative AI coding team.

### Version 6 — Tool Calling

Add Python execution, calculator, web search, file access, terminal, Git, and documentation-search tools. Evolve it into a Coding Agent that can act through external tools.

Completion criteria:

- Use a LangChain-supported Python version (preferably Python 3.12) in a virtual environment.
- Run all fundamentals lessons successfully.
- Replace or correct the rejected Azure API key/endpoint pair and verify a live model response.
- Run the complete coding agent on a small task.
- Add automated tests for tools and routing.
- Correct the source-file encoding.

### Version 7 — Human in the Loop

Add approvals, interrupts, human feedback, and execution resumption. Evolve it into a safe Production Coding Agent.

### Version 8 — Checkpointing

Add state persistence, resumption, and interruption recovery. Evolve it into a fault-tolerant Coding Agent.

### Version 9 — Streaming

Add streaming responses, live progress updates, and appropriate intermediate visibility. Evolve it into an interactive AI Coding Assistant.

### Version 10 — Error Handling

Add retry logic, tool-failure recovery, exception handling, and fallback paths. Evolve it into a reliable Production Agent.

### Version 11 — Observability

Add logging, tracing, metrics, debugging, and performance monitoring. Evolve it into an enterprise-grade AI Agent.

### Version 12 — Deployment

Deploy with FastAPI, Docker, environment-based configuration, and a cloud platform. Evolve it into a production-ready AI application.

## Broader AI Engineering Roadmap

After the LangGraph application is complete, continue with:

- Advanced prompt engineering and LLM APIs
- RAG, embeddings, vector databases, and retrieval optimization
- Agent memory and Model Context Protocol (MCP)
- OpenAI Agents SDK
- API design, authentication, and databases
- FastAPI, Docker, CI/CD, and cloud deployment
- Monitoring, security, and performance optimization

## Final Outcomes

- Build production-grade AI agents from scratch.
- Design multi-agent systems with robust state and memory.
- Integrate external tools, APIs, and RAG systems.
- Debug, monitor, secure, and deploy AI applications.
- Explain architectural decisions and trade-offs in interviews.
