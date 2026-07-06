"""
=============================================================================
PRODUCTION CODING AGENT — VERSION 8: CHECKPOINTING & PERSISTENCE
=============================================================================

Evolution from Version 7 (Human-in-the-Loop):
- State persists to SQLite — survives server restarts
- Multiple conversation threads (multi-user support)
- Full state history with time travel capability
- Resume interrupted sessions seamlessly
- Conversation replay for debugging

Architecture stays the same as V7, but now with DURABLE state:

  ┌─────────────────────────────────────────────────────────────┐
  │                    SQLite Database                            │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
  │  │ Thread: usr1 │  │ Thread: usr2 │  │ Thread: usr3 │      │
  │  │  cp1 → cp2   │  │  cp1 → cp2   │  │  cp1         │      │
  │  │  → cp3 → cp4 │  │  → cp3       │  │              │      │
  │  └──────────────┘  └──────────────┘  └──────────────┘      │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     Agent Graph                              │
  │  START → planner → coder ⟷ tools → reviewer → END         │
  │                                                              │
  │  Every node writes a checkpoint. State accumulates.          │
  │  On resume: load latest checkpoint, continue execution.      │
  └─────────────────────────────────────────────────────────────┘

=============================================================================
"""

import os
import sys
import math
import subprocess
import tempfile
import sqlite3
from typing import TypedDict, Annotated, Literal
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from model_config import get_chat_model

load_dotenv()


# =============================================================================
# HELPER
# =============================================================================

def _extract_text(content) -> str:
    """Extract plain text from LLM response content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content else ""


# =============================================================================
# SECTION 1: TOOLS (same as V7, classified by danger)
# =============================================================================

SAFE_TOOL_NAMES = set()
DANGEROUS_TOOL_NAMES = set()


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Safe operation."""
    safe_ns = {
        "__builtins__": {},
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "pi": math.pi, "e": math.e,
        "abs": abs, "round": round, "pow": pow,
    }
    try:
        return str(eval(expression, safe_ns))
    except Exception as e:
        return f"ERROR: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read file contents. Safe (read-only)."""
    try:
        p = Path(file_path)
        if not p.exists():
            return f"ERROR: Not found: {file_path}"
        if p.stat().st_size > 100_000:
            return f"ERROR: Too large"
        return p.read_text(encoding='utf-8')
    except Exception as e:
        return f"ERROR: {e}"


@tool
def list_directory(directory_path: str) -> str:
    """List directory contents. Safe (read-only)."""
    try:
        p = Path(directory_path)
        if not p.exists():
            return f"ERROR: Not found: {directory_path}"
        items = [f"  {'[DIR]' if i.is_dir() else '[FILE]'} {i.name}" for i in sorted(p.iterdir())]
        return f"Contents of {directory_path}:\n" + "\n".join(items) if items else "Empty"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def python_exec(code: str) -> str:
    """Execute Python code. DANGEROUS — requires approval."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_path)
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output = f"EXIT CODE: {result.returncode}\n{output}"
        return output if output.strip() else "(No output)"
    except subprocess.TimeoutExpired:
        os.unlink(temp_path)
        return "ERROR: Timed out (30s)"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write to a file. DANGEROUS — requires approval."""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return f"Wrote {len(content)} chars to {file_path}"
    except Exception as e:
        return f"ERROR: {e}"


# Tool classification
SAFE_TOOLS = [calculator, read_file, list_directory]
DANGEROUS_TOOLS = [python_exec, write_file]
ALL_TOOLS = SAFE_TOOLS + DANGEROUS_TOOLS
DANGEROUS_TOOL_NAMES = {t.name for t in DANGEROUS_TOOLS}


# =============================================================================
# SECTION 2: STATE
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_phase: str
    plan: str
    code_output: str
    review_feedback: str
    iteration: int


# =============================================================================
# SECTION 3: LLM CONFIG
# =============================================================================

llm = get_chat_model()
planner_llm = llm
coder_llm = llm.bind_tools(ALL_TOOLS)
reviewer_llm = llm.bind_tools([python_exec, read_file, list_directory])


# =============================================================================
# SECTION 4: NODES
# =============================================================================

def planner_node(state: AgentState) -> dict:
    system_prompt = SystemMessage(content="""You are a Senior Software Architect.
Create a concise plan. Do NOT write code.
Output: numbered steps, files needed, edge cases.""")
    response = planner_llm.invoke([system_prompt] + state["messages"])
    return {
        "messages": [response],
        "plan": _extract_text(response.content),
        "current_phase": "coding",
        "iteration": state.get("iteration", 0) + 1
    }


def coder_node(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=f"""You are a Senior Python Developer.
Plan: {state.get('plan', 'None')}
Feedback: {state.get('review_feedback', 'None')}
Use tools to implement and test. Say "IMPLEMENTATION COMPLETE" when done.""")
    response = coder_llm.invoke([system_prompt] + state["messages"])
    return {
        "messages": [response],
        "code_output": _extract_text(response.content),
        "current_phase": "coding"
    }


def reviewer_node(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=f"""You are a Code Reviewer.
Plan: {state.get('plan', 'None')}
If good, say "APPROVED". If not, explain fixes needed.""")
    response = reviewer_llm.invoke([system_prompt] + state["messages"])
    return {
        "messages": [response],
        "review_feedback": _extract_text(response.content),
        "current_phase": "reviewing"
    }


# Tool nodes
safe_tool_node = ToolNode(SAFE_TOOLS)
dangerous_tool_node = ToolNode(DANGEROUS_TOOLS)
reviewer_tool_node = ToolNode([python_exec, read_file, list_directory])


# =============================================================================
# SECTION 5: ROUTING
# =============================================================================

def route_coder(state: AgentState) -> Literal["route_tools", "reviewer"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "route_tools"
    return "reviewer"


def route_tools_by_danger(state: AgentState) -> Literal["safe_tools", "dangerous_tools"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            if tc["name"] in DANGEROUS_TOOL_NAMES:
                return "dangerous_tools"
    return "safe_tools"


def route_reviewer(state: AgentState) -> Literal["reviewer_tools", "end_or_revise"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "reviewer_tools"
    return "end_or_revise"


def should_revise(state: AgentState) -> Literal["coder", "end"]:
    review = _extract_text(state.get("review_feedback", ""))
    if state.get("iteration", 0) >= 3:
        return "end"
    if "APPROVED" in review.upper():
        return "end"
    return "coder"


# =============================================================================
# SECTION 6: GRAPH CONSTRUCTION
# =============================================================================
# NEW IN V8: The graph accepts a checkpointer parameter.
# - "memory" → MemorySaver (testing)
# - "sqlite" → SqliteSaver (persistent)
# - path string → SqliteSaver at that path
# =============================================================================

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "agent_state.db"
)


def build_graph():
    """Build the raw graph (without checkpointer — compile separately)."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("route_tools", lambda state: state)
    graph.add_node("safe_tools", safe_tool_node)
    graph.add_node("dangerous_tools", dangerous_tool_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reviewer_tools", reviewer_tool_node)
    graph.add_node("end_or_revise", lambda state: {})

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "coder")
    graph.add_conditional_edges("coder", route_coder)
    graph.add_conditional_edges("route_tools", route_tools_by_danger)
    graph.add_edge("safe_tools", "coder")
    graph.add_edge("dangerous_tools", "coder")
    graph.add_conditional_edges("reviewer", route_reviewer)
    graph.add_edge("reviewer_tools", "reviewer")
    graph.add_conditional_edges("end_or_revise", should_revise, {"coder": "coder", "end": END})

    return graph


def get_agent(storage: str = "sqlite", db_path: str = DEFAULT_DB_PATH):
    """
    Build and compile the agent with the specified storage backend.

    Args:
        storage: "memory" for MemorySaver, "sqlite" for SqliteSaver
        db_path: Path to SQLite database (only used if storage="sqlite")

    Returns:
        Compiled graph (and keeps checkpointer reference alive)
    """
    graph = build_graph()

    if storage == "memory":
        checkpointer = MemorySaver()
    elif storage == "sqlite":
        # Create a persistent SQLite connection (not a context manager)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()  # Create tables if they don't exist
    else:
        raise ValueError(f"Unknown storage: {storage}. Use 'memory' or 'sqlite'.")

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["dangerous_tools"]
    )
    # Keep reference so checkpointer isn't garbage collected
    compiled._checkpointer_ref = checkpointer
    return compiled


# =============================================================================
# SECTION 7: SESSION MANAGEMENT
# =============================================================================
# Production agents need to manage multiple sessions (threads).
# This section provides the interface for:
#   - Starting new sessions
#   - Resuming existing sessions
#   - Listing all sessions
#   - Inspecting session history
# =============================================================================

class AgentSession:
    """
    Manages a persistent coding agent session.
    
    Usage:
        session = AgentSession(db_path="my_agent.db")
        session.start("thread-1", "Build a calculator")
        # ... later, even after restart ...
        session.resume("thread-1")  # picks up where it left off
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, storage: str = "sqlite"):
        self.db_path = db_path
        self.storage = storage
        self.agent = get_agent(storage=storage, db_path=db_path)

    def start(self, thread_id: str, task: str) -> dict:
        """Start a new coding session."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_phase": "planning",
            "plan": "",
            "code_output": "",
            "review_feedback": "",
            "iteration": 0,
        }
        result = self.agent.invoke(initial_state, config)
        return self._process_result(result, config)

    def resume(self, thread_id: str) -> dict:
        """Resume an interrupted session (e.g., after approval or restart)."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        result = self.agent.invoke(None, config)
        return self._process_result(result, config)

    def approve(self, thread_id: str) -> dict:
        """Approve the pending dangerous operation and continue."""
        return self.resume(thread_id)

    def reject(self, thread_id: str, reason: str = "Rejected by human") -> dict:
        """Reject the pending operation with feedback."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        snapshot = self.agent.get_state(config)
        last_msg = snapshot.values["messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            rejections = [
                ToolMessage(content=f"REJECTED: {reason}", tool_call_id=tc["id"])
                for tc in last_msg.tool_calls
            ]
            self.agent.update_state(config, {"messages": rejections}, as_node="dangerous_tools")

        return self.resume(thread_id)

    def get_status(self, thread_id: str) -> dict:
        """Get the current status of a session."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        snapshot = self.agent.get_state(config)

        if not snapshot.values:
            return {"status": "not_found", "thread_id": thread_id}

        pending = snapshot.next
        status = "completed" if not pending else "waiting_approval" if "dangerous_tools" in pending else "in_progress"

        pending_tools = []
        if status == "waiting_approval":
            last = snapshot.values["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                pending_tools = [{"name": tc["name"], "args": tc["args"]} for tc in last.tool_calls]

        return {
            "status": status,
            "thread_id": thread_id,
            "messages": len(snapshot.values.get("messages", [])),
            "phase": snapshot.values.get("current_phase", "unknown"),
            "iteration": snapshot.values.get("iteration", 0),
            "pending_tools": pending_tools,
        }

    def get_history(self, thread_id: str, limit: int = 10) -> list:
        """Get checkpoint history for debugging/audit."""
        config = {"configurable": {"thread_id": thread_id}}
        history = []
        for i, state in enumerate(self.agent.get_state_history(config)):
            if i >= limit:
                break
            history.append({
                "checkpoint_id": state.config["configurable"]["checkpoint_id"],
                "messages": len(state.values.get("messages", [])),
                "next": list(state.next),
                "phase": state.values.get("current_phase", ""),
            })
        return history

    def _process_result(self, result: dict, config: dict) -> dict:
        """Process invoke result and return status."""
        snapshot = self.agent.get_state(config)
        pending = snapshot.next

        if not pending:
            return {"status": "completed", "result": result}
        elif "dangerous_tools" in pending:
            last = snapshot.values["messages"][-1]
            pending_tools = []
            if hasattr(last, "tool_calls") and last.tool_calls:
                pending_tools = [{"name": tc["name"], "args": tc["args"]} for tc in last.tool_calls]
            return {"status": "waiting_approval", "pending_tools": pending_tools}
        else:
            return {"status": "in_progress"}


# =============================================================================
# SECTION 8: INTERACTIVE CLI
# =============================================================================

def run_interactive(db_path: str = DEFAULT_DB_PATH):
    """Interactive CLI for the persistent coding agent."""
    session = AgentSession(db_path=db_path)

    print("=" * 70)
    print("PRODUCTION CODING AGENT v8 — PERSISTENT SESSIONS")
    print("=" * 70)
    print(f"Database: {db_path}")
    print("\nCommands:")
    print("  new <thread> <task>  — Start a new session")
    print("  resume <thread>      — Resume a session")
    print("  status <thread>      — Check session status")
    print("  approve <thread>     — Approve pending action")
    print("  reject <thread>      — Reject pending action")
    print("  history <thread>     — View checkpoint history")
    print("  quit                 — Exit")
    print("-" * 70)

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        action = parts[0].lower()

        if action == "quit":
            break
        elif action == "new" and len(parts) >= 3:
            thread_id, task = parts[1], parts[2]
            print(f"  Starting session '{thread_id}'...")
            result = session.start(thread_id, task)
            _print_status(result)
        elif action == "resume" and len(parts) >= 2:
            result = session.resume(parts[1])
            _print_status(result)
        elif action == "status" and len(parts) >= 2:
            status = session.get_status(parts[1])
            _print_status(status)
        elif action == "approve" and len(parts) >= 2:
            result = session.approve(parts[1])
            _print_status(result)
        elif action == "reject" and len(parts) >= 2:
            reason = parts[2] if len(parts) > 2 else "Rejected by user"
            result = session.reject(parts[1], reason)
            _print_status(result)
        elif action == "history" and len(parts) >= 2:
            history = session.get_history(parts[1])
            for entry in history:
                print(f"  {entry['checkpoint_id'][:12]}... | msgs={entry['messages']} | next={entry['next']} | phase={entry['phase']}")
        else:
            print("  Unknown command. Type 'quit' to exit.")


def _print_status(status: dict):
    """Pretty-print a status dict."""
    if "status" in status:
        print(f"  Status: {status['status']}")
    if "pending_tools" in status and status["pending_tools"]:
        print(f"  Waiting for approval:")
        for tool_info in status["pending_tools"]:
            args_str = str(tool_info['args'])
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            print(f"    - {tool_info['name']}: {args_str}")
    if "messages" in status and isinstance(status["messages"], int):
        print(f"  Messages: {status['messages']}")
    if "phase" in status:
        print(f"  Phase: {status['phase']}")


# =============================================================================
# SECTION 9: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--cli":
        run_interactive()
    else:
        # Quick demo: auto-approve mode
        print("=" * 60)
        print("V8 DEMO: Persistent Agent (auto-approve)")
        print("=" * 60)

        session = AgentSession(storage="memory")  # Use memory for quick demo
        thread = "demo-session"
        task = "Use python_exec to calculate the factorial of 10 and print the result."

        print(f"\nTask: {task}")
        print("-" * 60)

        result = session.start(thread, task)
        max_loops = 15
        loops = 0
        while result.get("status") == "waiting_approval" and loops < max_loops:
            pending = result.get("pending_tools", [])
            if pending:
                print(f"  [Auto-approve] {pending[0]['name']}")
            result = session.approve(thread)
            loops += 1

        print(f"\nFinal status: {result.get('status', 'unknown')}")
        status = session.get_status(thread)
        print(f"Messages: {status.get('messages', '?')}")
        print(f"History checkpoints: {len(session.get_history(thread, limit=50))}")
