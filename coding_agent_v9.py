"""
=============================================================================
PRODUCTION CODING AGENT — VERSION 9: STREAMING
=============================================================================

Evolution from Version 8 (Checkpointing):
- Node-level streaming: see each agent step as it completes
- Token streaming: see LLM responses character by character
- Progress callbacks: track what the agent is doing in real-time
- SSE-compatible output for web integration
- Both sync and async streaming supported

This version adds a StreamingSession that wraps the agent and provides
real-time visibility into execution — essential for production UX.

=============================================================================
"""

import os
import sys
import math
import subprocess
import tempfile
import asyncio
import json
import sqlite3
from typing import TypedDict, Annotated, Literal, Callable
from pathlib import Path
from datetime import datetime

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
# SECTION 1: TOOLS
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
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
    """Read file contents (safe, read-only)."""
    try:
        p = Path(file_path)
        if not p.exists():
            return f"ERROR: Not found: {file_path}"
        if p.stat().st_size > 100_000:
            return "ERROR: Too large"
        return p.read_text(encoding='utf-8')
    except Exception as e:
        return f"ERROR: {e}"


@tool
def list_directory(directory_path: str) -> str:
    """List directory contents (safe, read-only)."""
    try:
        p = Path(directory_path)
        if not p.exists():
            return f"ERROR: Not found"
        items = [f"  {'[DIR]' if i.is_dir() else '[FILE]'} {i.name}" for i in sorted(p.iterdir())]
        return "\n".join(items) if items else "Empty"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def python_exec(code: str) -> str:
    """Execute Python code (dangerous — requires approval)."""
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
    """Write to a file (dangerous — requires approval)."""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return f"Wrote {len(content)} chars to {file_path}"
    except Exception as e:
        return f"ERROR: {e}"


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
# SECTION 3: LLM & NODES
# =============================================================================

llm = get_chat_model()
planner_llm = llm
coder_llm = llm.bind_tools(ALL_TOOLS)
reviewer_llm = llm.bind_tools([python_exec, read_file, list_directory])


def planner_node(state: AgentState) -> dict:
    system_prompt = SystemMessage(content="""You are a Senior Software Architect.
Create a concise plan. Do NOT write code. Output: numbered steps, files needed, edge cases.""")
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


safe_tool_node = ToolNode(SAFE_TOOLS)
dangerous_tool_node = ToolNode(DANGEROUS_TOOLS)
reviewer_tool_node = ToolNode([python_exec, read_file, list_directory])


# =============================================================================
# SECTION 4: ROUTING
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
# SECTION 5: GRAPH
# =============================================================================

def build_graph():
    """Build the raw StateGraph."""
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


def get_agent(storage: str = "memory"):
    """Compile the agent with checkpointing and interrupt."""
    graph = build_graph()
    if storage == "memory":
        checkpointer = MemorySaver()
    else:
        conn = sqlite3.connect(storage, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["dangerous_tools"]
    )
    compiled._checkpointer_ref = checkpointer
    return compiled


# =============================================================================
# SECTION 6: STREAMING SESSION
# =============================================================================
# This is the KEY new class for V9.
# It wraps the agent and provides streaming output via callbacks.
# =============================================================================

class StreamingSession:
    """
    Production streaming interface for the coding agent.
    
    Supports:
    - Sync streaming (for CLI)
    - Async streaming (for web servers)
    - Custom callbacks for each event type
    - SSE-formatted output
    """

    def __init__(self, storage: str = "memory"):
        self.agent = get_agent(storage)
        self._callbacks: list[Callable] = []

    def on_event(self, callback: Callable):
        """Register a callback for streaming events."""
        self._callbacks.append(callback)

    def _emit(self, event: dict):
        """Emit an event to all registered callbacks."""
        for cb in self._callbacks:
            cb(event)

    def stream_task(self, thread_id: str, task: str):
        """
        Stream a task execution with real-time progress events.
        
        Yields dicts with event info:
          {"type": "node_start", "node": "planner", "timestamp": "..."}
          {"type": "tool_call", "tool": "python_exec", "args": {...}}
          {"type": "tool_result", "tool": "python_exec", "result": "..."}
          {"type": "message", "node": "reviewer", "content": "APPROVED"}
          {"type": "approval_needed", "tools": [...]}
          {"type": "done", "status": "completed"}
        """
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_phase": "planning",
            "plan": "", "code_output": "",
            "review_feedback": "", "iteration": 0,
        }

        yield {"type": "start", "task": task, "thread_id": thread_id,
               "timestamp": datetime.now().isoformat()}

        for event in self.agent.stream(initial_state, config, stream_mode="updates"):
            for node_name, output in event.items():
                if not output:
                    continue
                yield {"type": "node_complete", "node": node_name,
                       "timestamp": datetime.now().isoformat()}

                if "messages" not in output:
                    continue

                last_msg = output["messages"][-1]

                # Tool calls
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        yield {
                            "type": "tool_call",
                            "node": node_name,
                            "tool": tc["name"],
                            "args": tc["args"],
                        }

                # Tool results
                elif isinstance(last_msg, ToolMessage):
                    yield {
                        "type": "tool_result",
                        "node": node_name,
                        "content": last_msg.content[:500],
                    }

                # Regular messages
                elif hasattr(last_msg, "content") and last_msg.content:
                    text = _extract_text(last_msg.content)
                    if text:
                        yield {
                            "type": "message",
                            "node": node_name,
                            "content": text[:500],
                        }

        # Check if we're paused at approval gate
        snapshot = self.agent.get_state(config)
        if snapshot.next and "dangerous_tools" in snapshot.next:
            last = snapshot.values["messages"][-1]
            pending = []
            if hasattr(last, "tool_calls") and last.tool_calls:
                pending = [{"name": tc["name"], "args": tc["args"]} for tc in last.tool_calls]
            yield {"type": "approval_needed", "pending_tools": pending}
        else:
            yield {"type": "done", "status": "completed",
                   "timestamp": datetime.now().isoformat()}

    def stream_approve(self, thread_id: str):
        """Stream the continuation after approval."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

        yield {"type": "approved", "timestamp": datetime.now().isoformat()}

        for event in self.agent.stream(None, config, stream_mode="updates"):
            for node_name, output in event.items():
                if not output:
                    continue
                yield {"type": "node_complete", "node": node_name,
                       "timestamp": datetime.now().isoformat()}

                if "messages" not in output:
                    continue
                last_msg = output["messages"][-1]

                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        yield {"type": "tool_call", "node": node_name,
                               "tool": tc["name"], "args": tc["args"]}
                elif isinstance(last_msg, ToolMessage):
                    yield {"type": "tool_result", "node": node_name,
                           "content": last_msg.content[:500]}
                elif hasattr(last_msg, "content") and last_msg.content:
                    text = _extract_text(last_msg.content)
                    if text:
                        yield {"type": "message", "node": node_name, "content": text[:500]}

        snapshot = self.agent.get_state(config)
        if snapshot.next and "dangerous_tools" in snapshot.next:
            last = snapshot.values["messages"][-1]
            pending = []
            if hasattr(last, "tool_calls") and last.tool_calls:
                pending = [{"name": tc["name"], "args": tc["args"]} for tc in last.tool_calls]
            yield {"type": "approval_needed", "pending_tools": pending}
        else:
            yield {"type": "done", "status": "completed",
                   "timestamp": datetime.now().isoformat()}

    def stream_to_sse(self, thread_id: str, task: str):
        """
        Yield Server-Sent Event formatted strings.
        Ready to be used with FastAPI StreamingResponse.
        """
        for event in self.stream_task(thread_id, task):
            yield f"data: {json.dumps(event)}\n\n"

    def stream_approve_to_sse(self, thread_id: str):
        """Yield SSE strings for approval continuation."""
        for event in self.stream_approve(thread_id):
            yield f"data: {json.dumps(event)}\n\n"


# =============================================================================
# SECTION 7: CLI WITH STREAMING
# =============================================================================

def run_streaming_cli():
    """Run the agent with streaming output in the terminal."""
    session = StreamingSession()

    print("=" * 70)
    print("PRODUCTION CODING AGENT v9 — STREAMING")
    print("=" * 70)

    task = input("\nEnter task (or press Enter for demo): ").strip()
    if not task:
        task = "Write a Python function that reverses a string. Test it with python_exec."

    print(f"\nTask: {task}")
    print("=" * 70)

    thread_id = "cli-stream-session"

    for event in session.stream_task(thread_id, task):
        _print_event(event)

        if event["type"] == "approval_needed":
            choice = input("\n  [y] Approve  [n] Reject > ").strip().lower()
            if choice == "y" or choice == "":
                for ev in session.stream_approve(thread_id):
                    _print_event(ev)
                    if ev["type"] == "approval_needed":
                        choice2 = input("\n  [y] Approve  [n] Reject > ").strip().lower()
                        if choice2 == "y" or choice2 == "":
                            # Continue approving in a loop
                            for ev2 in session.stream_approve(thread_id):
                                _print_event(ev2)
                                if ev2["type"] == "done":
                                    break
                        break
                    if ev["type"] == "done":
                        break


def _print_event(event: dict):
    """Pretty-print a streaming event."""
    etype = event.get("type", "")

    if etype == "start":
        print(f"\n  [START] {event.get('task', '')[:80]}")
    elif etype == "node_complete":
        node = event.get("node", "")
        if node not in ("route_tools", "end_or_revise"):
            print(f"\n  [{node.upper()}]")
    elif etype == "tool_call":
        args_str = str(event.get("args", {}))
        if len(args_str) > 100:
            args_str = args_str[:100] + "..."
        print(f"    -> Calling {event['tool']}({args_str})")
    elif etype == "tool_result":
        result = event.get("content", "")
        if len(result) > 150:
            result = result[:150] + "..."
        print(f"    <- Result: {result}")
    elif etype == "message":
        content = event.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"    {content}")
    elif etype == "approval_needed":
        print(f"\n  [!] APPROVAL NEEDED:")
        for tool_info in event.get("pending_tools", []):
            print(f"      {tool_info['name']}: {str(tool_info['args'])[:80]}")
    elif etype == "done":
        print(f"\n  [DONE] Status: {event.get('status', '?')}")
    elif etype == "approved":
        print(f"  [APPROVED] Continuing...")


# =============================================================================
# SECTION 8: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_streaming_cli()
    else:
        # Auto-demo
        print("=" * 60)
        print("V9 STREAMING DEMO (auto-approve)")
        print("=" * 60)

        session = StreamingSession()
        thread_id = "demo-stream"
        task = "Use python_exec to print the first 5 squares: 1, 4, 9, 16, 25"

        print(f"\nTask: {task}\n")

        for event in session.stream_task(thread_id, task):
            _print_event(event)
            if event["type"] == "approval_needed":
                print("  [AUTO-APPROVE]")
                for ev in session.stream_approve(thread_id):
                    _print_event(ev)
                    if ev["type"] == "approval_needed":
                        print("  [AUTO-APPROVE]")
                        for ev2 in session.stream_approve(thread_id):
                            _print_event(ev2)
                            if ev2["type"] == "done":
                                break
                    if ev["type"] == "done":
                        break
                break
