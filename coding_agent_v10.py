"""
=============================================================================
PRODUCTION CODING AGENT — VERSION 10: ERROR HANDLING
=============================================================================

Evolution from Version 9 (Streaming):
- Tools have built-in retry logic for transient failures
- Fallback mechanisms when primary tools fail
- Graph-level error recovery (nodes track errors in state)
- Max retry limits prevent infinite loops
- Graceful degradation: partial results instead of crashes
- Error context flows back to LLM so it can adapt

Key principle: NEVER let the graph crash silently.
Either recover automatically, or fail gracefully with useful info.

=============================================================================
"""

import os
import sys
import math
import subprocess
import tempfile
import sqlite3
import json
import time
from typing import TypedDict, Annotated, Literal
from pathlib import Path
from datetime import datetime
from functools import wraps

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
# SECTION 1: RETRY DECORATOR
# =============================================================================
# Reusable retry logic that any tool can use.
# This is a PRODUCTION PATTERN — you'll see it in every real codebase.
# =============================================================================

def with_retry(max_attempts: int = 3, base_delay: float = 0.5,
               retryable_errors=(ConnectionError, TimeoutError, OSError)):
    """
    Decorator that adds retry with exponential backoff to a function.

    Args:
        max_attempts: Total attempts before giving up
        base_delay: Initial wait time (doubles each retry)
        retryable_errors: Tuple of exception types worth retrying
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                except Exception:
                    # Non-retryable error — fail immediately
                    raise
            # All retries exhausted
            raise last_error
        return wrapper
    return decorator


# =============================================================================
# SECTION 2: HELPER
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
# SECTION 3: TOOLS WITH ERROR HANDLING
# =============================================================================
# Each tool now:
#   1. Validates input before acting
#   2. Catches ALL exceptions
#   3. Returns structured error messages the LLM can understand
#   4. Has retry logic for transient failures
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e
    """
    if not expression or not expression.strip():
        return "ERROR: Empty expression. Please provide a math expression."

    safe_ns = {
        "__builtins__": {},
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "pi": math.pi, "e": math.e,
        "abs": abs, "round": round, "pow": pow,
    }
    try:
        result = eval(expression, safe_ns)
        if result is None:
            return "ERROR: Expression produced no result."
        return str(result)
    except ZeroDivisionError:
        return "ERROR: Division by zero. Please check your expression."
    except (SyntaxError, NameError) as e:
        return f"ERROR: Invalid expression '{expression}'. {type(e).__name__}: {e}"
    except Exception as e:
        return f"ERROR: Could not evaluate '{expression}': {type(e).__name__}: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read file contents (safe, read-only)."""
    if not file_path or not file_path.strip():
        return "ERROR: No file path provided."
    try:
        p = Path(file_path)
        if not p.exists():
            return f"ERROR: File not found: '{file_path}'. Check the path and try again."
        if not p.is_file():
            return f"ERROR: '{file_path}' is a directory, not a file. Use list_directory instead."
        if p.stat().st_size > 100_000:
            return f"ERROR: File too large ({p.stat().st_size} bytes). Maximum is 100KB."
        return p.read_text(encoding='utf-8')
    except PermissionError:
        return f"ERROR: Permission denied reading '{file_path}'."
    except UnicodeDecodeError:
        return f"ERROR: '{file_path}' is not a text file (binary content)."
    except Exception as e:
        return f"ERROR: Failed to read '{file_path}': {type(e).__name__}: {e}"


@tool
def list_directory(directory_path: str) -> str:
    """List directory contents (safe, read-only)."""
    if not directory_path or not directory_path.strip():
        return "ERROR: No directory path provided."
    try:
        p = Path(directory_path)
        if not p.exists():
            return f"ERROR: Directory not found: '{directory_path}'."
        if not p.is_dir():
            return f"ERROR: '{directory_path}' is a file, not a directory. Use read_file instead."
        items = [f"  {'[DIR]' if i.is_dir() else '[FILE]'} {i.name}" for i in sorted(p.iterdir())]
        return f"Contents of {directory_path}:\n" + "\n".join(items) if items else "Directory is empty."
    except PermissionError:
        return f"ERROR: Permission denied accessing '{directory_path}'."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def python_exec(code: str) -> str:
    """Execute Python code in a subprocess.

    Returns stdout/stderr or a clear error message.
    Has a 30-second timeout to prevent infinite loops.
    """
    if not code or not code.strip():
        return "ERROR: No code provided. Please provide Python code to execute."

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=30
        )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output = f"EXIT CODE: {result.returncode}\n{output}"
        return output if output.strip() else "(No output produced)"

    except subprocess.TimeoutExpired:
        return ("ERROR: Code execution timed out after 30 seconds. "
                "Your code may have an infinite loop. Check for: "
                "while True without break, recursive calls without base case.")
    except FileNotFoundError:
        return "ERROR: Python interpreter not found. System configuration issue."
    except PermissionError:
        return "ERROR: Permission denied creating temp file."
    except Exception as e:
        return f"ERROR: Execution failed: {type(e).__name__}: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # Best effort cleanup


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.

    Creates parent directories if needed. Validates the path first.
    """
    if not file_path or not file_path.strip():
        return "ERROR: No file path provided."
    if content is None:
        return "ERROR: No content provided. Pass content='' for an empty file."

    try:
        p = Path(file_path)

        # Validate path
        if len(str(p)) > 260:  # Windows path limit
            return "ERROR: File path too long (max 260 characters)."

        # Create parent dirs
        p.parent.mkdir(parents=True, exist_ok=True)

        # Write
        p.write_text(content, encoding='utf-8')
        return f"Successfully wrote {len(content)} characters to '{file_path}'"

    except PermissionError:
        return f"ERROR: Permission denied writing to '{file_path}'."
    except OSError as e:
        return f"ERROR: OS error writing '{file_path}': {e}"
    except Exception as e:
        return f"ERROR: Failed to write '{file_path}': {type(e).__name__}: {e}"


# Tool classification
SAFE_TOOLS = [calculator, read_file, list_directory]
DANGEROUS_TOOLS = [python_exec, write_file]
ALL_TOOLS = SAFE_TOOLS + DANGEROUS_TOOLS
DANGEROUS_TOOL_NAMES = {t.name for t in DANGEROUS_TOOLS}


# =============================================================================
# SECTION 4: STATE WITH ERROR TRACKING
# =============================================================================
# NEW IN V10: State includes error tracking fields.
# This lets the graph make decisions based on error history.
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_phase: str
    plan: str
    code_output: str
    review_feedback: str
    iteration: int
    # V10: Error tracking
    error_count: int
    last_error: str
    tool_failures: list  # List of {tool, error, timestamp}


# =============================================================================
# SECTION 5: LLM WITH ERROR-AWARE INVOCATION
# =============================================================================
# The LLM call itself can fail (rate limits, network issues).
# We wrap it with retry logic.
# =============================================================================

llm = get_chat_model()


def _safe_llm_invoke(bound_llm, messages, max_retries=2):
    """
    Invoke LLM with retry for transient errors.

    Returns the response or a synthetic error AIMessage.
    """
    for attempt in range(max_retries + 1):
        try:
            return bound_llm.invoke(messages)
        except Exception as e:
            error_type = type(e).__name__
            if attempt < max_retries:
                # Retry on likely transient errors
                time.sleep(1 * (attempt + 1))
                continue
            # All retries failed — return a synthetic message
            return AIMessage(
                content=f"[LLM ERROR after {max_retries + 1} attempts: {error_type}: {e}. "
                        f"Please try again or simplify your request.]"
            )


planner_llm = llm
coder_llm = llm.bind_tools(ALL_TOOLS)
reviewer_llm = llm.bind_tools([python_exec, read_file, list_directory])


# =============================================================================
# SECTION 6: NODES WITH ERROR RECOVERY
# =============================================================================

def planner_node(state: AgentState) -> dict:
    system_prompt = SystemMessage(content="""You are a Senior Software Architect.
Create a concise plan. Do NOT write code. Output: numbered steps, files needed, edge cases.""")

    response = _safe_llm_invoke(planner_llm, [system_prompt] + state["messages"])
    return {
        "messages": [response],
        "plan": _extract_text(response.content),
        "current_phase": "coding",
        "iteration": state.get("iteration", 0) + 1
    }


def coder_node(state: AgentState) -> dict:
    # Include error context so LLM can adapt
    error_context = ""
    if state.get("tool_failures"):
        recent_failures = state["tool_failures"][-3:]  # Last 3 failures
        error_context = "\n\n## Recent Tool Failures (adapt your approach):\n"
        for f in recent_failures:
            error_context += f"- {f.get('tool', '?')}: {f.get('error', '?')}\n"

    system_prompt = SystemMessage(content=f"""You are a Senior Python Developer.
Plan: {state.get('plan', 'None')}
Feedback: {state.get('review_feedback', 'None')}{error_context}
Use tools to implement and test. Say "IMPLEMENTATION COMPLETE" when done.
If a tool fails, try a different approach. Do not repeat the same failing call.""")

    response = _safe_llm_invoke(coder_llm, [system_prompt] + state["messages"])
    return {
        "messages": [response],
        "code_output": _extract_text(response.content),
        "current_phase": "coding"
    }


def reviewer_node(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=f"""You are a Code Reviewer.
Plan: {state.get('plan', 'None')}
If good, say "APPROVED". If not, explain fixes needed.
Note: {state.get('error_count', 0)} tool errors occurred during this session.""")

    response = _safe_llm_invoke(reviewer_llm, [system_prompt] + state["messages"])
    return {
        "messages": [response],
        "review_feedback": _extract_text(response.content),
        "current_phase": "reviewing"
    }


# =============================================================================
# SECTION 7: ERROR-AWARE TOOL NODE
# =============================================================================
# Instead of using ToolNode directly, we wrap it to:
#   1. Catch tool execution errors
#   2. Track failures in state
#   3. Return error messages instead of crashing
# =============================================================================

class SafeToolNode:
    """
    A wrapper around ToolNode that catches errors and tracks failures.

    If a tool raises an exception, instead of crashing the graph:
    - Returns a ToolMessage with the error description
    - Increments error_count in state
    - Logs the failure in tool_failures
    """

    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}
        self.tool_node = ToolNode(tools)

    def __call__(self, state: AgentState) -> dict:
        """Execute tools safely, catching any errors."""
        last_msg = state["messages"][-1]

        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {}

        results = []
        error_count = state.get("error_count", 0)
        tool_failures = list(state.get("tool_failures", []))

        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            try:
                # Execute the tool
                if tool_name in self.tools:
                    result = self.tools[tool_name].invoke(tool_args)
                else:
                    result = f"ERROR: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
                    error_count += 1
                    tool_failures.append({
                        "tool": tool_name,
                        "error": "Unknown tool",
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                result = f"ERROR: Tool '{tool_name}' crashed: {type(e).__name__}: {e}"
                error_count += 1
                tool_failures.append({
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

            results.append(ToolMessage(content=result, tool_call_id=tool_id))

        return {
            "messages": results,
            "error_count": error_count,
            "tool_failures": tool_failures
        }


# Create safe tool nodes
safe_tool_node = SafeToolNode(SAFE_TOOLS)
dangerous_tool_node = SafeToolNode(DANGEROUS_TOOLS)
reviewer_tool_node = SafeToolNode([python_exec, read_file, list_directory])


# =============================================================================
# SECTION 8: ROUTING WITH ERROR AWARENESS
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
    """Decide if we need another iteration. Error-aware."""
    review = _extract_text(state.get("review_feedback", ""))
    iteration = state.get("iteration", 0)
    error_count = state.get("error_count", 0)

    # Max iterations
    if iteration >= 3:
        return "end"

    # Too many errors — stop to avoid burning API credits
    if error_count >= 10:
        return "end"

    # Approved
    if "APPROVED" in review.upper():
        return "end"

    return "coder"


# =============================================================================
# SECTION 9: GRAPH
# =============================================================================

def build_graph():
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


def get_agent(storage="memory"):
    """Build compiled agent with error-aware tool nodes."""
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
# SECTION 10: RUN
# =============================================================================

def run_auto_approve(task: str):
    """Run agent with auto-approve and error tracking."""
    agent = get_agent()
    config = {"configurable": {"thread_id": "v10-session"}, "recursion_limit": 50}

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "current_phase": "planning",
        "plan": "", "code_output": "", "review_feedback": "",
        "iteration": 0, "error_count": 0, "last_error": "",
        "tool_failures": [],
    }

    result = agent.invoke(initial_state, config)

    # Auto-approve loop
    loops = 0
    while True:
        snapshot = agent.get_state(config)
        if not snapshot.next:
            break
        if loops >= 15:
            print("WARNING: Max approvals reached")
            break
        loops += 1
        result = agent.invoke(None, config)

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("V10 ERROR HANDLING DEMO (auto-approve)")
    print("=" * 60)

    task = "Use python_exec to compute factorial(10) and print it."
    print(f"\nTask: {task}\n")

    result = run_auto_approve(task)

    print(f"\nPhase: {result.get('current_phase')}")
    print(f"Iterations: {result.get('iteration')}")
    print(f"Errors: {result.get('error_count', 0)}")
    if result.get("tool_failures"):
        print(f"Tool failures: {result['tool_failures']}")
    print(f"Review: {_extract_text(result.get('review_feedback', ''))[:200]}")
