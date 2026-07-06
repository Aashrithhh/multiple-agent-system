"""
=============================================================================
PRODUCTION CODING AGENT — VERSION 7: HUMAN-IN-THE-LOOP
=============================================================================

Evolution from Version 6 (Tool Calling):
- Added approval gates before dangerous tool execution
- Tools are classified as SAFE or DANGEROUS
- Safe tools (calculator, read) execute automatically
- Dangerous tools (write_file, python_exec) require human approval
- Human can: approve, reject, edit tool args, or provide feedback
- Requires checkpointing (MemorySaver) for state persistence across pauses

Architecture:
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Planner   │ (no tools, no approval needed)
                    └──────┬──────┘
                           │
                    ┌──────┴──────────────────────────────────────┐
                    │                                              │
                    ▼                                              │
             ┌─────────────┐                                      │
             │   Coder     │                                      │
             └──────┬──────┘                                      │
                    │                                              │
            ┌───────┴───────┐                                     │
            │ has tool call?│                                      │
            └───┬───────┬───┘                                     │
           yes  │       │  no                                     │
                ▼       ▼                                         │
        ┌────────────┐  ┌────────────┐                           │
        │ Route by   │  │  Reviewer  │                           │
        │ danger lvl │  └─────┬──────┘                           │
        └──┬─────┬───┘        │                                  │
           │     │        ┌───┴───┐                              │
     safe  │     │ danger │approve?│                              │
           ▼     ▼        └─┬───┬─┘                              │
    ┌──────────┐ ┌────────┐ │   │                                │
    │safe_tools│ │APPROVAL│ yes  no ──► coder (revise)           │
    │(auto-run)│ │ GATE   │ │                                    │
    └────┬─────┘ │ PAUSE  │ ▼                                    │
         │       └────┬───┘ END                                  │
         │            │                                           │
         │    [Human approves]                                    │
         │            │                                           │
         │            ▼                                           │
         │    ┌──────────────┐                                   │
         │    │dangerous_tools│                                   │
         │    │  (execute)   │                                    │
         │    └──────┬───────┘                                   │
         │           │                                            │
         └─────┬─────┘                                            │
               │                                                  │
               ▼                                                  │
            coder (loop back)                                     │
                                                                  │
=============================================================================
"""

import os
import sys
import math
import subprocess
import tempfile
from typing import TypedDict, Annotated, Literal
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from model_config import get_chat_model

load_dotenv()


# =============================================================================
# HELPER
# =============================================================================

def _extract_text(content) -> str:
    """Extract plain text from LLM response content (handles Gemini list format)."""
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
# SECTION 1: TOOL DEFINITIONS WITH DANGER CLASSIFICATION
# =============================================================================
# NEW IN V7: We classify tools as SAFE or DANGEROUS.
# - SAFE tools run automatically (no human approval needed)
# - DANGEROUS tools pause execution and wait for human approval
#
# This is a common production pattern:
#   - Read operations → safe
#   - Write/execute operations → dangerous
# =============================================================================

# --- SAFE TOOLS (auto-execute, no approval needed) ---

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e
    """
    safe_namespace = {
        "__builtins__": {},
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log2": math.log2,
        "log10": math.log10, "pi": math.pi, "e": math.e,
        "abs": abs, "round": round, "pow": pow,
    }
    try:
        result = eval(expression, safe_namespace)
        return str(result)
    except Exception as e:
        return f"ERROR: Could not evaluate '{expression}': {e}"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file. This is a READ-ONLY operation (safe)."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: File not found: {file_path}"
        if not path.is_file():
            return f"ERROR: Not a file: {file_path}"
        if path.stat().st_size > 100_000:
            return f"ERROR: File too large ({path.stat().st_size} bytes)"
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f"ERROR: {e}"


@tool
def list_directory(directory_path: str) -> str:
    """List files and directories. This is a READ-ONLY operation (safe)."""
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"ERROR: Not found: {directory_path}"
        if not path.is_dir():
            return f"ERROR: Not a directory: {directory_path}"
        items = []
        for item in sorted(path.iterdir()):
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            items.append(f"  {prefix} {item.name}")
        return f"Contents of {directory_path}:\n" + "\n".join(items) if items else "Empty"
    except Exception as e:
        return f"ERROR: {e}"


# --- DANGEROUS TOOLS (require human approval) ---

@tool
def python_exec(code: str) -> str:
    """Execute Python code in a subprocess.
    
    WARNING: This executes arbitrary code. Requires human approval.
    """
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
        return "ERROR: Timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates or OVERWRITES the file.
    
    WARNING: This modifies the filesystem. Requires human approval.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"ERROR: {e}"


# --- TOOL CLASSIFICATION ---
# This is the KEY new concept in V7: separating tools by risk level

SAFE_TOOLS = [calculator, read_file, list_directory]
DANGEROUS_TOOLS = [python_exec, write_file]
ALL_TOOLS = SAFE_TOOLS + DANGEROUS_TOOLS

# Names for quick lookup
DANGEROUS_TOOL_NAMES = {t.name for t in DANGEROUS_TOOLS}


# =============================================================================
# SECTION 2: STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_phase: str
    plan: str
    code_output: str
    review_feedback: str
    iteration: int


# =============================================================================
# SECTION 3: LLM CONFIGURATION
# =============================================================================

llm = get_chat_model()
planner_llm = llm
coder_llm = llm.bind_tools(ALL_TOOLS)
reviewer_llm = llm.bind_tools([python_exec, read_file, list_directory])


# =============================================================================
# SECTION 4: NODE DEFINITIONS
# =============================================================================

def planner_node(state: AgentState) -> dict:
    """Plan the approach (no tools, no approval needed)."""
    system_prompt = SystemMessage(content="""You are a Senior Software Architect.
Analyze the task and create a concise plan. Do NOT write code — only plan.

Output format:
## Plan
1. [Step]
2. [Step]

## Files Needed
- filename.py: [purpose]

## Edge Cases
- [case]
""")
    messages = [system_prompt] + state["messages"]
    response = planner_llm.invoke(messages)
    return {
        "messages": [response],
        "plan": _extract_text(response.content),
        "current_phase": "coding",
        "iteration": state.get("iteration", 0) + 1
    }


def coder_node(state: AgentState) -> dict:
    """Implement code using tools. May trigger approval gates."""
    system_prompt = SystemMessage(content=f"""You are a Senior Python Developer.

## Current Plan
{state.get('plan', 'No plan')}

## Previous Review Feedback
{state.get('review_feedback', 'No previous review')}

## Instructions
1. Follow the plan step by step
2. Use python_exec to test code
3. Use write_file to save completed code
4. When done, say "IMPLEMENTATION COMPLETE"
""")
    messages = [system_prompt] + state["messages"]
    response = coder_llm.invoke(messages)
    return {
        "messages": [response],
        "code_output": _extract_text(response.content),
        "current_phase": "coding"
    }


def reviewer_node(state: AgentState) -> dict:
    """Review the code, can run tests to verify."""
    system_prompt = SystemMessage(content=f"""You are a Senior Code Reviewer.

## Checklist
1. Does the code match the plan?
2. Are there bugs?
3. Are edge cases handled?
4. Does it run without errors?

If code is good, say "APPROVED". If not, explain what to fix.

## Plan Was
{state.get('plan', 'No plan')}
""")
    messages = [system_prompt] + state["messages"]
    response = reviewer_llm.invoke(messages)
    return {
        "messages": [response],
        "review_feedback": _extract_text(response.content),
        "current_phase": "reviewing"
    }


# =============================================================================
# SECTION 5: TOOL NODES
# =============================================================================
# NEW IN V7: Two separate ToolNodes
#   - safe_tool_node: executes safe tools immediately
#   - dangerous_tool_node: only runs AFTER human approval (interrupt_before)
# =============================================================================

safe_tool_node = ToolNode(SAFE_TOOLS)
dangerous_tool_node = ToolNode(DANGEROUS_TOOLS)

# Reviewer's tools (read + exec for verification)
reviewer_tool_node = ToolNode([python_exec, read_file, list_directory])


# =============================================================================
# SECTION 6: ROUTING LOGIC
# =============================================================================
# NEW IN V7: route_coder_tools splits based on danger level
# =============================================================================

def route_coder(state: AgentState) -> Literal["route_tools", "reviewer"]:
    """After coder responds, check if it has tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "route_tools"
    return "reviewer"


def route_tools_by_danger(state: AgentState) -> Literal["safe_tools", "dangerous_tools"]:
    """
    KEY V7 LOGIC: Route tool calls by danger level.
    
    If ANY tool call is dangerous → route to dangerous_tools (human approval gate).
    If ALL tool calls are safe → route to safe_tools (auto-execute).
    
    Why route ALL to dangerous if even one is dangerous?
    Because ToolNode executes ALL pending tool_calls. We can't partially execute.
    In a production system, you'd split the calls — but for clarity, we route conservatively.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            if tc["name"] in DANGEROUS_TOOL_NAMES:
                return "dangerous_tools"
    return "safe_tools"


def route_reviewer(state: AgentState) -> Literal["reviewer_tools", "end_or_revise"]:
    """After reviewer responds, check for tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "reviewer_tools"
    return "end_or_revise"


def should_revise(state: AgentState) -> Literal["coder", "end"]:
    """Decide if another iteration is needed."""
    review = _extract_text(state.get("review_feedback", ""))
    iteration = state.get("iteration", 0)
    if iteration >= 3:
        return "end"
    if "APPROVED" in review.upper():
        return "end"
    return "coder"


# =============================================================================
# SECTION 7: GRAPH CONSTRUCTION
# =============================================================================
# The graph now has TWO paths for tool execution:
#   - safe_tools: no interrupt, auto-execute
#   - dangerous_tools: interrupt_before, human must approve
# =============================================================================

def build_coding_agent_graph():
    """
    Build the V7 agent with human-in-the-loop approval gates.
    
    Returns (compiled_graph, checkpointer) so caller can manage state.
    """
    graph = StateGraph(AgentState)
    
    # --- Nodes ---
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("route_tools", lambda state: state)  # passthrough for routing
    graph.add_node("safe_tools", safe_tool_node)
    graph.add_node("dangerous_tools", dangerous_tool_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reviewer_tools", reviewer_tool_node)
    graph.add_node("end_or_revise", lambda state: {})
    
    # --- Edges ---
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "coder")
    
    # Coder → check for tool calls
    graph.add_conditional_edges("coder", route_coder)
    
    # Route tools by danger level
    graph.add_conditional_edges("route_tools", route_tools_by_danger)
    
    # Safe tools → back to coder (no pause)
    graph.add_edge("safe_tools", "coder")
    
    # Dangerous tools → back to coder (but will be interrupted BEFORE execution)
    graph.add_edge("dangerous_tools", "coder")
    
    # Reviewer flow
    graph.add_conditional_edges("reviewer", route_reviewer)
    graph.add_edge("reviewer_tools", "reviewer")
    graph.add_conditional_edges("end_or_revise", should_revise, {
        "coder": "coder",
        "end": END
    })
    
    # --- Compile with checkpointer and interrupt ---
    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["dangerous_tools"]  # PAUSE before dangerous tool execution
    )
    
    return compiled


# =============================================================================
# SECTION 8: INTERACTIVE EXECUTION WITH APPROVAL FLOW
# =============================================================================
# This is the main execution loop. When the graph pauses at an approval gate,
# we display what the agent wants to do and ask the human to approve/reject.
# =============================================================================

def run_agent_interactive(task: str):
    """
    Run the agent with interactive human approval.
    
    When the agent wants to execute a dangerous tool:
    1. Execution pauses
    2. We show the human what's about to happen
    3. Human types: 'y' (approve), 'n' (reject), or 'e' (edit)
    4. Execution resumes based on human's decision
    """
    agent = build_coding_agent_graph()
    config = {"configurable": {"thread_id": "coding-session-1"}}
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "current_phase": "planning",
        "plan": "",
        "code_output": "",
        "review_feedback": "",
        "iteration": 0
    }
    
    print("=" * 70)
    print("PRODUCTION CODING AGENT v7 — HUMAN-IN-THE-LOOP")
    print("=" * 70)
    print(f"\nTask: {task}\n")
    print("-" * 70)
    
    # First invocation
    config["recursion_limit"] = 50
    result = agent.invoke(initial_state, config)
    
    # Loop: keep resuming until the graph reaches END
    while True:
        # Check the current graph state
        snapshot = agent.get_state(config)
        
        # If there are no next steps, we're done
        if not snapshot.next:
            print("\n" + "=" * 70)
            print("AGENT COMPLETED!")
            print("=" * 70)
            break
        
        # We're paused at an approval gate
        if "dangerous_tools" in snapshot.next:
            # Display what the agent wants to do
            last_msg = snapshot.values["messages"][-1]
            
            print("\n" + "!" * 70)
            print("APPROVAL REQUIRED — DANGEROUS OPERATION")
            print("!" * 70)
            
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for i, tc in enumerate(last_msg.tool_calls):
                    print(f"\n  Tool #{i+1}: {tc['name']}")
                    print(f"  Arguments:")
                    for key, value in tc['args'].items():
                        # Truncate long values (like code)
                        val_str = str(value)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        print(f"    {key}: {val_str}")
            
            print("\n" + "-" * 70)
            choice = input("  [y] Approve  [n] Reject  [e] Edit args  > ").strip().lower()
            
            if choice == 'y' or choice == '':
                # APPROVE: resume execution
                print("  -> Approved. Executing...")
                result = agent.invoke(None, config)
                
            elif choice == 'n':
                # REJECT: send rejection feedback and resume
                print("  -> Rejected.")
                reason = input("  Reason (optional): ").strip() or "Action rejected by human operator."
                
                rejection_messages = []
                for tc in last_msg.tool_calls:
                    rejection_messages.append(
                        ToolMessage(
                            content=f"REJECTED BY HUMAN: {reason}",
                            tool_call_id=tc["id"]
                        )
                    )
                agent.update_state(config, {"messages": rejection_messages})
                result = agent.invoke(None, config)
                
            elif choice == 'e':
                # EDIT: let human modify arguments
                print("  -> Edit mode.")
                edited_tool_calls = []
                for tc in last_msg.tool_calls:
                    print(f"\n  Editing tool: {tc['name']}")
                    edited_args = dict(tc['args'])
                    for key, value in tc['args'].items():
                        val_str = str(value)
                        if len(val_str) > 100:
                            val_str = val_str[:100] + "..."
                        new_val = input(f"    {key} [{val_str}]: ").strip()
                        if new_val:
                            edited_args[key] = new_val
                    edited_tool_calls.append({
                        "name": tc["name"],
                        "args": edited_args,
                        "id": tc["id"],
                        "type": "tool_call"
                    })
                
                edited_message = AIMessage(content="", tool_calls=edited_tool_calls)
                agent.update_state(config, {"messages": [edited_message]}, as_node="coder")
                result = agent.invoke(None, config)
            else:
                # Default to approve
                print("  -> Unrecognized input. Approving...")
                result = agent.invoke(None, config)
    
    # Print final summary
    print(f"\nPhase: {result.get('current_phase', 'done')}")
    print(f"Iterations: {result.get('iteration', 0)}")
    review = _extract_text(result.get("review_feedback", ""))
    if review:
        print(f"Review: {review[:200]}")


# =============================================================================
# SECTION 9: NON-INTERACTIVE MODE (for testing / CI)
# =============================================================================

def run_agent_auto_approve(task: str):
    """
    Run the agent with automatic approval of all dangerous actions.
    Useful for testing when you trust the task.
    """
    agent = build_coding_agent_graph()
    config = {"configurable": {"thread_id": "auto-approve-session"}}
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "current_phase": "planning",
        "plan": "",
        "code_output": "",
        "review_feedback": "",
        "iteration": 0
    }
    
    config["recursion_limit"] = 50
    result = agent.invoke(initial_state, config)
    
    # Auto-approve loop
    max_approvals = 20  # Safety limit
    approvals = 0
    
    while True:
        snapshot = agent.get_state(config)
        if not snapshot.next:
            break
        if approvals >= max_approvals:
            print("WARNING: Max auto-approvals reached. Stopping.")
            break
        approvals += 1
        result = agent.invoke(None, config)
    
    return result


# =============================================================================
# SECTION 10: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    task = """
    Create a Python function called 'word_frequency' that:
    1. Takes a string of text
    2. Returns a dict mapping each word (lowercased) to its count
    3. Ignores punctuation
    4. Handle empty string edge case
    Test it with python_exec, then save to 'word_frequency.py'
    """
    
    # Interactive mode: human approves each dangerous action
    run_agent_interactive(task)
