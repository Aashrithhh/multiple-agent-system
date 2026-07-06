"""
=============================================================================
PRODUCTION CODING AGENT — VERSION 6: TOOL CALLING
=============================================================================

Evolution from Version 5 (Multi-Agent System):
- Added real executable tools (Python REPL, file operations, web search)
- Agents can now ACT on the world, not just think
- Tool execution is sandboxed and results flow back into the graph

Architecture:
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Planner   │ (decides approach, uses no tools)
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐     ┌──────────────┐
                    │   Coder     │────▶│  Tool Node   │
                    │  (Agent)    │◀────│ (Executes)   │
                    └──────┬──────┘     └──────────────┘
                           │                    │
                           │  Tools available:  │
                           │  - python_exec     │
                           │  - read_file       │
                           │  - write_file      │
                           │  - list_directory  │
                           │  - calculator      │
                           ▼
                    ┌─────────────┐     ┌──────────────┐
                    │  Reviewer   │────▶│  Tool Node   │
                    │  (Agent)    │◀────│ (Executes)   │
                    └──────┬──────┘     └──────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │     END     │
                    └─────────────┘

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

from model_config import get_chat_model

# Load environment variables
load_dotenv()

# =============================================================================
# SECTION 1: TOOL DEFINITIONS
# =============================================================================
# Tools are regular Python functions decorated with @tool.
# The decorator does three things:
#   1. Registers the function's name, args, and docstring as metadata
#   2. Makes it callable by LangGraph's ToolNode
#   3. The docstring becomes the tool's description (LLM reads this to decide when to use it)
#
# CRITICAL: The docstring is what the LLM sees. Write it clearly.
# CRITICAL: Type hints define the schema the LLM must follow when calling the tool.
# =============================================================================


@tool
def python_exec(code: str) -> str:
    """Execute Python code and return the output.
    
    Use this tool when you need to:
    - Run Python code to verify it works
    - Test a function implementation
    - Perform calculations
    - Debug code by running it
    
    The code runs in a subprocess for safety.
    Returns stdout if successful, or the error message if it fails.
    """
    # WHY subprocess? Security. We don't want arbitrary code running in our process.
    # A production system would use Docker containers or AWS Lambda for true isolation.
    try:
        # Create a temporary file to hold the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        # Execute with a timeout to prevent infinite loops
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout — prevents runaway code
        )
        
        # Clean up
        os.unlink(temp_path)
        
        # Return stdout + stderr for complete picture
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output = f"EXIT CODE: {result.returncode}\n{output}"
        
        return output if output.strip() else "(No output produced)"
        
    except subprocess.TimeoutExpired:
        os.unlink(temp_path)
        return "ERROR: Code execution timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file and return it as text.
    
    Use this tool when you need to:
    - Examine existing code
    - Read configuration files
    - Check what's already written in a file
    
    Returns the file contents or an error message.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: File not found: {file_path}"
        if not path.is_file():
            return f"ERROR: Path is not a file: {file_path}"
        # Limit file size to prevent memory issues
        if path.stat().st_size > 100_000:  # 100KB limit
            return f"ERROR: File too large ({path.stat().st_size} bytes). Max 100KB."
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist, overwrites if it does.
    
    Use this tool when you need to:
    - Create a new Python file
    - Save generated code
    - Write configuration
    
    Returns a confirmation message or error.
    """
    try:
        path = Path(file_path)
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def list_directory(directory_path: str) -> str:
    """List files and directories at the given path.
    
    Use this tool when you need to:
    - Explore project structure
    - Find files in a directory
    - Check what files exist
    
    Returns a formatted list of contents.
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"ERROR: Directory not found: {directory_path}"
        if not path.is_dir():
            return f"ERROR: Path is not a directory: {directory_path}"
        
        items = []
        for item in sorted(path.iterdir()):
            prefix = "📁" if item.is_dir() else "📄"
            items.append(f"  {prefix} {item.name}")
        
        return f"Contents of {directory_path}:\n" + "\n".join(items) if items else "Directory is empty"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    
    Use this tool for quick math calculations without running full Python code.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e
    
    Examples: "2 + 2", "sqrt(144)", "sin(pi/2)"
    """
    # WHY a separate calculator tool? 
    # 1. Faster than spawning a Python subprocess for simple math
    # 2. Safer — only math operations allowed
    # 3. Teaching point: tools should be focused and single-purpose
    
    # Safe namespace — only math functions available
    safe_namespace = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
    }
    
    try:
        result = eval(expression, safe_namespace)  # Safe because namespace is restricted
        return str(result)
    except Exception as e:
        return f"ERROR: Could not evaluate '{expression}': {str(e)}"


# =============================================================================
# SECTION 2: STATE DEFINITION
# =============================================================================
# State flows through the entire graph. Every node reads from and writes to state.
# add_messages is a REDUCER — it appends messages instead of overwriting.
# =============================================================================


class AgentState(TypedDict):
    """
    The shared state that flows through our graph.
    
    messages: The conversation history (accumulates via add_messages reducer)
    current_phase: Which phase of work we're in (planning, coding, reviewing)
    plan: The plan created by the Planner agent
    code_output: The code produced by the Coder agent
    review_feedback: Feedback from the Reviewer agent
    iteration: How many plan→code→review cycles we've done
    """
    messages: Annotated[list, add_messages]
    current_phase: str
    plan: str
    code_output: str
    review_feedback: str
    iteration: int


# =============================================================================
# SECTION 3: LLM CONFIGURATION WITH TOOLS
# =============================================================================
# bind_tools() is the KEY method. It tells the LLM:
#   "Here are tools you CAN use. Here are their names, descriptions, and schemas."
#
# The LLM then includes tool_calls in its response when it wants to use one.
# It does NOT execute the tool — it just says "I want to call X with args Y"
# =============================================================================

# Collect all our tools into a list
coder_tools = [python_exec, read_file, write_file, list_directory, calculator]
reviewer_tools = [python_exec, read_file, list_directory]  # Reviewer can run code to verify

# Create LLM instances with tools bound
# WHY separate instances? Each agent gets different tools and system prompts.
llm = get_chat_model()

# Planner gets NO tools — it should only think, not act
planner_llm = llm

# Coder gets ALL tools — it needs to write files, run code, etc.
coder_llm = llm.bind_tools(coder_tools)

# Reviewer gets READ + EXECUTE tools — it verifies but doesn't modify
reviewer_llm = llm.bind_tools(reviewer_tools)


# =============================================================================
# SECTION 4: NODE DEFINITIONS
# =============================================================================
# Each node is a function that:
#   1. Receives the current state
#   2. Does work (calls LLM, processes data)
#   3. Returns a partial state update (only the fields that changed)
# =============================================================================


def planner_node(state: AgentState) -> dict:
    """
    The Planner analyzes the task and creates a step-by-step plan.
    
    It does NOT use tools — planning is pure reasoning.
    It receives the user's request and outputs a structured plan.
    """
    system_prompt = SystemMessage(content="""You are a Senior Software Architect.

Your job is to analyze coding tasks and create clear, actionable plans.

Rules:
1. Break the task into numbered steps
2. Specify what files need to be created/modified
3. Identify edge cases and error handling needed
4. Keep plans concise but complete
5. Do NOT write code — only plan

Output format:
## Plan
1. [Step 1]
2. [Step 2]
...

## Files Needed
- filename.py: [purpose]

## Edge Cases
- [case 1]
- [case 2]
""")
    
    messages = [system_prompt] + state["messages"]
    response = planner_llm.invoke(messages)
    
    return {
        "messages": [response],
        "plan": response.content,
        "current_phase": "coding",
        "iteration": state.get("iteration", 0) + 1
    }


def coder_node(state: AgentState) -> dict:
    """
    The Coder implements the plan using tools.
    
    This is where tool calling happens:
    1. LLM sees the plan and available tools
    2. LLM decides which tools to use
    3. LLM outputs tool_calls in its response
    4. We detect tool_calls and route to ToolNode
    
    IMPORTANT: This node may be called MULTIPLE times in a loop:
      coder_node → tool_node → coder_node → tool_node → ... → coder_node (done)
    Each iteration, the LLM sees previous tool results and decides what to do next.
    """
    system_prompt = SystemMessage(content=f"""You are a Senior Python Developer.

You have access to tools to help you write and test code.

## Current Plan
{state.get('plan', 'No plan provided')}

## Previous Review Feedback
{state.get('review_feedback', 'No previous review')}

## Instructions
1. Follow the plan step by step
2. Use python_exec to test your code as you write it
3. Use write_file to save completed code
4. Make sure code is clean, typed, and documented
5. When you're done implementing and testing, say "IMPLEMENTATION COMPLETE"

## Code Quality Standards
- Use type hints
- Add docstrings
- Handle errors gracefully
- Follow PEP 8
""")
    
    messages = [system_prompt] + state["messages"]
    response = coder_llm.invoke(messages)
    
    # The response might contain tool_calls — that's handled by routing logic
    return {
        "messages": [response],
        "code_output": response.content if response.content else "",
        "current_phase": "coding"  # stays in coding until tools are done
    }


def reviewer_node(state: AgentState) -> dict:
    """
    The Reviewer checks the Coder's work using tools.
    
    It can:
    - Read files to check the code
    - Run the code to verify it works
    - Provide feedback for improvements
    """
    system_prompt = SystemMessage(content=f"""You are a Senior Code Reviewer.

You have tools to read and execute code for verification.

## Your Review Checklist
1. ✅ Does the code match the plan?
2. ✅ Are there any bugs?
3. ✅ Are edge cases handled?
4. ✅ Is error handling adequate?
5. ✅ Is the code clean and well-documented?
6. ✅ Does it actually run without errors?

## Instructions
- Use read_file to examine the code
- Use python_exec to run tests if needed
- Be specific about any issues found
- If code is good, say "APPROVED"
- If code needs changes, explain exactly what to fix

## Current Plan Was
{state.get('plan', 'No plan')}
""")
    
    messages = [system_prompt] + state["messages"]
    response = reviewer_llm.invoke(messages)
    
    return {
        "messages": [response],
        "review_feedback": response.content if response.content else "",
        "current_phase": "reviewing"
    }


# =============================================================================
# SECTION 5: TOOL NODES
# =============================================================================
# ToolNode is LangGraph's pre-built node that:
#   1. Reads tool_calls from the last AIMessage
#   2. Executes the corresponding Python function
#   3. Returns ToolMessages with results
#
# WHY pre-built? Because tool execution follows the same pattern every time.
# You COULD write it manually, but ToolNode handles edge cases for you.
# =============================================================================

# Create ToolNodes for each agent that uses tools
coder_tool_node = ToolNode(coder_tools)
reviewer_tool_node = ToolNode(reviewer_tools)


# =============================================================================
# SECTION 6: ROUTING LOGIC
# =============================================================================
# Routing functions determine where to go next based on state.
# The critical routing is: "Did the LLM request a tool call?"
#
# How to detect tool calls:
#   response.tool_calls → list of tool calls (empty if none)
#
# If tool_calls exist → route to ToolNode
# If no tool_calls → agent is done, route to next phase
# =============================================================================


def route_coder(state: AgentState) -> Literal["coder_tools", "reviewer"]:
    """
    After the coder responds, check if it wants to use tools.
    
    This is the CORE of tool calling routing:
    - If the last message has tool_calls → go execute them
    - If no tool_calls → coder is done, move to reviewer
    """
    last_message = state["messages"][-1]
    
    # Check if the LLM wants to call tools
    # tool_calls is a list of {name, args, id} dicts
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "coder_tools"
    
    # No tool calls — coder is finished
    return "reviewer"


def route_reviewer(state: AgentState) -> Literal["reviewer_tools", "end_or_revise"]:
    """
    After the reviewer responds, check if it wants to use tools.
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "reviewer_tools"
    
    return "end_or_revise"


def should_revise(state: AgentState) -> Literal["coder", "end"]:
    """
    After review is complete, decide if we need another iteration.
    
    - If reviewer said "APPROVED" → END
    - If iteration limit reached → END (prevent infinite loops)
    - Otherwise → back to coder for revisions
    """
    review = state.get("review_feedback", "")
    iteration = state.get("iteration", 0)
    
    # Safety: max 3 iterations to prevent infinite loops
    if iteration >= 3:
        return "end"
    
    # Check if reviewer approved
    if "APPROVED" in review.upper():
        return "end"
    
    # Needs revision
    return "coder"


# =============================================================================
# SECTION 7: GRAPH CONSTRUCTION
# =============================================================================
# This is where we wire everything together.
# The graph defines the FLOW of execution.
# =============================================================================


def build_coding_agent_graph():
    """
    Construct the full Production Coding Agent graph with tool calling.
    
    Graph structure:
    
    START → planner → coder ⟷ coder_tools (loop until done)
                        ↓
                    reviewer ⟷ reviewer_tools (loop until done)
                        ↓
                  end_or_revise → END (if approved)
                        ↓
                      coder (if needs revision)
    """
    # Initialize the graph with our state schema
    graph = StateGraph(AgentState)
    
    # --- Add Nodes ---
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("coder_tools", coder_tool_node)      # Executes coder's tool calls
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reviewer_tools", reviewer_tool_node)  # Executes reviewer's tool calls
    graph.add_node("end_or_revise", lambda state: {})
    
    # --- Add Edges ---
    
    # START → planner (always starts with planning)
    graph.add_edge(START, "planner")
    
    # planner → coder (after planning, always code)
    graph.add_edge("planner", "coder")
    
    # coder → conditional: tools or reviewer
    graph.add_conditional_edges("coder", route_coder)
    
    # coder_tools → coder (after tool execution, go back to coder to continue)
    # This creates the tool-calling LOOP
    graph.add_edge("coder_tools", "coder")
    
    # reviewer → conditional: tools or end_or_revise
    graph.add_conditional_edges("reviewer", route_reviewer)
    
    # reviewer_tools → reviewer (after tool execution, go back to reviewer)
    graph.add_edge("reviewer_tools", "reviewer")
    
    # end_or_revise → conditional: end or back to coder
    graph.add_conditional_edges("end_or_revise", should_revise, {
        "coder": "coder",
        "end": END
    })
    
    # Compile the graph
    return graph.compile()


# =============================================================================
# SECTION 8: MAIN EXECUTION
# =============================================================================


def run_agent(task: str):
    """
    Run the Production Coding Agent on a task.
    
    Args:
        task: Natural language description of what to build
    """
    # Build the graph
    agent = build_coding_agent_graph()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "current_phase": "planning",
        "plan": "",
        "code_output": "",
        "review_feedback": "",
        "iteration": 0
    }
    
    print("=" * 70)
    print("🤖 PRODUCTION CODING AGENT v6 — TOOL CALLING")
    print("=" * 70)
    print(f"\n📋 Task: {task}\n")
    print("-" * 70)
    
    # Stream execution to see each step
    for event in agent.stream(initial_state, {"recursion_limit": 50}):
        for node_name, output in event.items():
            print(f"\n{'='*50}")
            print(f"📍 Node: {node_name}")
            print(f"{'='*50}")
            
            if "messages" in output:
                last_msg = output["messages"][-1]
                
                # Print content if it exists
                if hasattr(last_msg, "content") and last_msg.content:
                    # Truncate long outputs for readability
                    content = last_msg.content
                    if len(content) > 500:
                        content = content[:500] + "\n... (truncated)"
                    print(f"\n{content}")
                
                # Print tool calls if any
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"\n🔧 Tool Call: {tc['name']}")
                        args_str = str(tc['args'])
                        if len(args_str) > 200:
                            args_str = args_str[:200] + "..."
                        print(f"   Args: {args_str}")
                
                # Print tool results
                if isinstance(last_msg, ToolMessage):
                    result = last_msg.content
                    if len(result) > 300:
                        result = result[:300] + "\n... (truncated)"
                    print(f"\n📤 Tool Result:\n{result}")
    
    print("\n" + "=" * 70)
    print("✅ Agent execution complete!")
    print("=" * 70)


# =============================================================================
# SECTION 9: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example task — this is what the agent will build
    task = """
    Create a Python function called 'analyze_text' that:
    1. Takes a string input
    2. Returns a dictionary with:
       - word_count: number of words
       - char_count: number of characters (excluding spaces)
       - sentence_count: number of sentences
       - most_common_word: the most frequently used word
       - average_word_length: average length of words
    3. Handle edge cases (empty string, None input)
    4. Include type hints and docstring
    5. Write tests that verify the function works
    
    Save the implementation to 'text_analyzer.py' and tests to 'test_text_analyzer.py'
    """
    
    run_agent(task)
