"""
=============================================================================
ERROR HANDLING — FUNDAMENTALS
=============================================================================

Production agents WILL fail. Tools timeout, APIs return 500s, LLMs hallucinate
invalid tool calls, networks drop. This file teaches:

  1. Tool-level error handling (individual tool failures)
  2. Retry with backoff (transient failures)
  3. Fallback tools (if tool A fails, try tool B)
  4. Graph-level error recovery (node failures)
  5. Graceful degradation (continue despite partial failures)

=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import time
import random
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from model_config import get_chat_model


# =============================================================================
# LESSON 1: Tool-Level Error Handling
# =============================================================================
# Tools MUST handle their own exceptions and return error messages as strings.
# If a tool raises an unhandled exception, the entire graph crashes.
#
# Rule: Tools should NEVER raise. They should ALWAYS return a string
# (either the result or a descriptive error message).
#
# Why? The LLM can READ error messages and adjust its approach.
# If the tool crashes, the LLM gets nothing and the graph dies.
# =============================================================================

# BAD: This tool will crash the graph on invalid input
@tool
def bad_divide(a: float, b: float) -> str:
    """Divide a by b. (BAD - no error handling)"""
    return str(a / b)  # ZeroDivisionError if b=0!


# GOOD: This tool handles all errors gracefully
@tool
def safe_divide(a: float, b: float) -> str:
    """Divide a by b safely.

    Returns the result or a clear error message.
    """
    if b == 0:
        return "ERROR: Cannot divide by zero. Please use a non-zero divisor."
    try:
        result = a / b
        return str(result)
    except Exception as e:
        return f"ERROR: Division failed: {e}"


def lesson_1_tool_error_handling():
    """Demonstrate tool-level error handling."""
    print("=" * 60)
    print("LESSON 1: Tool-Level Error Handling")
    print("=" * 60)

    # BAD tool crashes
    print("\n[BAD tool] Dividing 10 / 0:")
    try:
        result = bad_divide.invoke({"a": 10, "b": 0})
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  CRASHED: {type(e).__name__}: {e}")
        print("  (This would kill the entire agent graph!)")

    # GOOD tool returns error message
    print("\n[GOOD tool] Dividing 10 / 0:")
    result = safe_divide.invoke({"a": 10, "b": 0})
    print(f"  Result: {result}")
    print("  (LLM can read this and try a different approach)")

    # GOOD tool works normally
    print("\n[GOOD tool] Dividing 10 / 3:")
    result = safe_divide.invoke({"a": 10, "b": 3})
    print(f"  Result: {result}")


# =============================================================================
# LESSON 2: Retry with Exponential Backoff
# =============================================================================
# Transient failures (network timeouts, rate limits, 503 errors) often
# succeed on retry. But you must:
#   1. Limit retries (don't retry forever)
#   2. Use exponential backoff (wait longer between each retry)
#   3. Only retry on transient errors (don't retry on 400 Bad Request)
# =============================================================================

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry a function with exponential backoff.

    Args:
        func: Callable to retry
        max_retries: Maximum number of attempts
        base_delay: Initial delay in seconds (doubles each retry)

    Returns:
        Function result or raises the last exception
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s, 8s...
                time.sleep(delay)
    raise last_error


# A tool that simulates flaky behavior (fails sometimes)
_flaky_call_count = {"n": 0}


@tool
def flaky_api(query: str) -> str:
    """Call an external API that sometimes fails."""
    _flaky_call_count["n"] += 1

    # Simulate: fails first 2 times, succeeds on 3rd
    if _flaky_call_count["n"] <= 2:
        raise ConnectionError(f"API timeout on attempt {_flaky_call_count['n']}")

    return f"API result for '{query}': success (attempt {_flaky_call_count['n']})"


@tool
def reliable_api(query: str) -> str:
    """Call an API with built-in retry logic."""
    attempts = 0
    max_attempts = 3
    base_delay = 0.1  # Short delay for demo

    while attempts < max_attempts:
        attempts += 1
        try:
            # Simulate random failure (50% chance)
            if random.random() < 0.4 and attempts < max_attempts:
                raise ConnectionError(f"Timeout on attempt {attempts}")
            return f"Result for '{query}': data retrieved successfully"
        except ConnectionError as e:
            if attempts >= max_attempts:
                return f"ERROR: API failed after {max_attempts} attempts. Last error: {e}"
            time.sleep(base_delay * (2 ** (attempts - 1)))

    return "ERROR: Unexpected state"


def lesson_2_retry_logic():
    """Demonstrate retry with backoff."""
    print("\n" + "=" * 60)
    print("LESSON 2: Retry with Exponential Backoff")
    print("=" * 60)

    # Without retry: first call fails
    _flaky_call_count["n"] = 0
    print("\n[Without retry] Calling flaky API:")
    try:
        result = flaky_api.invoke({"query": "test"})
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # With retry wrapper: succeeds on 3rd attempt
    _flaky_call_count["n"] = 0
    print("\n[With retry] Calling flaky API (max 3 attempts):")
    try:
        result = retry_with_backoff(
            lambda: flaky_api.invoke({"query": "test"}),
            max_retries=3,
            base_delay=0.1
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Still failed after retries: {e}")

    # Built-in retry in tool
    print("\n[Built-in retry] Calling reliable_api:")
    result = reliable_api.invoke({"query": "weather data"})
    print(f"  Result: {result}")


# =============================================================================
# LESSON 3: Fallback Tools
# =============================================================================
# If the primary tool fails, use a fallback.
# Example: If web search fails, use cached results.
# =============================================================================

@tool
def primary_search(query: str) -> str:
    """Search the web for information (primary source)."""
    # Simulate failure
    raise ConnectionError("Web search API is down")


@tool
def cached_search(query: str) -> str:
    """Search local cache for information (fallback)."""
    # Always works — uses local data
    cache = {
        "python": "Python is a programming language created by Guido van Rossum.",
        "langgraph": "LangGraph is a framework for building stateful AI agents.",
    }
    for key, value in cache.items():
        if key in query.lower():
            return f"[FROM CACHE] {value}"
    return f"[FROM CACHE] No cached data for '{query}'"


@tool
def search_with_fallback(query: str) -> str:
    """Search for information. Uses web search first, falls back to cache."""
    # Try primary
    try:
        # Simulate web search
        raise ConnectionError("Service unavailable")
    except (ConnectionError, TimeoutError):
        pass  # Fall through to fallback

    # Fallback to cache
    cache = {
        "python": "Python is a high-level programming language.",
        "langgraph": "LangGraph builds stateful AI agent applications.",
    }
    for key, value in cache.items():
        if key in query.lower():
            return f"[FALLBACK] {value}"
    return f"[FALLBACK] No information available for '{query}'"


def lesson_3_fallback_tools():
    """Demonstrate fallback tool pattern."""
    print("\n" + "=" * 60)
    print("LESSON 3: Fallback Tools")
    print("=" * 60)

    # Primary fails
    print("\n[Primary tool] Searching 'python':")
    try:
        result = primary_search.invoke({"query": "python"})
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Fallback works
    print("\n[Fallback tool] Searching 'python':")
    result = cached_search.invoke({"query": "python"})
    print(f"  Result: {result}")

    # Combined tool with built-in fallback
    print("\n[Combined tool] Searching 'langgraph':")
    result = search_with_fallback.invoke({"query": "langgraph"})
    print(f"  Result: {result}")


# =============================================================================
# LESSON 4: Graph-Level Error Recovery
# =============================================================================
# When a node fails, the graph can:
#   1. Catch the error in the node itself (preferred)
#   2. Use LangGraph's built-in error handling
#   3. Route to a fallback node
#
# Pattern: "error boundary" node that catches failures and decides what to do.
# =============================================================================

class ErrorState(TypedDict):
    messages: Annotated[list, add_messages]
    error_count: int
    last_error: str


def lesson_4_graph_error_recovery():
    """Demonstrate error recovery at graph level."""
    print("\n" + "=" * 60)
    print("LESSON 4: Graph-Level Error Recovery")
    print("=" * 60)

    call_count = {"n": 0}

    def unreliable_node(state: ErrorState) -> dict:
        """A node that fails the first time but succeeds on retry."""
        call_count["n"] += 1
        if call_count["n"] <= 1:
            # Simulate failure — but DON'T crash. Return error in state.
            return {
                "messages": [AIMessage(content="I encountered an error. Retrying...")],
                "last_error": "API returned 503 Service Unavailable",
                "error_count": state.get("error_count", 0) + 1
            }
        # Success on retry
        return {
            "messages": [AIMessage(content="Operation completed successfully!")],
            "last_error": "",
            "error_count": state.get("error_count", 0)
        }

    def error_router(state: ErrorState) -> Literal["retry", "give_up", "done"]:
        """Decide what to do after a potential error."""
        if state.get("last_error"):
            if state.get("error_count", 0) >= 3:
                return "give_up"
            return "retry"
        return "done"

    def give_up_node(state: ErrorState) -> dict:
        return {
            "messages": [AIMessage(content=f"Failed after {state['error_count']} attempts. Error: {state['last_error']}")],
        }

    # Build graph with error recovery
    graph = StateGraph(ErrorState)
    graph.add_node("worker", unreliable_node)
    graph.add_node("give_up", give_up_node)

    graph.add_edge(START, "worker")
    graph.add_conditional_edges("worker", error_router, {
        "retry": "worker",  # Loop back to retry
        "give_up": "give_up",
        "done": END
    })
    graph.add_edge("give_up", END)

    app = graph.compile()

    # Run it — should fail once, then succeed on retry
    call_count["n"] = 0
    print("\n[Running graph with unreliable node]")
    result = app.invoke({
        "messages": [HumanMessage(content="Do the thing")],
        "error_count": 0,
        "last_error": ""
    })

    print(f"  Attempts: {call_count['n']}")
    print(f"  Error count in state: {result['error_count']}")
    print(f"  Final message: {result['messages'][-1].content}")


# =============================================================================
# LESSON 5: Graceful Degradation
# =============================================================================
# Sometimes you can't fully complete a task, but you can give a PARTIAL result.
# This is better than crashing with nothing.
#
# Pattern: Track which steps succeeded/failed, return what you have.
# =============================================================================

def lesson_5_graceful_degradation():
    """Demonstrate graceful degradation pattern."""
    print("\n" + "=" * 60)
    print("LESSON 5: Graceful Degradation")
    print("=" * 60)

    class TaskState(TypedDict):
        messages: Annotated[list, add_messages]
        results: dict
        errors: list

    def step_1(state: TaskState) -> dict:
        """First step — always succeeds."""
        results = dict(state.get("results", {}))
        results["step_1"] = "Fetched user data successfully"
        return {"results": results}

    def step_2(state: TaskState) -> dict:
        """Second step — fails (simulated)."""
        errors = list(state.get("errors", []))
        errors.append("step_2: Database connection timeout")
        return {"errors": errors}

    def step_3(state: TaskState) -> dict:
        """Third step — succeeds despite step 2 failure."""
        results = dict(state.get("results", {}))
        results["step_3"] = "Generated report from available data"
        return {"results": results}

    def summarize(state: TaskState) -> dict:
        """Final step — summarize what worked and what didn't."""
        results = state.get("results", {})
        errors = state.get("errors", [])

        if errors:
            summary = f"Completed with partial results. Succeeded: {list(results.keys())}. Failed: {errors}"
        else:
            summary = f"All steps completed successfully: {list(results.keys())}"

        return {"messages": [AIMessage(content=summary)]}

    graph = StateGraph(TaskState)
    graph.add_node("step_1", step_1)
    graph.add_node("step_2", step_2)
    graph.add_node("step_3", step_3)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "step_3")  # Continue despite failure
    graph.add_edge("step_3", "summarize")
    graph.add_edge("summarize", END)

    app = graph.compile()

    result = app.invoke({
        "messages": [HumanMessage(content="Process the task")],
        "results": {},
        "errors": []
    })

    print(f"\n  Results collected: {result['results']}")
    print(f"  Errors collected: {result['errors']}")
    print(f"  Summary: {result['messages'][-1].content}")
    print("\n  (Agent continued and returned partial results instead of crashing)")


# =============================================================================
# RUN ALL LESSONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ERROR HANDLING FUNDAMENTALS")
    print("=" * 60)

    lesson_1_tool_error_handling()
    lesson_2_retry_logic()
    lesson_3_fallback_tools()
    lesson_4_graph_error_recovery()
    lesson_5_graceful_degradation()

    print("\n\n" + "=" * 60)
    print("ALL ERROR HANDLING LESSONS COMPLETED")
    print("=" * 60)
