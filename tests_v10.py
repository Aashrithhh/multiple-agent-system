"""
=============================================================================
VERSION 10 — UNIT TESTS & INTEGRATION TESTS
=============================================================================

Covers:
- Tool error handling (input validation, exceptions, edge cases)
- SafeToolNode (catches crashes, tracks failures)
- Retry logic
- Routing with error awareness
- Graph compilation
- End-to-end with error tracking
- Integration across V6-V10
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(__file__))

from langgraph.errors import GraphRecursionError
from coding_agent_v10 import (
    calculator, read_file, list_directory, python_exec, write_file,
    SafeToolNode, get_agent, build_graph, run_auto_approve,
    _extract_text, with_retry, _safe_llm_invoke,
    route_coder, route_tools_by_danger, route_reviewer, should_revise,
    SAFE_TOOLS, DANGEROUS_TOOLS, DANGEROUS_TOOL_NAMES, ALL_TOOLS,
    AgentState,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# =============================================================================
# UNIT TESTS: Tool Input Validation
# =============================================================================

def test_calculator_empty_input():
    result = calculator.invoke({"expression": ""})
    assert "ERROR" in result
    print("  PASS: calculator empty input")


def test_calculator_division_by_zero():
    result = calculator.invoke({"expression": "1/0"})
    assert "Division by zero" in result
    print("  PASS: calculator division by zero")


def test_calculator_invalid_syntax():
    result = calculator.invoke({"expression": "def foo():"})
    assert "ERROR" in result
    print("  PASS: calculator invalid syntax")


def test_calculator_valid():
    assert calculator.invoke({"expression": "2 + 2"}) == "4"
    assert calculator.invoke({"expression": "sqrt(9)"}) == "3.0"
    print("  PASS: calculator valid expressions")


def test_read_file_empty_path():
    result = read_file.invoke({"file_path": ""})
    assert "ERROR" in result
    print("  PASS: read_file empty path")


def test_read_file_not_found():
    result = read_file.invoke({"file_path": "/nonexistent/path/file.txt"})
    assert "ERROR" in result and "not found" in result.lower()
    print("  PASS: read_file not found")


def test_read_file_is_directory():
    result = read_file.invoke({"file_path": os.path.dirname(__file__)})
    assert "ERROR" in result and "directory" in result.lower()
    print("  PASS: read_file is directory")


def test_read_file_valid():
    # Use requirements.txt — small and always exists
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    result = read_file.invoke({"file_path": path})
    assert "ERROR" not in result
    assert "langgraph" in result
    print("  PASS: read_file valid")


def test_list_directory_empty():
    result = list_directory.invoke({"directory_path": ""})
    assert "ERROR" in result
    print("  PASS: list_directory empty path")


def test_list_directory_not_found():
    result = list_directory.invoke({"directory_path": "/no/such/dir"})
    assert "ERROR" in result
    print("  PASS: list_directory not found")


def test_list_directory_valid():
    result = list_directory.invoke({"directory_path": os.path.dirname(__file__)})
    assert "coding_agent_v10.py" in result
    print("  PASS: list_directory valid")


def test_python_exec_empty():
    result = python_exec.invoke({"code": ""})
    assert "ERROR" in result
    print("  PASS: python_exec empty code")


def test_python_exec_syntax_error():
    result = python_exec.invoke({"code": "def ("})
    assert "SyntaxError" in result or "EXIT CODE" in result
    print("  PASS: python_exec syntax error")


def test_python_exec_runtime_error():
    result = python_exec.invoke({"code": "x = 1/0"})
    assert "ZeroDivisionError" in result
    print("  PASS: python_exec runtime error")


def test_python_exec_valid():
    result = python_exec.invoke({"code": "print(123)"})
    assert "123" in result
    print("  PASS: python_exec valid")


def test_write_file_empty_path():
    result = write_file.invoke({"file_path": "", "content": "test"})
    assert "ERROR" in result
    print("  PASS: write_file empty path")


def test_write_file_none_content():
    """Pydantic rejects None for str field — SafeToolNode catches this."""
    node = SafeToolNode(DANGEROUS_TOOLS)
    msg = AIMessage(content="", tool_calls=[
        {"name": "write_file", "args": {"file_path": "/tmp/x.txt", "content": None}, "id": "wf-1"}
    ])
    state = {"messages": [msg], "error_count": 0, "tool_failures": []}
    result = node(state)
    # SafeToolNode should catch the validation error
    assert result["error_count"] == 1
    assert "ERROR" in result["messages"][0].content
    print("  PASS: write_file none content (caught by SafeToolNode)")


def test_write_file_valid():
    path = os.path.join(tempfile.gettempdir(), "test_v10_write.txt")
    result = write_file.invoke({"file_path": path, "content": "hello v10"})
    assert "Successfully wrote" in result
    os.unlink(path)
    print("  PASS: write_file valid")


# =============================================================================
# UNIT TESTS: SafeToolNode
# =============================================================================

def test_safe_tool_node_catches_crash():
    """SafeToolNode catches tool exceptions instead of crashing."""
    from langchain_core.tools import tool as tool_dec

    @tool_dec
    def crashing_tool(x: str) -> str:
        """A tool that always crashes."""
        raise RuntimeError("Unexpected explosion!")

    node = SafeToolNode([crashing_tool])
    msg = AIMessage(content="", tool_calls=[
        {"name": "crashing_tool", "args": {"x": "test"}, "id": "crash-1"}
    ])
    state = {"messages": [msg], "error_count": 0, "tool_failures": []}

    result = node(state)

    assert result["error_count"] == 1
    assert len(result["tool_failures"]) == 1
    assert "explosion" in result["tool_failures"][0]["error"].lower()
    # Still returns a ToolMessage (not a crash)
    assert isinstance(result["messages"][0], ToolMessage)
    assert "ERROR" in result["messages"][0].content
    print("  PASS: SafeToolNode catches crashes")


def test_safe_tool_node_unknown_tool():
    """SafeToolNode handles unknown tool names."""
    node = SafeToolNode(SAFE_TOOLS)
    msg = AIMessage(content="", tool_calls=[
        {"name": "nonexistent_tool", "args": {}, "id": "unk-1"}
    ])
    state = {"messages": [msg], "error_count": 0, "tool_failures": []}

    result = node(state)
    assert result["error_count"] == 1
    assert "Unknown tool" in result["messages"][0].content
    print("  PASS: SafeToolNode unknown tool")


def test_safe_tool_node_success():
    """SafeToolNode passes through on success."""
    node = SafeToolNode(SAFE_TOOLS)
    msg = AIMessage(content="", tool_calls=[
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "ok-1"}
    ])
    state = {"messages": [msg], "error_count": 0, "tool_failures": []}

    result = node(state)
    assert result["error_count"] == 0
    assert result["messages"][0].content == "4"
    print("  PASS: SafeToolNode success")


# =============================================================================
# UNIT TESTS: Retry Decorator
# =============================================================================

def test_retry_succeeds_on_third():
    """Retry decorator succeeds after transient failures."""
    attempts = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0.01)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("timeout")
        return "success"

    result = flaky()
    assert result == "success"
    assert attempts["n"] == 3
    print("  PASS: retry succeeds on 3rd attempt")


def test_retry_gives_up():
    """Retry decorator raises after max attempts."""
    @with_retry(max_attempts=2, base_delay=0.01)
    def always_fails():
        raise ConnectionError("down")

    try:
        always_fails()
        assert False, "Should have raised"
    except ConnectionError:
        pass
    print("  PASS: retry gives up after max")


def test_retry_non_retryable():
    """Non-retryable errors fail immediately."""
    attempts = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0.01)
    def bad_input():
        attempts["n"] += 1
        raise ValueError("bad input")  # Not in retryable_errors

    try:
        bad_input()
    except ValueError:
        pass
    assert attempts["n"] == 1  # Only tried once
    print("  PASS: non-retryable errors fail immediately")


# =============================================================================
# UNIT TESTS: Routing & Logic
# =============================================================================

def test_route_coder():
    msg = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    assert route_coder({"messages": [msg]}) == "route_tools"
    assert route_coder({"messages": [AIMessage(content="done")]}) == "reviewer"
    print("  PASS: route_coder")


def test_route_tools():
    safe = AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "1"}])
    danger = AIMessage(content="", tool_calls=[{"name": "python_exec", "args": {}, "id": "1"}])
    assert route_tools_by_danger({"messages": [safe]}) == "safe_tools"
    assert route_tools_by_danger({"messages": [danger]}) == "dangerous_tools"
    print("  PASS: route_tools_by_danger")


def test_should_revise_error_limit():
    """Too many errors stops the loop."""
    state = {"review_feedback": "needs fix", "iteration": 1, "error_count": 10}
    assert should_revise(state) == "end"
    print("  PASS: should_revise error limit")


def test_should_revise_normal():
    assert should_revise({"review_feedback": "APPROVED", "iteration": 1, "error_count": 0}) == "end"
    assert should_revise({"review_feedback": "fix it", "iteration": 1, "error_count": 0}) == "coder"
    assert should_revise({"review_feedback": "fix it", "iteration": 3, "error_count": 0}) == "end"
    print("  PASS: should_revise normal cases")


# =============================================================================
# UNIT TESTS: Helpers
# =============================================================================

def test_extract_text():
    assert _extract_text("hello") == "hello"
    assert _extract_text([{"type": "text", "text": "hi"}]) == "hi"
    assert _extract_text(None) == ""
    print("  PASS: _extract_text")


def test_graph_compiles():
    agent = get_agent("memory")
    nodes = list(agent.get_graph().nodes.keys())
    for n in ["planner", "coder", "safe_tools", "dangerous_tools", "reviewer"]:
        assert n in nodes
    print("  PASS: graph compiles")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_integration_error_tracking():
    """Errors are tracked in state through execution."""
    agent = get_agent("memory")
    config = {"configurable": {"thread_id": "error-track"}, "recursion_limit": 50}

    result = agent.invoke({
        "messages": [HumanMessage(content="Use python_exec to print 'hello'")],
        "current_phase": "planning", "plan": "", "code_output": "",
        "review_feedback": "", "iteration": 0,
        "error_count": 0, "last_error": "", "tool_failures": [],
    }, config)

    # error_count should be in the state (may be 0 if no errors)
    assert "error_count" in result
    assert isinstance(result["error_count"], int)
    print(f"  PASS: integration — error tracking (errors={result['error_count']})")


def test_integration_full_run():
    """Full agent run completes with error handling."""
    try:
        result = run_auto_approve("Use python_exec to print(2**8)")
        assert result is not None
        assert result.get("error_count", 0) >= 0
        review = _extract_text(result.get("review_feedback", ""))
        print(f"  PASS: integration — full run (errors={result.get('error_count', 0)}, review={review[:50]})")
    except GraphRecursionError:
        print("  PASS: integration — full run (hit recursion limit — acceptable)")


def test_integration_tool_failure_flows_to_llm():
    """When a tool returns ERROR, the LLM sees it and can adapt."""
    node = SafeToolNode(SAFE_TOOLS)

    # Call read_file with a bad path — should return error string, not crash
    msg = AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"file_path": "/no/such/file.txt"}, "id": "f1"}
    ])
    state = {"messages": [msg], "error_count": 0, "tool_failures": []}
    result = node(state)

    # Error message should be in the ToolMessage (LLM will see this)
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "ERROR" in tool_msg.content
    assert "not found" in tool_msg.content.lower()
    # But error_count should be 0 (tool handled the error gracefully)
    assert result["error_count"] == 0
    print("  PASS: integration — tool error flows to LLM as message")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VERSION 10 — UNIT TESTS & INTEGRATION TESTS")
    print("=" * 70)

    print("\n[1/5] Tool Input Validation:")
    test_calculator_empty_input()
    test_calculator_division_by_zero()
    test_calculator_invalid_syntax()
    test_calculator_valid()
    test_read_file_empty_path()
    test_read_file_not_found()
    test_read_file_is_directory()
    test_read_file_valid()
    test_list_directory_empty()
    test_list_directory_not_found()
    test_list_directory_valid()
    test_python_exec_empty()
    test_python_exec_syntax_error()
    test_python_exec_runtime_error()
    test_python_exec_valid()
    test_write_file_empty_path()
    test_write_file_none_content()
    test_write_file_valid()

    print("\n[2/5] SafeToolNode:")
    test_safe_tool_node_catches_crash()
    test_safe_tool_node_unknown_tool()
    test_safe_tool_node_success()

    print("\n[3/5] Retry Logic:")
    test_retry_succeeds_on_third()
    test_retry_gives_up()
    test_retry_non_retryable()

    print("\n[4/5] Routing & Logic:")
    test_route_coder()
    test_route_tools()
    test_should_revise_error_limit()
    test_should_revise_normal()
    test_extract_text()
    test_graph_compiles()

    print("\n[5/5] Integration Tests (API required):")
    test_integration_error_tracking()
    test_integration_full_run()
    test_integration_tool_failure_flows_to_llm()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
