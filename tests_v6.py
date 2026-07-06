"""
Automated tests for Version 6: Tool Calling.

Tests cover:
1. Tool definitions work correctly
2. Model config loads and connects
3. Graph compiles and routes properly
4. End-to-end agent execution
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from coding_agent_v6 import (
    python_exec, read_file, write_file, list_directory, calculator,
    build_coding_agent_graph, route_coder, route_reviewer, should_revise,
    _extract_text, AgentState
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from model_config import get_chat_model


def test_tool_python_exec():
    """Test that python_exec runs code and returns output."""
    result = python_exec.invoke({"code": "print(2 + 2)"})
    assert "4" in result, f"Expected '4' in result, got: {result}"
    print("  PASS: python_exec basic execution")


def test_tool_python_exec_error():
    """Test that python_exec handles errors gracefully."""
    result = python_exec.invoke({"code": "raise ValueError('test error')"})
    assert "ValueError" in result or "EXIT CODE" in result
    print("  PASS: python_exec error handling")


def test_tool_python_exec_timeout():
    """Test that python_exec times out on long-running code."""
    result = python_exec.invoke({"code": "import time; time.sleep(35)"})
    assert "timed out" in result.lower() or "timeout" in result.lower()
    print("  PASS: python_exec timeout")


def test_tool_calculator():
    """Test calculator with various expressions."""
    assert calculator.invoke({"expression": "2 + 2"}) == "4"
    assert calculator.invoke({"expression": "sqrt(144)"}) == "12.0"
    assert calculator.invoke({"expression": "pi"}).startswith("3.14")
    assert "ERROR" in calculator.invoke({"expression": "import os"})
    print("  PASS: calculator")


def test_tool_read_write_file(tmp_path=None):
    """Test file read/write tools."""
    import tempfile
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test.txt")

    # Write
    result = write_file.invoke({"file_path": test_file, "content": "hello world"})
    assert "Successfully wrote" in result

    # Read
    result = read_file.invoke({"file_path": test_file})
    assert result == "hello world"

    # Read nonexistent
    result = read_file.invoke({"file_path": "/nonexistent/path.txt"})
    assert "ERROR" in result

    # Cleanup
    os.unlink(test_file)
    os.rmdir(test_dir)
    print("  PASS: read_file / write_file")


def test_tool_list_directory():
    """Test directory listing."""
    result = list_directory.invoke({"directory_path": os.path.dirname(__file__)})
    assert "coding_agent_v6.py" in result
    assert "ERROR" not in result

    result = list_directory.invoke({"directory_path": "/nonexistent/dir"})
    assert "ERROR" in result
    print("  PASS: list_directory")


def test_extract_text():
    """Test content extraction from various formats."""
    # Plain string
    assert _extract_text("hello") == "hello"

    # Gemini-style list
    assert _extract_text([{"type": "text", "text": "hello"}]) == "hello"

    # Multiple blocks
    assert _extract_text([
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"}
    ]) == "hello\nworld"

    # Empty/None
    assert _extract_text("") == ""
    assert _extract_text(None) == ""
    print("  PASS: _extract_text")


def test_graph_compiles():
    """Test that the agent graph compiles without error."""
    graph = build_coding_agent_graph()
    nodes = list(graph.get_graph().nodes.keys())
    assert "planner" in nodes
    assert "coder" in nodes
    assert "coder_tools" in nodes
    assert "reviewer" in nodes
    assert "reviewer_tools" in nodes
    assert "end_or_revise" in nodes
    print("  PASS: graph compiles")


def test_route_coder_with_tool_calls():
    """Test that routing detects tool calls correctly."""
    # Message WITH tool calls → route to tools
    msg_with_tools = AIMessage(content="", tool_calls=[
        {"name": "python_exec", "args": {"code": "print(1)"}, "id": "123"}
    ])
    state = {"messages": [msg_with_tools]}
    assert route_coder(state) == "coder_tools"

    # Message WITHOUT tool calls → route to reviewer
    msg_without_tools = AIMessage(content="Done!")
    state = {"messages": [msg_without_tools]}
    assert route_coder(state) == "reviewer"
    print("  PASS: route_coder")


def test_route_reviewer_with_tool_calls():
    """Test reviewer routing."""
    msg_with_tools = AIMessage(content="", tool_calls=[
        {"name": "python_exec", "args": {"code": "print(1)"}, "id": "456"}
    ])
    state = {"messages": [msg_with_tools]}
    assert route_reviewer(state) == "reviewer_tools"

    msg_without_tools = AIMessage(content="APPROVED")
    state = {"messages": [msg_without_tools]}
    assert route_reviewer(state) == "end_or_revise"
    print("  PASS: route_reviewer")


def test_should_revise():
    """Test revision logic."""
    # Approved → end
    state = {"review_feedback": "Code looks great. APPROVED.", "iteration": 1}
    assert should_revise(state) == "end"

    # Not approved → revise
    state = {"review_feedback": "Bug found in line 5.", "iteration": 1}
    assert should_revise(state) == "coder"

    # Max iterations → end regardless
    state = {"review_feedback": "Still has bugs.", "iteration": 3}
    assert should_revise(state) == "end"

    # Gemini-style list content with APPROVED
    state = {"review_feedback": [{"type": "text", "text": "APPROVED"}], "iteration": 1}
    assert should_revise(state) == "end"
    print("  PASS: should_revise")


def test_model_connection():
    """Test that the LLM API connection works."""
    llm = get_chat_model()
    response = llm.invoke("Reply with only the word 'OK'")
    text = _extract_text(response.content)
    assert len(text) > 0
    print("  PASS: model connection")


def test_model_tool_calling():
    """Test that the LLM can make tool calls."""
    from langchain_core.tools import tool as tool_decorator

    @tool_decorator
    def test_add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    llm = get_chat_model().bind_tools([test_add])
    response = llm.invoke([HumanMessage(content="What is 7 + 3?")])
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["name"] == "test_add"
    print("  PASS: model tool calling")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TOOL CALLING V6 — AUTOMATED TESTS")
    print("=" * 60)

    # --- Offline tests (no API needed) ---
    print("\n[1/3] Tool & Logic Tests (offline):")
    test_tool_python_exec()
    test_tool_python_exec_error()
    test_tool_calculator()
    test_tool_read_write_file()
    test_tool_list_directory()
    test_extract_text()
    test_graph_compiles()
    test_route_coder_with_tool_calls()
    test_route_reviewer_with_tool_calls()
    test_should_revise()

    # --- Online tests (need API) ---
    print("\n[2/3] API Connection Tests:")
    test_model_connection()
    test_model_tool_calling()

    # --- Skip the slow timeout test by default ---
    print("\n[3/3] Slow Tests (skipped by default):")
    print("  SKIP: python_exec timeout (takes 30s+)")
    # Uncomment to run: test_tool_python_exec_timeout()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
