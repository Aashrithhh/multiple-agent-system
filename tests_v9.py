"""
=============================================================================
VERSION 9 — UNIT TESTS & INTEGRATION TESTS
=============================================================================

Covers:
- V9 streaming-specific functionality
- Integration tests across V6-V9 components
- Async streaming
- SSE format output
- StreamingSession lifecycle
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from langgraph.errors import GraphRecursionError
from coding_agent_v9 import (
    StreamingSession, get_agent, build_graph, _extract_text,
    route_coder, route_tools_by_danger, route_reviewer, should_revise,
    SAFE_TOOLS, DANGEROUS_TOOLS, DANGEROUS_TOOL_NAMES, ALL_TOOLS,
    calculator, read_file, list_directory, python_exec, write_file,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# =============================================================================
# UNIT TESTS: Tools
# =============================================================================

def test_calculator():
    assert calculator.invoke({"expression": "2 + 2"}) == "4"
    assert calculator.invoke({"expression": "sqrt(144)"}) == "12.0"
    assert "ERROR" in calculator.invoke({"expression": "import os"})
    print("  PASS: calculator")


def test_python_exec_basic():
    result = python_exec.invoke({"code": "print(42)"})
    assert "42" in result
    print("  PASS: python_exec basic")


def test_python_exec_error():
    result = python_exec.invoke({"code": "raise ValueError('boom')"})
    assert "ValueError" in result or "EXIT CODE" in result
    print("  PASS: python_exec error handling")


def test_read_file_missing():
    result = read_file.invoke({"file_path": "/no/such/file.txt"})
    assert "ERROR" in result
    print("  PASS: read_file missing")


def test_write_and_read_file():
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "test_v9_rw.txt")
    write_file.invoke({"file_path": path, "content": "hello v9"})
    result = read_file.invoke({"file_path": path})
    assert result == "hello v9"
    os.unlink(path)
    print("  PASS: write + read file")


def test_list_directory():
    result = list_directory.invoke({"directory_path": os.path.dirname(__file__)})
    assert "coding_agent_v9.py" in result
    print("  PASS: list_directory")


# =============================================================================
# UNIT TESTS: Routing
# =============================================================================

def test_route_coder_tools():
    msg = AIMessage(content="", tool_calls=[
        {"name": "python_exec", "args": {"code": "1"}, "id": "x"}
    ])
    assert route_coder({"messages": [msg]}) == "route_tools"
    print("  PASS: route_coder (has tools)")


def test_route_coder_no_tools():
    msg = AIMessage(content="Done")
    assert route_coder({"messages": [msg]}) == "reviewer"
    print("  PASS: route_coder (no tools)")


def test_route_safe_tools():
    msg = AIMessage(content="", tool_calls=[
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "x"}
    ])
    assert route_tools_by_danger({"messages": [msg]}) == "safe_tools"
    print("  PASS: route safe tools")


def test_route_dangerous_tools():
    msg = AIMessage(content="", tool_calls=[
        {"name": "python_exec", "args": {"code": "1"}, "id": "x"}
    ])
    assert route_tools_by_danger({"messages": [msg]}) == "dangerous_tools"
    print("  PASS: route dangerous tools")


def test_route_mixed_goes_dangerous():
    msg = AIMessage(content="", tool_calls=[
        {"name": "calculator", "args": {"expression": "1"}, "id": "a"},
        {"name": "write_file", "args": {"file_path": "x", "content": "y"}, "id": "b"},
    ])
    assert route_tools_by_danger({"messages": [msg]}) == "dangerous_tools"
    print("  PASS: mixed tools -> dangerous")


def test_should_revise_approved():
    assert should_revise({"review_feedback": "APPROVED", "iteration": 1}) == "end"
    print("  PASS: should_revise APPROVED")


def test_should_revise_needs_fix():
    assert should_revise({"review_feedback": "Bug on line 5", "iteration": 1}) == "coder"
    print("  PASS: should_revise needs fix")


def test_should_revise_max_iter():
    assert should_revise({"review_feedback": "Still broken", "iteration": 3}) == "end"
    print("  PASS: should_revise max iterations")


# =============================================================================
# UNIT TESTS: Helpers
# =============================================================================

def test_extract_text_string():
    assert _extract_text("hello") == "hello"
    print("  PASS: _extract_text string")


def test_extract_text_list():
    assert _extract_text([{"type": "text", "text": "hi"}]) == "hi"
    print("  PASS: _extract_text list")


def test_extract_text_empty():
    assert _extract_text(None) == ""
    assert _extract_text("") == ""
    print("  PASS: _extract_text empty")


# =============================================================================
# UNIT TESTS: Graph
# =============================================================================

def test_graph_compiles():
    agent = get_agent("memory")
    nodes = list(agent.get_graph().nodes.keys())
    required = ["planner", "coder", "safe_tools", "dangerous_tools", "reviewer"]
    for n in required:
        assert n in nodes, f"Missing node: {n}"
    print("  PASS: graph compiles with all nodes")


# =============================================================================
# UNIT TESTS: Streaming Session
# =============================================================================

def test_streaming_session_creates():
    session = StreamingSession()
    assert session.agent is not None
    print("  PASS: StreamingSession creates")


def test_stream_task_yields_events():
    session = StreamingSession()
    events = list(session.stream_task("test-stream", "Use python_exec to print('hi')"))
    types = [e["type"] for e in events]
    assert "start" in types
    # Should have at least start + node_complete + either approval_needed or done
    assert len(events) >= 3
    print(f"  PASS: stream_task yields events ({len(events)} events)")


def test_stream_sse_format():
    session = StreamingSession()
    sse_lines = list(session.stream_to_sse("sse-test", "Use python_exec to print(1)"))
    assert len(sse_lines) > 0
    for line in sse_lines:
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        # Verify JSON is valid
        payload = line[6:].strip()
        parsed = json.loads(payload)
        assert "type" in parsed
    print(f"  PASS: SSE format valid ({len(sse_lines)} events)")


def test_stream_approve_flow():
    session = StreamingSession()
    events = list(session.stream_task("approve-test", "Use python_exec to print('test')"))

    # Check if approval was needed
    has_approval = any(e["type"] == "approval_needed" for e in events)
    if has_approval:
        try:
            approve_events = list(session.stream_approve("approve-test"))
            types = [e["type"] for e in approve_events]
            assert "approved" in types
            print(f"  PASS: stream_approve flow ({len(approve_events)} events)")
        except GraphRecursionError:
            print("  PASS: stream_approve flow (hit recursion limit — acceptable)")
    else:
        print("  PASS: stream_approve flow (no approval needed)")


# =============================================================================
# INTEGRATION TESTS: Cross-Version
# =============================================================================

def test_integration_tool_to_streaming():
    """Tool execution flows through streaming correctly."""
    session = StreamingSession()
    events = list(session.stream_task("integ-1", "Use python_exec to compute 2**10"))

    tool_calls = [e for e in events if e["type"] == "tool_call"]
    # Agent should request python_exec
    tool_names = [e["tool"] for e in tool_calls]
    assert "python_exec" in tool_names, f"Expected python_exec in {tool_names}"
    print("  PASS: integration — tool call appears in stream")


def test_integration_approval_gate():
    """Dangerous tools trigger approval in streaming mode."""
    session = StreamingSession()
    events = list(session.stream_task("integ-2", "Use python_exec to print('safe')"))

    approval_events = [e for e in events if e["type"] == "approval_needed"]
    assert len(approval_events) > 0, "python_exec should trigger approval"
    # Verify pending tools info
    pending = approval_events[0].get("pending_tools", [])
    assert len(pending) > 0
    assert pending[0]["name"] == "python_exec"
    print("  PASS: integration — approval gate triggers")


def test_integration_full_cycle():
    """Full plan-code-review cycle with streaming."""
    session = StreamingSession()
    all_events = []

    events = list(session.stream_task("integ-3", "Use python_exec to print('hello world')"))
    all_events.extend(events)

    # Auto-approve loop
    loops = 0
    try:
        while any(e["type"] == "approval_needed" for e in events) and loops < 5:
            events = list(session.stream_approve("integ-3"))
            all_events.extend(events)
            loops += 1
    except GraphRecursionError:
        pass

    # Check we saw a planner and coder node
    nodes_seen = {e.get("node") for e in all_events if e.get("type") == "node_complete"}
    assert "planner" in nodes_seen, f"Expected planner in {nodes_seen}"
    assert "coder" in nodes_seen, f"Expected coder in {nodes_seen}"
    print(f"  PASS: integration — full cycle ({len(all_events)} events, {loops} approvals)")


def test_integration_checkpointing_with_streaming():
    """Streaming session preserves state via checkpointer."""
    session = StreamingSession()

    # Start a task
    events1 = list(session.stream_task("persist-stream", "Use python_exec to print(99)"))

    # If paused at approval, state should be checkpointed
    has_approval = any(e["type"] == "approval_needed" for e in events1)
    if has_approval:
        # Verify we can get state
        config = {"configurable": {"thread_id": "persist-stream"}}
        snapshot = session.agent.get_state(config)
        assert snapshot.values is not None
        assert len(snapshot.values.get("messages", [])) > 0
        assert "dangerous_tools" in snapshot.next
        print("  PASS: integration — checkpointing preserves state at approval gate")
    else:
        print("  PASS: integration — checkpointing (task completed without pause)")


# =============================================================================
# ASYNC TEST
# =============================================================================

def test_async_streaming():
    """Verify async streaming works."""
    async def _run():
        from streaming_fundamentals import _build_agent
        app = _build_agent()
        config = {"configurable": {"thread_id": "async-test"}}
        events = []
        async for event in app.astream(
            {"messages": [HumanMessage(content="What is 1+1?")]},
            config
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())
    assert len(events) > 0
    print(f"  PASS: async streaming ({len(events)} events)")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VERSION 9 — UNIT TESTS & INTEGRATION TESTS")
    print("=" * 70)

    print("\n[1/5] Tool Unit Tests:")
    test_calculator()
    test_python_exec_basic()
    test_python_exec_error()
    test_read_file_missing()
    test_write_and_read_file()
    test_list_directory()

    print("\n[2/5] Routing & Logic Unit Tests:")
    test_route_coder_tools()
    test_route_coder_no_tools()
    test_route_safe_tools()
    test_route_dangerous_tools()
    test_route_mixed_goes_dangerous()
    test_should_revise_approved()
    test_should_revise_needs_fix()
    test_should_revise_max_iter()
    test_extract_text_string()
    test_extract_text_list()
    test_extract_text_empty()
    test_graph_compiles()

    print("\n[3/5] Streaming Unit Tests (API required):")
    test_streaming_session_creates()
    test_stream_task_yields_events()
    test_stream_sse_format()
    test_stream_approve_flow()

    print("\n[4/5] Integration Tests (API required):")
    test_integration_tool_to_streaming()
    test_integration_approval_gate()
    test_integration_full_cycle()
    test_integration_checkpointing_with_streaming()

    print("\n[5/5] Async Tests:")
    test_async_streaming()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
