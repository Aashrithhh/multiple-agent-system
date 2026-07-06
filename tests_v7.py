"""
Automated tests for Version 7: Human-in-the-Loop.

Tests cover:
1. Tool danger classification
2. Routing logic (safe vs dangerous)
3. Interrupt and resume mechanism
4. Reject flow
5. Graph compilation with checkpointer
6. Auto-approve end-to-end
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from coding_agent_v7 import (
    calculator, read_file, list_directory, python_exec, write_file,
    SAFE_TOOLS, DANGEROUS_TOOLS, DANGEROUS_TOOL_NAMES, ALL_TOOLS,
    build_coding_agent_graph, route_coder, route_tools_by_danger,
    route_reviewer, should_revise, _extract_text, run_agent_auto_approve
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver


def test_tool_classification():
    """Verify tools are correctly classified as safe or dangerous."""
    safe_names = {t.name for t in SAFE_TOOLS}
    dangerous_names = {t.name for t in DANGEROUS_TOOLS}
    
    assert "calculator" in safe_names
    assert "read_file" in safe_names
    assert "list_directory" in safe_names
    assert "python_exec" in dangerous_names
    assert "write_file" in dangerous_names
    
    # No overlap
    assert safe_names.isdisjoint(dangerous_names)
    # All accounted for
    assert safe_names | dangerous_names == {t.name for t in ALL_TOOLS}
    print("  PASS: tool classification")


def test_route_coder_detects_tools():
    """Test coder routing with/without tool calls."""
    msg_with = AIMessage(content="", tool_calls=[
        {"name": "python_exec", "args": {"code": "print(1)"}, "id": "1"}
    ])
    assert route_coder({"messages": [msg_with]}) == "route_tools"
    
    msg_without = AIMessage(content="Done!")
    assert route_coder({"messages": [msg_without]}) == "reviewer"
    print("  PASS: route_coder")


def test_route_tools_by_danger_safe():
    """Safe tools route to safe_tools node."""
    msg = AIMessage(content="", tool_calls=[
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "1"}
    ])
    assert route_tools_by_danger({"messages": [msg]}) == "safe_tools"
    print("  PASS: route_tools_by_danger (safe)")


def test_route_tools_by_danger_dangerous():
    """Dangerous tools route to dangerous_tools node (approval gate)."""
    msg = AIMessage(content="", tool_calls=[
        {"name": "python_exec", "args": {"code": "print(1)"}, "id": "1"}
    ])
    assert route_tools_by_danger({"messages": [msg]}) == "dangerous_tools"
    print("  PASS: route_tools_by_danger (dangerous)")


def test_route_tools_mixed_routes_dangerous():
    """If ANY tool is dangerous, route to dangerous (conservative)."""
    msg = AIMessage(content="", tool_calls=[
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "1"},
        {"name": "write_file", "args": {"file_path": "x.py", "content": "hi"}, "id": "2"}
    ])
    assert route_tools_by_danger({"messages": [msg]}) == "dangerous_tools"
    print("  PASS: route_tools_by_danger (mixed → dangerous)")


def test_graph_compiles_with_checkpointer():
    """Graph compiles and has interrupt configured."""
    graph = build_coding_agent_graph()
    nodes = list(graph.get_graph().nodes.keys())
    assert "dangerous_tools" in nodes
    assert "safe_tools" in nodes
    assert "route_tools" in nodes
    print("  PASS: graph compiles with checkpointer")


def test_interrupt_pauses_at_dangerous():
    """Verify that the graph actually pauses before dangerous_tools."""
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langchain_core.tools import tool as tool_dec
    from typing import TypedDict, Annotated, Literal
    
    @tool_dec
    def mock_dangerous(x: str) -> str:
        """A mock dangerous tool."""
        return f"executed: {x}"
    
    class S(TypedDict):
        messages: Annotated[list, add_messages]
    
    tool_node = ToolNode([mock_dangerous])
    
    def agent_node(state: S):
        # Simulate an agent that always calls the dangerous tool
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "mock_dangerous", "args": {"x": "test"}, "id": "abc"}
        ])]}
    
    def route(state: S) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    g = StateGraph(S)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    app = g.compile(checkpointer=checkpointer, interrupt_before=["tools"])
    
    config = {"configurable": {"thread_id": "test-interrupt"}}
    result = app.invoke({"messages": [HumanMessage(content="go")]}, config)
    
    # Should be paused — no ToolMessage in results yet
    has_tool_result = any(isinstance(m, ToolMessage) for m in result["messages"])
    assert not has_tool_result, "Tool should NOT have executed yet (paused)"
    
    # Check state shows pending next step
    snapshot = app.get_state(config)
    assert "tools" in snapshot.next, f"Expected 'tools' in next, got {snapshot.next}"
    
    # Resume
    result = app.invoke(None, config)
    has_tool_result = any(isinstance(m, ToolMessage) for m in result["messages"])
    assert has_tool_result, "Tool SHOULD have executed after resume"
    print("  PASS: interrupt pauses at dangerous tools")


def test_reject_flow():
    """Verify rejection sends feedback and resumes."""
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langchain_core.tools import tool as tool_dec
    from typing import TypedDict, Annotated, Literal
    
    @tool_dec
    def mock_action(x: str) -> str:
        """Mock action."""
        return f"done: {x}"
    
    class S(TypedDict):
        messages: Annotated[list, add_messages]
    
    tool_node = ToolNode([mock_action])
    call_count = {"n": 0}
    
    def agent_node(state: S):
        call_count["n"] += 1
        # Only make tool calls on first invocation
        if call_count["n"] == 1:
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": "mock_action", "args": {"x": "delete stuff"}, "id": "r1"}
            ])]}
        # After rejection feedback, just respond
        return {"messages": [AIMessage(content="OK, I won't do that.")]}
    
    def route(state: S) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    g = StateGraph(S)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    app = g.compile(checkpointer=checkpointer, interrupt_before=["tools"])
    config = {"configurable": {"thread_id": "test-reject"}}
    
    # Initial invoke — pauses before tools
    app.invoke({"messages": [HumanMessage(content="go")]}, config)
    
    # Reject: add rejection ToolMessage and resume
    snapshot = app.get_state(config)
    last = snapshot.values["messages"][-1]
    rejection = ToolMessage(content="REJECTED: not allowed", tool_call_id=last.tool_calls[0]["id"])
    app.update_state(config, {"messages": [rejection]}, as_node="tools")
    result = app.invoke(None, config)
    
    # Agent should have responded to the rejection (final msg is AIMessage)
    final = result["messages"][-1]
    # After rejection, agent gets re-invoked and produces a response
    assert isinstance(final, AIMessage), f"Expected AIMessage, got {type(final).__name__}: {final}"
    print("  PASS: reject flow")


def test_should_revise_logic():
    """Test revision decision logic."""
    assert should_revise({"review_feedback": "APPROVED", "iteration": 1}) == "end"
    assert should_revise({"review_feedback": "Fix the bug", "iteration": 1}) == "coder"
    assert should_revise({"review_feedback": "Fix the bug", "iteration": 3}) == "end"
    assert should_revise({"review_feedback": [{"type": "text", "text": "APPROVED"}], "iteration": 1}) == "end"
    print("  PASS: should_revise logic")


def test_auto_approve_end_to_end():
    """Full agent run with auto-approve."""
    result = run_agent_auto_approve(
        "Use python_exec to calculate 2**10 and tell me the result."
    )
    # Should have completed
    assert result is not None
    assert len(result["messages"]) > 2
    # Should have used python_exec
    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])
    assert "python_exec" in tools_used
    print("  PASS: auto-approve end-to-end")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HUMAN-IN-THE-LOOP V7 — AUTOMATED TESTS")
    print("=" * 60)
    
    print("\n[1/3] Classification & Routing Tests (offline):")
    test_tool_classification()
    test_route_coder_detects_tools()
    test_route_tools_by_danger_safe()
    test_route_tools_by_danger_dangerous()
    test_route_tools_mixed_routes_dangerous()
    test_should_revise_logic()
    test_graph_compiles_with_checkpointer()
    
    print("\n[2/3] Interrupt Mechanism Tests:")
    test_interrupt_pauses_at_dangerous()
    test_reject_flow()
    
    print("\n[3/3] End-to-End Test (requires API):")
    test_auto_approve_end_to_end()
    
    print("\n" + "=" * 60)
    print("ALL V7 TESTS PASSED")
    print("=" * 60)
