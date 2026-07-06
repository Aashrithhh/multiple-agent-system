"""
=============================================================================
STREAMING — FUNDAMENTALS
=============================================================================

Three levels of streaming in LangGraph:
  1. Node-level: see output after each node completes
  2. Update-level: see state changes at each step
  3. Token-level: see LLM tokens as they're generated

=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import asyncio
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from model_config import get_chat_model


# =============================================================================
# SETUP: Simple agent for demonstrations
# =============================================================================

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return str(a * b)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_agent():
    """Build a simple tool-calling agent."""
    tools = [add, multiply]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def route(state: State) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# LESSON 1: Node-Level Streaming (.stream())
# =============================================================================
# graph.stream() yields output AFTER each node completes.
# You see: "planner finished", "coder finished", "tools finished", etc.
#
# This is the SIMPLEST form of streaming.
# Perfect for showing progress in a UI: "Step 1 done... Step 2 done..."
# =============================================================================

def lesson_1_node_streaming():
    """Stream node-by-node output."""
    print("=" * 60)
    print("LESSON 1: Node-Level Streaming (.stream())")
    print("=" * 60)

    app = _build_agent()
    config = {"configurable": {"thread_id": "stream-1"}}

    print("\nTask: 'What is (5 + 3) * 2? Use tools step by step.'\n")
    print("-" * 40)

    # .stream() yields a dict {node_name: output} after each node
    for event in app.stream(
        {"messages": [HumanMessage(content="What is (5 + 3) * 2? Use tools step by step.")]},
        config
    ):
        # event is {node_name: node_output}
        for node_name, output in event.items():
            print(f"\n[Node: {node_name}]")
            if "messages" in output:
                last_msg = output["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"  -> Tool call: {tc['name']}({tc['args']})")
                elif isinstance(last_msg, ToolMessage):
                    print(f"  -> Tool result: {last_msg.content}")
                else:
                    content = last_msg.content
                    if isinstance(content, list):
                        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                    if content:
                        print(f"  -> {content[:200]}")

    print("\n" + "-" * 40)
    print("Each [Node: ...] appeared as soon as that node finished.")


# =============================================================================
# LESSON 2: Stream Mode "updates" vs "values"
# =============================================================================
# LangGraph supports different stream modes:
#   - "values": yields the FULL state after each node
#   - "updates": yields only the CHANGES from each node (default)
#
# "updates" = more efficient (less data)
# "values" = easier to work with (full state every time)
# =============================================================================

def lesson_2_stream_modes():
    """Compare stream_mode='values' vs 'updates'."""
    print("\n" + "=" * 60)
    print("LESSON 2: Stream Modes (values vs updates)")
    print("=" * 60)

    app = _build_agent()
    config = {"configurable": {"thread_id": "stream-2"}}

    # MODE: "updates" (default) — only changes
    print("\n--- Mode: 'updates' (shows only what changed) ---")
    for event in app.stream(
        {"messages": [HumanMessage(content="What is 7 + 8?")]},
        config,
        stream_mode="updates"
    ):
        for node_name, output in event.items():
            msg_count = len(output.get("messages", []))
            print(f"  [{node_name}] produced {msg_count} new message(s)")

    # MODE: "values" — full state snapshot
    config2 = {"configurable": {"thread_id": "stream-2b"}}
    print("\n--- Mode: 'values' (shows full state each time) ---")
    for state_snapshot in app.stream(
        {"messages": [HumanMessage(content="What is 7 + 8?")]},
        config2,
        stream_mode="values"
    ):
        total_msgs = len(state_snapshot.get("messages", []))
        print(f"  State now has {total_msgs} total messages")


# =============================================================================
# LESSON 3: Async Streaming with astream()
# =============================================================================
# For web servers (FastAPI, etc.), you need ASYNC streaming.
# graph.astream() is the async version of .stream().
#
# This is what production apps use:
#   - FastAPI StreamingResponse
#   - WebSocket push
#   - Server-Sent Events (SSE)
# =============================================================================

async def _lesson_3_async():
    """Async streaming demonstration."""
    app = _build_agent()
    config = {"configurable": {"thread_id": "stream-3"}}

    print("\n--- Async streaming (same output, but non-blocking) ---")
    async for event in app.astream(
        {"messages": [HumanMessage(content="What is 10 + 20?")]},
        config
    ):
        for node_name, output in event.items():
            if "messages" in output:
                last = output["messages"][-1]
                if isinstance(last, ToolMessage):
                    print(f"  [{node_name}] Tool result: {last.content}")
                elif hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"  [{node_name}] Calling: {[tc['name'] for tc in last.tool_calls]}")
                else:
                    content = last.content
                    if isinstance(content, list):
                        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                    print(f"  [{node_name}] Response: {content[:100]}")


def lesson_3_async_streaming():
    """Wrapper to run async lesson."""
    print("\n" + "=" * 60)
    print("LESSON 3: Async Streaming (astream)")
    print("=" * 60)
    asyncio.run(_lesson_3_async())
    print("\n  In production, this feeds into FastAPI StreamingResponse")


# =============================================================================
# LESSON 4: Event-Level Streaming (astream_events)
# =============================================================================
# The most GRANULAR streaming. Shows EVERY internal event:
#   - LLM starts generating
#   - Each token produced
#   - Tool call initiated
#   - Tool call completed
#
# Use this for:
#   - Real-time token display (ChatGPT-style)
#   - Detailed progress bars
#   - Monitoring dashboards
# =============================================================================

async def _lesson_4_events():
    """Event-level streaming."""
    app = _build_agent()
    config = {"configurable": {"thread_id": "stream-4"}}

    print("\n--- Event stream (granular internal events) ---")
    event_count = 0
    async for event in app.astream_events(
        {"messages": [HumanMessage(content="What is 3 + 4?")]},
        config=config,
        version="v2"
    ):
        kind = event["event"]
        # Filter to interesting events only
        if kind == "on_chat_model_start":
            print(f"  [EVENT] LLM started generating...")
        elif kind == "on_chat_model_end":
            print(f"  [EVENT] LLM finished generating")
        elif kind == "on_tool_start":
            print(f"  [EVENT] Tool '{event['name']}' starting...")
        elif kind == "on_tool_end":
            output = event.get("data", {}).get("output", "")
            print(f"  [EVENT] Tool '{event['name']}' done: {str(output)[:80]}")
        elif kind == "on_chat_model_stream":
            # This is individual token streaming
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                if isinstance(content, str) and content.strip():
                    print(f"  [TOKEN] {content}", end="")
        event_count += 1

    print(f"\n\n  Total events captured: {event_count}")


def lesson_4_event_streaming():
    """Wrapper for event streaming lesson."""
    print("\n" + "=" * 60)
    print("LESSON 4: Event-Level Streaming (astream_events)")
    print("=" * 60)
    asyncio.run(_lesson_4_events())


# =============================================================================
# LESSON 5: Streaming for Production (SSE format)
# =============================================================================
# In production, streaming is typically delivered via Server-Sent Events (SSE).
# Here's what the output format looks like for a web client.
# =============================================================================

def lesson_5_production_format():
    """Show what production streaming output looks like."""
    print("\n" + "=" * 60)
    print("LESSON 5: Production Streaming Format (SSE)")
    print("=" * 60)

    app = _build_agent()
    config = {"configurable": {"thread_id": "stream-5"}}

    print("\n--- Server-Sent Events format ---\n")

    for event in app.stream(
        {"messages": [HumanMessage(content="What is 6 * 7?")]},
        config,
        stream_mode="updates"
    ):
        for node_name, output in event.items():
            # Format as SSE (what you'd send over HTTP)
            if "messages" in output:
                last = output["messages"][-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    for tc in last.tool_calls:
                        print(f'data: {{"type": "tool_call", "node": "{node_name}", "tool": "{tc["name"]}", "args": {tc["args"]}}}')
                elif isinstance(last, ToolMessage):
                    print(f'data: {{"type": "tool_result", "node": "{node_name}", "result": "{last.content}"}}')
                else:
                    content = last.content
                    if isinstance(content, list):
                        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                    if content:
                        safe_content = content[:100].replace('"', '\\"').replace('\n', ' ')
                        print(f'data: {{"type": "message", "node": "{node_name}", "content": "{safe_content}"}}')
            print()  # Empty line between SSE events

    print('data: {"type": "done"}')
    print("\n  ^ This is what your frontend JavaScript receives via EventSource")


# =============================================================================
# RUN ALL LESSONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STREAMING FUNDAMENTALS")
    print("=" * 60)

    lesson_1_node_streaming()
    lesson_2_stream_modes()
    lesson_3_async_streaming()
    lesson_4_event_streaming()
    lesson_5_production_format()

    print("\n\n" + "=" * 60)
    print("ALL STREAMING LESSONS COMPLETED")
    print("=" * 60)
