"""
=============================================================================
CHECKPOINTING & PERSISTENCE — FUNDAMENTALS
=============================================================================

This file teaches checkpointing step-by-step:
  1. MemorySaver basics (state survives between invocations)
  2. Thread isolation (multiple conversations)
  3. State history and time travel
  4. SqliteSaver (survives process restarts)
  5. Resuming from a specific checkpoint

Key insight: Checkpointing is what makes agents STATEFUL across time.
Without it, every invoke() is a fresh start.

=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import tempfile
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from model_config import get_chat_model


# =============================================================================
# SHARED TOOLS AND STATE
# =============================================================================

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_simple_agent():
    """Build a simple agent graph (reused across lessons)."""
    tools = [add]
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
    return graph


# =============================================================================
# LESSON 1: MemorySaver — State Persists Between Invocations
# =============================================================================
# Without a checkpointer: every invoke() starts fresh.
# With a checkpointer: state accumulates across invocations.
#
# This is how "memory" in chatbots actually works under the hood.
# The LLM doesn't remember — the CHECKPOINTER remembers for it.
# =============================================================================

def lesson_1_memory_saver():
    """Show how MemorySaver keeps state between invocations."""
    print("=" * 60)
    print("LESSON 1: MemorySaver — State Persists")
    print("=" * 60)

    graph = _build_simple_agent()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # Same thread_id = same conversation
    config = {"configurable": {"thread_id": "user-alice"}}

    # First message
    print("\n[Turn 1]")
    result = app.invoke({"messages": [HumanMessage(content="Hi, my name is Alice")]}, config)
    response = result["messages"][-1]
    content = response.content if isinstance(response.content, str) else str(response.content)
    print(f"  User: Hi, my name is Alice")
    print(f"  Agent: {content[:150]}")

    # Second message — agent REMEMBERS the name because state was checkpointed
    print("\n[Turn 2]")
    result = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
    response = result["messages"][-1]
    content = response.content if isinstance(response.content, str) else str(response.content)
    print(f"  User: What's my name?")
    print(f"  Agent: {content[:150]}")

    # Verify: the full message history is preserved
    state = app.get_state(config)
    print(f"\n  Total messages in state: {len(state.values['messages'])}")
    print(f"  (2 human + 2 AI = 4 messages accumulated)")


# =============================================================================
# LESSON 2: Thread Isolation — Multiple Conversations
# =============================================================================
# Each thread_id creates a completely independent conversation.
# This is how production apps handle multiple users simultaneously.
#
# thread_id is like a SESSION ID in web applications.
# =============================================================================

def lesson_2_thread_isolation():
    """Show that different thread_ids have independent state."""
    print("\n" + "=" * 60)
    print("LESSON 2: Thread Isolation")
    print("=" * 60)

    graph = _build_simple_agent()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # Two different users, two different threads
    config_alice = {"configurable": {"thread_id": "alice-session"}}
    config_bob = {"configurable": {"thread_id": "bob-session"}}

    # Alice's conversation
    print("\n[Alice's thread]")
    app.invoke({"messages": [HumanMessage(content="My favorite color is blue")]}, config_alice)
    print("  Alice: My favorite color is blue")

    # Bob's conversation (completely independent)
    print("\n[Bob's thread]")
    app.invoke({"messages": [HumanMessage(content="My favorite food is pizza")]}, config_bob)
    print("  Bob: My favorite food is pizza")

    # Verify isolation: ask each thread about the other's info
    print("\n[Testing isolation]")
    result_alice = app.invoke(
        {"messages": [HumanMessage(content="What's my favorite color?")]},
        config_alice
    )
    alice_response = result_alice["messages"][-1]
    a_content = alice_response.content if isinstance(alice_response.content, str) else str(alice_response.content)
    print(f"  Alice asks 'What's my favorite color?': {a_content[:100]}")

    result_bob = app.invoke(
        {"messages": [HumanMessage(content="What's my favorite food?")]},
        config_bob
    )
    bob_response = result_bob["messages"][-1]
    b_content = bob_response.content if isinstance(bob_response.content, str) else str(bob_response.content)
    print(f"  Bob asks 'What's my favorite food?': {b_content[:100]}")

    # State counts
    alice_state = app.get_state(config_alice)
    bob_state = app.get_state(config_bob)
    print(f"\n  Alice's messages: {len(alice_state.values['messages'])}")
    print(f"  Bob's messages: {len(bob_state.values['messages'])}")


# =============================================================================
# LESSON 3: State History and Time Travel
# =============================================================================
# LangGraph saves a checkpoint AFTER EVERY NODE execution.
# You can inspect the entire history and even resume from a past state.
#
# This is incredibly powerful for:
#   - Debugging: "What exactly happened at step 3?"
#   - Recovery: "Go back to before the agent made a mistake"
#   - Auditing: "Show me every decision the agent made"
# =============================================================================

def lesson_3_state_history():
    """Show how to inspect and traverse checkpoint history."""
    print("\n" + "=" * 60)
    print("LESSON 3: State History & Time Travel")
    print("=" * 60)

    graph = _build_simple_agent()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "history-demo"}}

    # Have a multi-turn conversation
    app.invoke({"messages": [HumanMessage(content="What is 5 + 3?")]}, config)
    app.invoke({"messages": [HumanMessage(content="Now add 10 to that result")]}, config)

    # Get the FULL history of checkpoints
    print("\n[Checkpoint History] (newest first)")
    print("-" * 50)

    history = list(app.get_state_history(config))
    for i, state in enumerate(history):
        msg_count = len(state.values.get("messages", []))
        next_nodes = state.next
        checkpoint_id = state.config["configurable"]["checkpoint_id"]
        print(f"  [{i}] checkpoint={checkpoint_id[:12]}... | msgs={msg_count} | next={next_nodes}")

    print(f"\n  Total checkpoints: {len(history)}")
    print("  (One checkpoint per node execution)")

    # TIME TRAVEL: resume from an earlier checkpoint
    if len(history) > 2:
        old_state = history[-3]  # Pick an earlier state
        old_config = old_state.config
        print(f"\n[Time Travel] Resuming from checkpoint {old_config['configurable']['checkpoint_id'][:12]}...")
        print(f"  That state had {len(old_state.values['messages'])} messages")
        # You could now invoke with this config to branch from that point


# =============================================================================
# LESSON 4: SqliteSaver — Survives Process Restarts
# =============================================================================
# MemorySaver loses everything when your Python process exits.
# SqliteSaver writes to a .db file on disk — state survives restarts.
#
# This simulates what happens in production:
#   1. User starts a conversation
#   2. Server restarts (deploy, crash, etc.)
#   3. User comes back — conversation continues seamlessly
# =============================================================================

def lesson_4_sqlite_persistence():
    """Show how SqliteSaver persists state to disk."""
    print("\n" + "=" * 60)
    print("LESSON 4: SqliteSaver — Disk Persistence")
    print("=" * 60)

    from langgraph.checkpoint.sqlite import SqliteSaver

    # Create a temporary database file
    db_path = os.path.join(tempfile.gettempdir(), "langgraph_lesson4.db")

    # --- PHASE 1: First "server session" ---
    print("\n[Phase 1] First server session — saving state to disk")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = _build_simple_agent()
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "persistent-thread"}}

        result = app.invoke(
            {"messages": [HumanMessage(content="Remember: the secret code is ALPHA-7")]},
            config
        )
        response = result["messages"][-1]
        content = response.content if isinstance(response.content, str) else str(response.content)
        print(f"  User: Remember: the secret code is ALPHA-7")
        print(f"  Agent: {content[:150]}")
        print(f"  State saved to: {db_path}")

    # At this point, the SqliteSaver is closed. In production, this is like a server restart.
    print("\n  [--- Simulating server restart ---]")

    # --- PHASE 2: Second "server session" — state is recovered from disk ---
    print("\n[Phase 2] New server session — recovering state from disk")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = _build_simple_agent()
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "persistent-thread"}}

        # Ask about something from the previous session
        result = app.invoke(
            {"messages": [HumanMessage(content="What's the secret code I told you?")]},
            config
        )
        response = result["messages"][-1]
        content = response.content if isinstance(response.content, str) else str(response.content)
        print(f"  User: What's the secret code?")
        print(f"  Agent: {content[:150]}")

        state = app.get_state(config)
        print(f"\n  Total messages recovered: {len(state.values['messages'])}")
        print(f"  (Previous session's messages are still there!)")

    # Cleanup
    os.unlink(db_path)
    print(f"\n  Cleaned up test database")


# =============================================================================
# LESSON 5: Resuming from a Specific Checkpoint (Branching)
# =============================================================================
# You can resume execution from ANY past checkpoint.
# This creates a "branch" — like git branches for agent state.
#
# Use case: "The agent went wrong at step 3. Let me go back to step 2
# and give it different input."
# =============================================================================

def lesson_5_resume_from_checkpoint():
    """Show how to branch from a specific past state."""
    print("\n" + "=" * 60)
    print("LESSON 5: Resume from Specific Checkpoint (Branching)")
    print("=" * 60)

    graph = _build_simple_agent()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "branch-demo"}}

    # Build up a conversation
    print("\n[Building conversation]")
    app.invoke({"messages": [HumanMessage(content="I'm working on project X")]}, config)
    print("  Turn 1: I'm working on project X")

    app.invoke({"messages": [HumanMessage(content="The deadline is Friday")]}, config)
    print("  Turn 2: The deadline is Friday")

    app.invoke({"messages": [HumanMessage(content="Actually cancel everything, start project Y instead")]}, config)
    print("  Turn 3: Actually cancel everything, start project Y instead")

    # Oops! Turn 3 was a mistake. Let's go back to after Turn 2.
    print("\n[Finding checkpoint after Turn 2]")
    history = list(app.get_state_history(config))

    # Find the state that had 4 messages (2 human + 2 AI = after turn 2)
    target_state = None
    for state in history:
        msg_count = len(state.values.get("messages", []))
        if msg_count == 4:  # After turn 2 completes
            target_state = state
            break

    if target_state:
        print(f"  Found checkpoint with {len(target_state.values['messages'])} messages")

        # Resume from that checkpoint with DIFFERENT input
        branch_config = target_state.config
        print("\n[Branching: giving different Turn 3]")
        result = app.invoke(
            {"messages": [HumanMessage(content="Let's extend the deadline to next Monday")]},
            branch_config
        )
        response = result["messages"][-1]
        content = response.content if isinstance(response.content, str) else str(response.content)
        print(f"  User: Let's extend the deadline to next Monday")
        print(f"  Agent: {content[:200]}")
        print(f"\n  We branched from the past and took a different path!")
    else:
        print("  Could not find target checkpoint (this is OK in demo)")


# =============================================================================
# RUN ALL LESSONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHECKPOINTING & PERSISTENCE FUNDAMENTALS")
    print("=" * 60)

    lesson_1_memory_saver()
    lesson_2_thread_isolation()
    lesson_3_state_history()
    lesson_4_sqlite_persistence()
    lesson_5_resume_from_checkpoint()

    print("\n\n" + "=" * 60)
    print("ALL CHECKPOINTING LESSONS COMPLETED")
    print("=" * 60)
