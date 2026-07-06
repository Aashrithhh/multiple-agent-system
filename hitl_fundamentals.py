"""
=============================================================================
HUMAN-IN-THE-LOOP — FUNDAMENTALS (Learn Before the Full Agent)
=============================================================================

This file teaches the three HITL patterns step-by-step:
  1. Basic interrupt and resume
  2. Approve/Reject a tool call
  3. Edit a tool call before execution
  4. Provide human feedback mid-execution

Key concept: HITL requires CHECKPOINTING.
  - Without checkpoints, graph state is lost when execution pauses.
  - With checkpoints, state is saved so we can resume later.

=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from model_config import get_chat_model


# =============================================================================
# TOOLS (reused from V6)
# =============================================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.
    
    This is a DANGEROUS action — once sent, it cannot be undone.
    Use this tool when you need to send an email to someone.
    """
    # In production, this would call an email API
    return f"Email sent to {to} with subject '{subject}'"


@tool
def delete_file(file_path: str) -> str:
    """Delete a file permanently.
    
    WARNING: This action is irreversible.
    """
    # In production, this would actually delete
    return f"File '{file_path}' deleted permanently"


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe math expression. This is a SAFE action."""
    try:
        # Restricted namespace for safety
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# SHARED STATE
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# =============================================================================
# LESSON 1: Basic Interrupt and Resume
# =============================================================================
# The simplest HITL pattern: pause the graph before a specific node,
# then resume it when the human says "go ahead".
#
# KEY REQUIREMENT: You MUST use a checkpointer (MemorySaver) for interrupts.
# Without it, the graph can't save state and resume later.
# =============================================================================

def lesson_1_basic_interrupt():
    """Demonstrate pausing and resuming execution."""
    print("=" * 60)
    print("LESSON 1: Basic Interrupt and Resume")
    print("=" * 60)
    
    tools = [send_email, delete_file, calculate]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def agent(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: State) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    # Build graph
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    # CRITICAL: compile with checkpointer AND interrupt_before
    # MemorySaver stores state in memory (for production, use SqliteSaver or PostgresSaver)
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"]  # PAUSE before executing any tool
    )
    
    # Each conversation needs a unique thread_id for the checkpointer
    config = {"configurable": {"thread_id": "lesson-1"}}
    
    # --- First invocation: agent decides to use a tool, then PAUSES ---
    print("\n[1] Sending task to agent...")
    result = app.invoke(
        {"messages": [HumanMessage(content="Send an email to boss@company.com saying I'll be late today")]},
        config
    )
    
    # The graph paused! Let's see what tool the agent wants to call
    print("\n[2] Graph PAUSED before tool execution!")
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            print(f"    Pending tool: {tc['name']}")
            print(f"    Args: {tc['args']}")
    
    # --- Human decision: approve (resume with None) ---
    print("\n[3] Human APPROVES. Resuming execution...")
    result = app.invoke(None, config)  # None = "continue where you left off"
    
    # Now the tool was executed
    print("\n[4] Execution complete!")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"    Tool result: {msg.content}")
    
    # Print final response
    final = result["messages"][-1]
    content = final.content if isinstance(final.content, str) else str(final.content)
    print(f"    Agent says: {content[:200]}")


# =============================================================================
# LESSON 2: Approve vs Reject
# =============================================================================
# Sometimes you want to REJECT a tool call entirely.
# To reject: modify the state to remove the tool call and add feedback.
# =============================================================================

def lesson_2_approve_reject():
    """Demonstrate approving and rejecting tool calls."""
    print("\n" + "=" * 60)
    print("LESSON 2: Approve vs Reject")
    print("=" * 60)
    
    tools = [send_email, delete_file, calculate]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def agent(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: State) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"]
    )
    
    # --- SCENARIO: Agent wants to delete a file. Human REJECTS. ---
    config = {"configurable": {"thread_id": "lesson-2-reject"}}
    
    print("\n[1] Agent asked to delete an important file...")
    result = app.invoke(
        {"messages": [HumanMessage(content="Delete the file /etc/passwd")]},
        config
    )
    
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print(f"    Agent wants to call: {last_msg.tool_calls[0]['name']}")
        print(f"    Args: {last_msg.tool_calls[0]['args']}")
    
    # REJECT: Instead of resuming, we update state with feedback
    print("\n[2] Human REJECTS! Adding feedback to state...")
    
    # To reject: provide ToolMessage(s) with rejection info for each tool_call
    # Then the agent will see the rejection and respond accordingly
    rejection_messages = []
    for tc in last_msg.tool_calls:
        rejection_messages.append(
            ToolMessage(
                content="REJECTED BY HUMAN: This action is not allowed. "
                        "Deleting system files is dangerous and prohibited.",
                tool_call_id=tc["id"]
            )
        )
    
    # Update the graph state with rejection messages
    app.update_state(config, {"messages": rejection_messages})
    
    # Resume — agent sees the rejection and responds
    print("\n[3] Resuming with rejection feedback...")
    result = app.invoke(None, config)
    
    final = result["messages"][-1]
    content = final.content if isinstance(final.content, str) else str(final.content)
    print(f"    Agent responds: {content[:300]}")


# =============================================================================
# LESSON 3: Edit Tool Call Arguments
# =============================================================================
# Sometimes the agent has the right idea but wrong details.
# You can EDIT the tool call arguments before letting it execute.
# =============================================================================

def lesson_3_edit_tool_call():
    """Demonstrate editing tool call arguments before execution."""
    print("\n" + "=" * 60)
    print("LESSON 3: Edit Tool Call Arguments")
    print("=" * 60)
    
    tools = [send_email, calculate]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def agent(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: State) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"]
    )
    
    config = {"configurable": {"thread_id": "lesson-3-edit"}}
    
    # Agent wants to send email but we want to change the recipient
    print("\n[1] Asking agent to send email...")
    result = app.invoke(
        {"messages": [HumanMessage(content="Send an email to john@example.com saying the meeting is at 3pm")]},
        config
    )
    
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        original_args = last_msg.tool_calls[0]["args"]
        print(f"    Original tool call: {last_msg.tool_calls[0]['name']}")
        print(f"    Original args: {original_args}")
    
    # EDIT: Change the recipient
    print("\n[2] Human EDITS: changing recipient to team@example.com...")
    
    # Create a new AIMessage with modified tool calls
    edited_tool_calls = []
    for tc in last_msg.tool_calls:
        edited_args = dict(tc["args"])
        edited_args["to"] = "team@example.com"  # Changed!
        edited_tool_calls.append({
            "name": tc["name"],
            "args": edited_args,
            "id": tc["id"],
            "type": "tool_call"
        })
    
    edited_message = AIMessage(content="", tool_calls=edited_tool_calls)
    
    # Update state: replace the last message (the AI's tool call) with our edited version
    # as_node="agent" means "pretend this update came from the agent node"
    app.update_state(config, {"messages": [edited_message]}, as_node="agent")
    
    # Resume execution with edited args
    print("\n[3] Resuming with edited tool call...")
    result = app.invoke(None, config)
    
    # Check the tool result
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"    Tool result: {msg.content}")
    
    final = result["messages"][-1]
    content = final.content if isinstance(final.content, str) else str(final.content)
    print(f"    Agent says: {content[:200]}")


# =============================================================================
# LESSON 4: Human Feedback Loop
# =============================================================================
# Instead of approve/reject/edit, the human provides open-ended FEEDBACK
# that the agent uses to adjust its approach.
#
# Pattern: agent → interrupt → human provides guidance → agent retries
# =============================================================================

def lesson_4_feedback_loop():
    """Demonstrate a human feedback loop."""
    print("\n" + "=" * 60)
    print("LESSON 4: Human Feedback Loop")
    print("=" * 60)
    
    tools = [send_email, calculate]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def agent(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: State) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"]
    )
    
    config = {"configurable": {"thread_id": "lesson-4-feedback"}}
    
    print("\n[1] Agent tries to send email...")
    result = app.invoke(
        {"messages": [HumanMessage(content="Email alice@company.com about the project deadline")]},
        config
    )
    
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print(f"    Agent wants: {last_msg.tool_calls[0]['name']}")
        print(f"    Args: {last_msg.tool_calls[0]['args']}")
    
    # Human provides feedback instead of approving
    print("\n[2] Human provides feedback: 'Make the tone more formal and mention Friday deadline'")
    
    # Reject current tool calls and provide feedback as ToolMessages + HumanMessage
    feedback_messages = []
    for tc in last_msg.tool_calls:
        feedback_messages.append(
            ToolMessage(
                content="REJECTED: Please revise. Make the tone more formal and mention the deadline is Friday.",
                tool_call_id=tc["id"]
            )
        )
    
    # Update state with the rejection + feedback
    app.update_state(config, {"messages": feedback_messages})
    
    # Resume — agent will see the feedback and try again
    print("\n[3] Agent revises based on feedback...")
    result = app.invoke(None, config)
    
    # Check what the agent does now
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print(f"    Revised tool call: {last_msg.tool_calls[0]['name']}")
        print(f"    Revised args: {last_msg.tool_calls[0]['args']}")
        
        # This time, approve it
        print("\n[4] Human APPROVES the revised version!")
        result = app.invoke(None, config)
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                print(f"    Tool result: {msg.content}")
    else:
        content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        print(f"    Agent responded: {content[:300]}")


# =============================================================================
# RUN ALL LESSONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HUMAN-IN-THE-LOOP FUNDAMENTALS")
    print("=" * 60)
    
    lesson_1_basic_interrupt()
    lesson_2_approve_reject()
    lesson_3_edit_tool_call()
    lesson_4_feedback_loop()
    
    print("\n\n" + "=" * 60)
    print("ALL HITL LESSONS COMPLETED")
    print("=" * 60)
