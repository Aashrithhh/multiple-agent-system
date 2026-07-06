"""
=============================================================================
TOOL CALLING — FUNDAMENTALS (Learn Before the Full Agent)
=============================================================================

This file teaches tool calling step-by-step, building up from the simplest
possible example to a complete tool-calling loop.

Run each section independently to see how it works.

=============================================================================
"""

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Literal

from model_config import get_chat_model

# =============================================================================
# LESSON 1: Defining a Tool
# =============================================================================
# A tool is just a Python function with:
#   1. @tool decorator
#   2. Type hints (these become the input schema)
#   3. A docstring (this is what the LLM reads to understand the tool)
#
# The LLM NEVER sees your code. It only sees:
#   - Name: "add_numbers"
#   - Description: "Add two numbers together..."
#   - Schema: {a: int, b: int}
# =============================================================================

@tool
def add_numbers(a: int, b: int) -> str:
    """Add two numbers together and return the result.
    
    Use this when you need to perform addition.
    """
    return str(a + b)


@tool
def multiply_numbers(a: int, b: int) -> str:
    """Multiply two numbers together and return the result.
    
    Use this when you need to perform multiplication.
    """
    return str(a * b)


def lesson_1_tool_schema():
    """See what the LLM actually sees about our tools."""
    print("=" * 50)
    print("LESSON 1: Tool Schema (What the LLM Sees)")
    print("=" * 50)
    
    # This is what gets sent to the LLM API
    print(f"\nTool Name: {add_numbers.name}")
    print(f"Tool Description: {add_numbers.description}")
    print(f"Tool Schema: {add_numbers.args_schema.schema()}")
    print()
    print(f"Tool Name: {multiply_numbers.name}")
    print(f"Tool Description: {multiply_numbers.description}")
    print(f"Tool Schema: {multiply_numbers.args_schema.schema()}")


# =============================================================================
# LESSON 2: Binding Tools to an LLM
# =============================================================================
# bind_tools() tells the LLM: "These tools are available to you."
# After binding, the LLM can choose to include tool_calls in its response.
#
# KEY INSIGHT: The LLM does NOT automatically use tools.
# It DECIDES whether a tool is needed based on the user's question.
# =============================================================================

def lesson_2_bind_tools():
    """Show how the LLM decides to use (or not use) tools."""
    print("\n" + "=" * 50)
    print("LESSON 2: Binding Tools & LLM Decision Making")
    print("=" * 50)
    
    llm = get_chat_model()
    
    # Bind our tools
    llm_with_tools = llm.bind_tools([add_numbers, multiply_numbers])
    
    # --- Case A: LLM decides to USE a tool ---
    print("\n--- Case A: Question that needs a tool ---")
    response_a = llm_with_tools.invoke([
        HumanMessage(content="What is 42 + 58?")
    ])
    print(f"Content: {response_a.content}")
    print(f"Tool Calls: {response_a.tool_calls}")
    # Expected: tool_calls will contain add_numbers with args {a: 42, b: 58}
    
    # --- Case B: LLM decides NOT to use a tool ---
    print("\n--- Case B: Question that doesn't need a tool ---")
    response_b = llm_with_tools.invoke([
        HumanMessage(content="What is the capital of France?")
    ])
    print(f"Content: {response_b.content}")
    print(f"Tool Calls: {response_b.tool_calls}")
    # Expected: tool_calls will be empty, content will have the answer


# =============================================================================
# LESSON 3: The Tool Call → Execution → Response Cycle
# =============================================================================
# This is the MOST IMPORTANT concept.
#
# The full cycle:
#   1. User asks question
#   2. LLM responds with tool_calls (NOT the answer yet!)
#   3. WE execute the tool (the LLM doesn't)
#   4. We send the result back as a ToolMessage
#   5. LLM uses the result to give the final answer
#
# This is a MULTI-TURN conversation with the LLM.
# =============================================================================

def lesson_3_manual_tool_execution():
    """Manually execute the full tool-calling cycle."""
    print("\n" + "=" * 50)
    print("LESSON 3: Manual Tool Execution Cycle")
    print("=" * 50)
    
    llm = get_chat_model()
    llm_with_tools = llm.bind_tools([add_numbers, multiply_numbers])
    
    # Step 1: User asks a question
    messages = [HumanMessage(content="What is 15 * 7?")]
    print(f"\n[Step 1] User: {messages[0].content}")
    
    # Step 2: LLM responds with a tool call
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)
    print(f"\n[Step 2] LLM wants to call: {ai_response.tool_calls}")
    
    # Step 3: WE execute the tool
    # The tool_call has: name, args, id
    tool_call = ai_response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]
    
    # Find and execute the right tool
    tool_map = {"add_numbers": add_numbers, "multiply_numbers": multiply_numbers}
    result = tool_map[tool_name].invoke(tool_args)
    print(f"\n[Step 3] We executed {tool_name}({tool_args}) → {result}")
    
    # Step 4: Send result back as ToolMessage
    # CRITICAL: tool_call_id MUST match the id from the tool_call
    tool_message = ToolMessage(content=result, tool_call_id=tool_id)
    messages.append(tool_message)
    print(f"\n[Step 4] Sent ToolMessage back to LLM")
    
    # Step 5: LLM gives final answer
    final_response = llm_with_tools.invoke(messages)
    print(f"\n[Step 5] LLM Final Answer: {final_response.content}")
    print(f"         Tool Calls: {final_response.tool_calls}")  # Should be empty now


# =============================================================================
# LESSON 4: ToolNode — LangGraph's Automatic Executor
# =============================================================================
# Lesson 3 was manual. In practice, we use LangGraph's ToolNode.
# ToolNode automatically:
#   1. Reads tool_calls from the last AIMessage
#   2. Executes the corresponding functions
#   3. Returns ToolMessages with results
#
# This eliminates the manual execution code.
# =============================================================================

# State used by lessons 4 and 5
class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]


def lesson_4_tool_node():
    """Show how ToolNode automates tool execution."""
    print("\n" + "=" * 50)
    print("LESSON 4: ToolNode (Automatic Execution)")
    print("=" * 50)
    
    # Create a ToolNode with our tools
    tool_node = ToolNode([add_numbers, multiply_numbers])
    
    # Simulate what happens after LLM returns tool_calls
    # We manually create an AIMessage with tool_calls for demonstration
    fake_ai_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "add_numbers", "args": {"a": 10, "b": 20}, "id": "call_123"},
            {"name": "multiply_numbers", "args": {"a": 5, "b": 6}, "id": "call_456"},
        ]
    )
    
    # ToolNode takes state with messages, executes ALL tool calls
    # LangGraph 1.x injects ToolNode runtime context inside a compiled graph.
    demo_graph = StateGraph(SimpleState)
    demo_graph.add_node("tools", tool_node)
    demo_graph.add_edge(START, "tools")
    demo_graph.add_edge("tools", END)
    demo_app = demo_graph.compile()
    result = demo_app.invoke({"messages": [fake_ai_message]})
    
    print("\nToolNode executed both tools automatically:")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  Tool Call ID: {msg.tool_call_id}")
            print(f"  Result: {msg.content}")
            print()


# =============================================================================
# LESSON 5: Complete Graph with Tool Calling Loop
# =============================================================================
# Now we put it all together in a LangGraph graph.
# The pattern is ALWAYS:
#
#   agent_node ─── has tool calls? ──→ tool_node ──→ agent_node (loop)
#       │
#       └── no tool calls ──→ END
#
# This loop continues until the LLM decides it's done (no more tool calls).
# =============================================================================


def lesson_5_complete_graph():
    """Build a complete tool-calling graph."""
    print("\n" + "=" * 50)
    print("LESSON 5: Complete Tool-Calling Graph")
    print("=" * 50)
    
    tools = [add_numbers, multiply_numbers]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)
    
    # The agent node — calls the LLM
    def agent(state: SimpleState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    # Router — check if we need to execute tools
    def should_continue(state: SimpleState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"
    
    # Build the graph
    graph = StateGraph(SimpleState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END
    })
    graph.add_edge("tools", "agent")  # After tools, go back to agent
    
    app = graph.compile()
    
    # Run it!
    print("\n--- Running: 'What is (3 + 4) * 5?' ---")
    print("(This requires TWO tool calls: add then multiply)\n")
    
    result = app.invoke({
        "messages": [HumanMessage(content="What is (3 + 4) * 5? Use the tools step by step.")]
    })
    
    # Print the full message history to see the loop
    print("\n--- Full Message History ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  [{i}] {msg_type}: calls {[tc['name'] for tc in msg.tool_calls]}")
        elif isinstance(msg, ToolMessage):
            print(f"  [{i}] {msg_type}: result = {msg.content}")
        else:
            content = msg.content[:100] if msg.content else "(empty)"
            print(f"  [{i}] {msg_type}: {content}")


# =============================================================================
# RUN ALL LESSONS
# =============================================================================

if __name__ == "__main__":
    lesson_1_tool_schema()
    lesson_2_bind_tools()
    lesson_3_manual_tool_execution()
    lesson_4_tool_node()
    lesson_5_complete_graph()
    
    print("\n\n" + "=" * 70)
    print("ALL 5 LESSONS COMPLETED SUCCESSFULLY")
    print("=" * 70)
