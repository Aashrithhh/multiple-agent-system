"""Quick test: Run the full coding agent on a simple task."""
import warnings
warnings.filterwarnings("ignore")

from coding_agent_v6 import build_coding_agent_graph
from langchain_core.messages import HumanMessage, ToolMessage

graph = build_coding_agent_graph()

task = (
    "Write a Python function called 'fibonacci(n)' that returns the nth fibonacci number. "
    "Test it with python_exec for n=10 (should return 55)."
)

print("=" * 60)
print("RUNNING FULL AGENT ON FIBONACCI TASK")
print("=" * 60)

result = graph.invoke(
    {
        "messages": [HumanMessage(content=task)],
        "current_phase": "planning",
        "plan": "",
        "code_output": "",
        "review_feedback": "",
        "iteration": 0,
    },
    {"recursion_limit": 30}
)

print("\n" + "=" * 60)
print("AGENT COMPLETED")
print("=" * 60)
print(f"Total messages exchanged: {len(result['messages'])}")
print(f"Final phase: {result.get('current_phase', 'unknown')}")
print(f"Iterations: {result.get('iteration', 0)}")

# Show tool usage summary
tool_calls_made = []
for msg in result["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls_made.append(tc["name"])

print(f"\nTools used: {tool_calls_made}")

# Print tool results
print("\n--- Tool Execution Results ---")
for msg in result["messages"]:
    if isinstance(msg, ToolMessage):
        print(f"  Result: {msg.content[:200]}")

# Print final review
review = result.get("review_feedback", "")
if review:
    review_text = review if isinstance(review, str) else str(review)
    print(f"\n--- Review Verdict ---")
    print(review_text[:500])
