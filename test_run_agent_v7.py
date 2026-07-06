"""Test V7 agent end-to-end with auto-approve mode."""
import warnings
warnings.filterwarnings("ignore")

from coding_agent_v7 import run_agent_auto_approve, _extract_text
from langchain_core.messages import ToolMessage

task = (
    "Write a Python function called 'is_palindrome(s)' that checks if a string "
    "is a palindrome (ignoring case and spaces). Test it with python_exec."
)

print("=" * 60)
print("V7 AGENT — AUTO-APPROVE MODE")
print("=" * 60)
print(f"Task: {task}\n")

result = run_agent_auto_approve(task)

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
print(f"Total messages: {len(result['messages'])}")
print(f"Iterations: {result.get('iteration', 0)}")

# Show tools used
tools_used = []
for msg in result["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tools_used.append(tc["name"])
print(f"Tools called: {tools_used}")

# Show tool results
print("\n--- Tool Results ---")
for msg in result["messages"]:
    if isinstance(msg, ToolMessage):
        content = msg.content[:150] if len(msg.content) > 150 else msg.content
        print(f"  {content}")

# Show review
review = _extract_text(result.get("review_feedback", ""))
if review:
    print(f"\nReview: {review[:300]}")
