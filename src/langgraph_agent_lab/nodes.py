"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


class ClassificationResult(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(...)
    rationale: str = Field(default="")


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ClassificationResult)
    result = structured_llm.invoke(
        (
            "Classify this support request into exactly one route.\n"
            "Routes: simple, tool, missing_info, risky, error.\n"
            "Priority order when multiple seem possible: risky > tool > missing_info > error > simple.\n"
            "Definitions:\n"
            "- risky: side effects like refunds, deletions, emails, cancellations, account changes\n"
            "- tool: information lookup like order status, tracking, search, retrieval\n"
            "- missing_info: request is too vague or lacks key details\n"
            "- error: system failure, crash, timeout, unavailable service\n"
            "- simple: general question answerable directly\n\n"
            f"User query: {query}"
        )
    )
    route = result.route
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "classified",
                "query classified",
                route=route,
                rationale=result.rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    query = state.get("query", "").strip()
    attempt = state.get("attempt", 0)

    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure while handling '{query}'"
    elif route == "tool":
        result = f"Tool lookup success: found status for request '{query}'"
    elif route == "risky":
        result = f"Tool execution success: prepared or simulated action for '{query}'"
    else:
        result = f"Tool success: completed processing for '{query}'"

    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                "completed" if "ERROR" not in result else "failed",
                "mock tool executed",
                attempt=attempt,
                result=result,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest_result = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest_result else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                evaluation_result=evaluation_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    proposed_action = state.get("proposed_action")
    llm = get_llm(temperature=0.2)
    final_answer = llm.invoke(
        (
            "You are a support assistant. Write a concise helpful answer grounded only in the context below.\n"
            "If tool results exist, use them. If approval exists, mention approval status when relevant.\n"
            "Do not invent facts beyond the provided context.\n\n"
            f"User query: {query}\n"
            f"Tool results: {tool_results}\n"
            f"Proposed action: {proposed_action}\n"
            f"Approval: {approval}\n"
        )
    ).content
    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "answer",
                "completed",
                "final answer generated",
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    pending_question = (
        f"Could you clarify your request? I need more detail to help with: '{query}'."
    )
    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "events": [
            make_event(
                "clarify",
                "clarification_requested",
                "asked user for more information",
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    proposed_action = (
        f"Proposed risky action based on user request: '{query}'. "
        "This may cause side effects and requires human approval."
    )
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "approval_required",
                "prepared risky action for approval",
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    approval = {
        "approved": True,
        "reviewer": "mock-reviewer",
        "comment": "Auto-approved for local testing",
    }
    return {
        "approval": approval,
        "events": [
            make_event(
                "approval",
                "approved",
                "mock approval granted",
                reviewer=approval["reviewer"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    next_attempt = state.get("attempt", 0) + 1
    return {
        "attempt": next_attempt,
        "errors": [f"retry attempt {next_attempt} after transient failure"],
        "events": [
            make_event(
                "retry",
                "retry_scheduled",
                "transient failure recorded",
                attempt=next_attempt,
                max_attempts=state.get("max_attempts", 3),
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    return {
        "final_answer": "I could not complete this request after multiple attempts. Please escalate or try again later.",
        "events": [
            make_event(
                "dead_letter",
                "failed",
                "maximum retry attempts exceeded",
                attempt=state.get("attempt", 0),
                max_attempts=state.get("max_attempts", 3),
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
