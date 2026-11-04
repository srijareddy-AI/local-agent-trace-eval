"""Trace evaluation for the baseline agent.

Metrics (all computed from the traces, never from the agent's claims):
- tool_choice_accuracy: trace tool sequence == expected sequence.
- task_success_rate: final answer contains the expected answer
  (for the date task, today's UTC date, computed at eval time).
- hallucination_rate: a case is flagged when the final answer contains a
  number that appears neither in any tool output nor in the question,
  OR when the answer claims success while every tool call errored.
  The lookup-not-found task is the control: inventing a price there
  would count as a hallucination; correctly saying "not found" does not.
- failure_recovery_rate: among tasks whose first tool call errored,
  the share that still ended in a correct final answer.
- mean_steps: mean number of trace steps per task.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run  # noqa: E402
from src.trace import write_traces  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _hallucinated(trace, question):
    allowed = _numbers(question)
    for step in trace["steps"]:
        allowed |= _numbers(json.dumps(step["tool_output"]))
    # Dates are allowed: the date tool legitimately introduces today's date.
    if any(s["tool"] == "get_date_time" for s in trace["steps"]):
        allowed |= _numbers(trace["final_answer"])
    invented = _numbers(trace["final_answer"]) - allowed
    if invented:
        return True
    if trace["steps"] and all(
        "error" in (s["tool_output"] or {}) for s in trace["steps"]
    ):
        # Every tool failed, yet the agent still produced an answer that
        # is not an explicit failure statement.
        low = trace["final_answer"].lower()
        if "could not" not in low and "failed" not in low and "no " not in low:
            return True
    return False


def main():
    with open(os.path.join(BASE_DIR, "evals", "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    traces, rows = [], []
    for task in tasks:
        trace = run(task["question"])
        expected = task["expected_answer"] or today
        success = expected.lower() in trace["final_answer"].lower()
        tool_seq = [s["tool"] for s in trace["steps"]]
        first_errored = bool(trace["steps"]) and "error" in (
            trace["steps"][0]["tool_output"] or {}
        )
        rows.append(
            {
                "id": task["id"],
                "category": task["category"],
                "tool_choice_correct": tool_seq == task["expected_tool_sequence"],
                "tool_sequence": tool_seq,
                "expected_tool_sequence": task["expected_tool_sequence"],
                "task_success": success,
                "hallucinated": _hallucinated(trace, task["question"]),
                "first_tool_errored": first_errored,
                "recovered": bool(trace["recovered_from_error"]),
                "n_steps": trace["n_steps"],
                "final_answer": trace["final_answer"],
            }
        )
        traces.append(trace)
    n = len(rows)
    errored = [r for r in rows if r["first_tool_errored"]]
    results = {
        "agent": "rule-based baseline (deterministic, clearly labelled)",
        "n_tasks": n,
        "tool_choice_accuracy": round(
            sum(r["tool_choice_correct"] for r in rows) / n, 3
        ),
        "task_success_rate": round(sum(r["task_success"] for r in rows) / n, 3),
        "hallucination_rate": round(sum(r["hallucinated"] for r in rows) / n, 3),
        "failure_recovery_rate": (
            round(
                sum(r["task_success"] for r in errored) / len(errored), 3
            )
            if errored
            else None
        ),
        "n_tasks_with_first_tool_error": len(errored),
        "mean_steps": round(sum(r["n_steps"] for r in rows) / n, 2),
        "rows": rows,
    }
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    write_traces(traces, os.path.join(BASE_DIR, "results", "traces.jsonl"))
    with open(
        os.path.join(BASE_DIR, "results", "eval_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "rows"}, indent=2))
    for row in rows:
        print(
            f"{row['id']:>24} success={row['task_success']} "
            f"tool_ok={row['tool_choice_correct']} hallucinated={row['hallucinated']} "
            f"steps={row['n_steps']} recovered={row['recovered']}"
        )


if __name__ == "__main__":
    main()
