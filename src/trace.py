"""Trace logging: one machine-readable record per agent run.

A trace is the evidence that the agent actually used its tools. If a value
appears in the final answer but never appeared in a tool output (or in the
question itself), the evaluator flags it as a hallucination.
"""
import json
import time


class Trace:
    def __init__(self, question):
        self.question = question
        self.steps = []
        self.started = time.time()

    def add(self, thought, tool, tool_input, tool_output):
        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "thought": thought,
                "tool": tool,
                "tool_input": tool_input,
                "tool_output": tool_output,
            }
        )

    def finish(self, final_answer, recovered=False):
        return {
            "question": self.question,
            "final_answer": final_answer,
            "steps": self.steps,
            "n_steps": len(self.steps),
            "recovered_from_error": recovered,
            "elapsed_seconds": round(time.time() - self.started, 3),
        }


def write_traces(traces, path):
    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")
