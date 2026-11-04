"""Optional Ollama agent (NOT the measured baseline).

Asks a local Ollama model (default llama3.2:1b) to pick a tool as JSON,
executes the tool, and returns the result to the model for a final answer.
Tool outputs are the only source of facts: if the model returns a tool
name that does not exist, or unparseable JSON, that step is logged as an
error and the run ends with an explicit failure statement, not an
invented answer.

This path is evaluated separately from the baseline and only when it is
actually run against a live Ollama server. If the server is unavailable,
this module reports that fact; it never fakes an Ollama result.
"""
import json
import os
import urllib.request

from .tools import TOOLS
from .trace import Trace

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11436")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

SYSTEM = (
    "You are a tool-using agent. Reply with ONLY a JSON object: "
    '{"tool": "<tool name>", "input": "<tool input as a string>"} '
    "or {\"tool\": \"final\", \"input\": \"<final answer>\"}. "
    "Tools: search_documents, calculator, lookup_record, get_date_time. "
    "Use a tool before answering any factual question. Never invent a "
    "tool result; if a tool errors, say so."
)


def _chat(messages):
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"]


def run_ollama(question, max_steps=4):
    trace = Trace(question)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        for _ in range(max_steps):
            raw = _chat(messages)
            try:
                parsed = json.loads(raw.strip().strip("`"))
            except Exception:  # noqa: BLE001 - logged as a trace error
                trace.add("Model output was not valid JSON.", "llm", raw, {"error": "unparseable"})
                return trace.finish(
                    "The model response could not be parsed, so I do not have an answer."
                )
            tool, tool_input = parsed.get("tool"), parsed.get("input", "")
            if tool == "final":
                return trace.finish(str(tool_input))
            if tool not in TOOLS:
                trace.add("Model chose an unknown tool.", "llm", raw, {"error": f"unknown tool {tool!r}"})
                return trace.finish(
                    "The model chose a tool that does not exist, so I do not have an answer."
                )
            output = TOOLS[tool](tool_input) if tool_input or tool == "get_date_time" else TOOLS[tool]()
            trace.add(f"Execute {tool}.", tool, tool_input, output)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Tool {tool} returned: {json.dumps(output)}"}
            )
        return trace.finish("The agent ran out of steps without a final answer.")
    except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
        return trace.finish(f"Ollama path failed: {exc}")
