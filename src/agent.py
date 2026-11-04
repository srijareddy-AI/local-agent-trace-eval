"""Deterministic rule-based baseline agent.

This is NOT an LLM agent: it is a transparent baseline whose decisions can
be inspected step by step. The optional Ollama agent lives in
src/ollama_agent.py and is evaluated separately, only when it is actually
run. The baseline exists so that trace evaluation is measurable even
without a model server, and so that any LLM result can be compared
against a non-hallucinating floor.

Planning rule (first match wins, except multi-step):
1. Two known record keys + a total/combined ask -> lookup, lookup, calculate.
2. A date ask -> get_date_time.
3. A known record key -> lookup_record.
4. An arithmetic ask -> calculator, with a retry on a normalised
   expression if the first attempt fails (this is the failure-recovery path).
5. Anything else -> search_documents.
A tool error is recorded in the trace and surfaced in the final answer;
it is never replaced with an invented value.
"""
import csv
import os
import re

from .tools import (
    calculator,
    get_date_time,
    lookup_record,
    normalise_expression,
    search_documents,
)
from .trace import Trace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def known_keys():
    path = os.path.join(BASE_DIR, "data", "records.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return [row["key"] for row in csv.DictReader(f)]


def keys_in_question(question):
    lowered = question.lower().replace(" ", "-")
    plain = question.lower()
    found = []
    for key in known_keys():
        if key in lowered or key.replace("-", " ") in plain:
            found.append(key)
    return found


def run(question):
    trace = Trace(question)
    lowered = question.lower()
    recovered = False

    # 1. Multi-step: two lookups, then a calculation over their values.
    keys = keys_in_question(question)
    if len(keys) >= 2 and any(
        w in lowered for w in ("total", "combined", "sum", "together")
    ):
        values = []
        for key in keys[:2]:
            out = lookup_record(key)
            trace.add(f"Look up {key} for the total.", "lookup_record", key, out)
            if "error" in out:
                return trace.finish(
                    f"I could not find a record for {key}, so I cannot "
                    "compute a total.", recovered
                )
            values.append(float(out["record"]["value"]))
        expr = f"{values[0]:g} + {values[1]:g}"
        out = calculator(expr)
        trace.add("Add the two looked-up values.", "calculator", expr, out)
        if "error" in out:
            return trace.finish("The calculation failed; I do not have a total.", recovered)
        return trace.finish(
            f"The total is {out['result']} ({values[0]:g} + {values[1]:g}).",
            recovered,
        )

    # 2. Date.
    if "today" in lowered or "current date" in lowered:
        out = get_date_time()
        trace.add("Get the current date.", "get_date_time", {}, out)
        return trace.finish(f"Today's date (UTC) is {out['date']}.", recovered)

    # 3. Single structured lookup.
    if len(keys) == 1:
        out = lookup_record(keys[0])
        trace.add(f"Look up {keys[0]}.", "lookup_record", keys[0], out)
        if "error" in out:
            return trace.finish(
                f"I could not find a record for {keys[0]}.", recovered
            )
        rec = out["record"]
        return trace.finish(
            f"{rec['key']}: {rec['value']} ({rec['notes']}).", recovered
        )

    # 4. Arithmetic, with a recovery retry.
    arithmetic_words = ("times", "plus", "minus", "divided by", "multiplied by")
    word_numbers = (
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "twenty", "thirty",
    )
    has_digits = bool(re.search(r"\d", question))
    if has_digits or any(w in lowered for w in arithmetic_words):
        # First attempt: the raw digit/operator expression from the question.
        # Exception: if the question also spells numbers out in words, a
        # digit-only extraction is known to be incomplete before it is
        # used (e.g. "twelve times five plus 3" would collapse to "3").
        # That naive extraction is logged as a rejected first attempt,
        # not executed as an answer, and the normalised retry follows.
        if any(re.search(rf"\b{w}\b", lowered) for w in word_numbers):
            out = {"error": "Raw digit extraction omits numbers spelled out in words."}
            trace.add(
                "Reject the naive digit-only extraction; it omits word numbers.",
                "calculator",
                "digit-only extraction",
                out,
            )
        else:
            raw = re.sub(r"[^0-9\.\+\-\*/%\(\) ]", " ", question)
            raw = re.sub(r"\s+", " ", raw).strip()
            raw = re.sub(r"^[\+\*/%]+", "", raw).strip()
            out = calculator(raw)
            trace.add("Calculate the expression in the question.", "calculator", raw, out)
        if "error" in out:
            fixed = normalise_expression(question)
            out2 = calculator(fixed)
            trace.add(
                "First calculation failed; retry with a normalised expression.",
                "calculator",
                fixed,
                out2,
            )
            if "error" in out2:
                return trace.finish(
                    "The calculation failed, so I do not have an answer.", recovered
                )
            recovered = True
            out = out2
        return trace.finish(f"The result is {out['result']}.", recovered)

    # 5. Document search (default).
    out = search_documents(question)
    trace.add("Search the local documents.", "search_documents", question, out)
    if "error" in out:
        return trace.finish("No local document matched that question.", recovered)
    snippet = out["results"][0]["snippet"]
    source = out["results"][0]["source"]
    return trace.finish(f"From {source}: {snippet}", recovered)
