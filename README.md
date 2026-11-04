# Local Agentic Tool-Use Agent with Trace Eval

> **Demo/research project only. All tools are read-only; the agent cannot
> write files, send messages, change state, or spend money. There are no
> consequential actions in this system, by design.**

## The problem

Most student agent demos show a happy-path chat. A hiring manager cannot
tell from that whether the agent actually chose the right tool, recovered
from a tool failure, or quietly hallucinated a result it never received.
This project proves those three things with logged traces and measured
evaluation, instead of a chat transcript.

## What it does

A local agent answers questions by calling real local tools only:

| Tool | What it does |
| --- | --- |
| `search_documents` | Keyword search over local markdown docs (company policies) |
| `calculator` | Safe AST-based arithmetic — no `eval()`, no names, no calls |
| `lookup_record` | Lookup in a local structured CSV (products, locations, benefits) |
| `get_date_time` | Current UTC date/time |

Every run produces a machine-readable trace: thought, tool chosen, tool
input, tool output, next step, final answer (`results/traces.jsonl`).
The evaluator (`src/evaluate.py`) scores those traces on a fixed test
set of 12 tasks — it never trusts the agent's own claims.

## Results (measured, not claimed)

**Headline result — deterministic rule-based baseline, clearly labelled
as such** (`python3 -m src.evaluate`, standard library only):

| Metric | Result |
| --- | --- |
| Tool-choice accuracy | **1.000** (12/12 sequences matched) |
| Task success rate | **1.000** (12/12) |
| Hallucination rate | **0.000** (0/12) |
| Failure-recovery rate | **1.000** (3/3 tasks whose first tool attempt errored still ended correct) |
| Mean steps per task | 1.33 |

The 3 first-attempt errors were: two calculator retries (numbers spelled
out in words, and a malformed `12 * * 5 + 10`) and one document search
with no match, which the agent correctly surfaced as "No local document
matched" instead of inventing a price for a product that does not exist.
Tests: 8/8 passed (`python3 -m unittest discover -s tests`).

Two bugs were found and fixed during this build, not hidden: word-number
arithmetic was initially answered from a stray digit alone, and document
search initially matched on stopwords alone ("the", "what"). Both fixes
are in `src/agent.py` and `src/tools.py`, and both are covered by the
test set.

### The Ollama path — tested live, and honestly unreliable

`src/ollama_agent.py` asks a local `llama3.2:1b` (temperature 0) to pick
tools as JSON. Tested live on 2026-11-04 against the local Ollama server:

- **Task success: 0/2.** On "What is 15 * 4 + 20?" the model called the
  calculator correctly twice (result 80), then kept calling tools
  instead of stopping, then emitted unparseable JSON with an extra
  closing brace. On the lookup question it emitted unparseable JSON on
  the first call.
- **The safety property held:** both runs ended in an explicit parse
  failure, not a hallucinated answer.
- **Conclusion:** the 1B model's JSON tool-choice output is unreliable
  for this format. The headline result above is therefore the baseline,
  not an LLM result. A structured-output/tool-calling format or a larger
  model would be needed before claiming an LLM-agent number.
  Full detail: `results/ollama_results.json`.

## Architecture

```
question
   │
   ▼
rule-based planner (src/agent.py)      ── or ──  Ollama planner (src/ollama_agent.py)
   │  multi-step → lookup, lookup, calculate
   │  date → get_date_time · key → lookup_record
   │  arithmetic → calculator (retry on normalised expression)
   │  otherwise → search_documents
   ▼
read-only tools (src/tools.py)  →  trace (src/trace.py)
   ▼
final answer built ONLY from tool outputs; errors surface as failures
   ▼
evaluator (src/evaluate.py) scores traces: tool choice, success,
hallucination, recovery, step count
```

## Run it

```bash
# Evaluate the baseline (no dependencies)
python3 -m src.evaluate

# Ask one question from the CLI, with its full trace
python3 main.py "What is the total cost of the aurora laptop and the orbit headphones combined?"

# Tests
python3 -m unittest discover -s tests

# Streamlit demo (optional)
pip install -r requirements.txt
streamlit run app.py

# Optional Ollama path (needs a running local Ollama server with llama3.2:1b)
OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODEL=llama3.2:1b python3 -c \
  "from src.ollama_agent import run_ollama; print(run_ollama('What is 15 * 4 + 20?'))"
```

## Skills this demonstrates → roles

| Skill | Where to see it | Role fit |
| --- | --- | --- |
| Agentic AI / tool use | `src/agent.py`, `src/tools.py` | AI/ML Engineer |
| Evaluation design | `src/evaluate.py`, `evals/tasks.json` | AI/ML Engineer, Data Scientist |
| Failure recovery & safety | Recovery retries, read-only tools, no-hallucination control task | AI/ML Engineer |
| Local LLM work | `src/ollama_agent.py`, honest Ollama finding above | AI/ML Engineer, GenAI roles |
| Python, reproducible testing | `tests/`, fixed test set, trace files | AI/ML Engineer, Data Scientist |

## Limitations (read before reusing)

- The baseline is deterministic and rule-based — it is a measurable
  floor, not a general-purpose LLM agent. It handles the task families
  in `evals/tasks.json`; it will not generalise to arbitrary questions.
- The test set is small (12 tasks) and the documents/records are
  synthetic company data written for this demo. Scores prove the
  pipeline, traces, and gates work — not real-world agent performance.
- The Ollama path, as tested, is unreliable for JSON tool choice
  (see above). Do not cite an LLM-agent success rate from this repo.
- No web, database, or paid APIs are used anywhere. That is deliberate:
  the point is traceability, not breadth of tools.

## Project structure

```
src/agent.py          Rule-based baseline planner
src/tools.py          Read-only tools (search, calculator, lookup, date)
src/trace.py          Trace logging
src/evaluate.py       Trace evaluation and metrics
src/ollama_agent.py   Optional local Ollama planner (tested, unreliable)
main.py               CLI demo  ·  app.py  Streamlit demo
data/docs/            4 synthetic policy documents
data/records.csv      7 synthetic structured records
evals/tasks.json      12 fixed tasks with expected tools and answers
results/              eval_results.json, traces.jsonl, ollama_results.json
tests/test_agent.py   8 tests (tools, multi-step, recovery, no-hallucination)
```
