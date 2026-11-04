"""CLI demo: ask the baseline agent a question and print its trace."""
import json
import sys

from src.agent import run

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the total cost of the aurora laptop and the orbit headphones combined?"
    trace = run(question)
    print(json.dumps(trace, indent=2))
