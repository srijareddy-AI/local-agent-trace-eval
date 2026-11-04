"""Streamlit demo: ask a question, see the answer and the full trace."""
import streamlit as st

from src.agent import run

st.set_page_config(page_title="Local Agentic Tool-Use Agent")
st.title("Local Agentic Tool-Use Agent with Trace Eval")
st.write(
    "Read-only local tools only: document search, calculator, structured "
    "lookup, and date/time. Every step is logged in the trace below. "
    "Try: 'What is the total cost of the aurora laptop and the orbit "
    "headphones combined?'"
)
question = st.text_input(
    "Question",
    "What is the total cost of the aurora laptop and the orbit headphones combined?",
)
if st.button("Ask the agent"):
    trace = run(question)
    st.subheader("Answer")
    st.write(trace["final_answer"])
    st.subheader("Trace")
    for step in trace["steps"]:
        with st.expander(f"Step {step['step']}: {step['tool']}"):
            st.write("Thought:", step["thought"])
            st.write("Input:", step["tool_input"])
            st.write("Output:", step["tool_output"])
