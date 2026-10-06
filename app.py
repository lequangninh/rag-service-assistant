"""Web demo: uv run streamlit run app.py"""

from __future__ import annotations

import anthropic
import streamlit as st

from rag.answer import MissingAPIKeyError
from rag.pipeline import Assistant

EXAMPLES = [
    "How much is a deep clean for a 3-bedroom home?",
    "What happens if I cancel the day before?",
    "Can I pay with cash?",
    "Do you clean swimming pools?",
]

st.set_page_config(page_title="BrightClean Assistant", page_icon="🧽", layout="centered")


@st.cache_resource(show_spinner="Loading documents and building the search index…")
def load_assistant() -> Assistant:
    return Assistant()


assistant = load_assistant()

st.title("BrightClean Assistant")
st.caption(
    "Answers questions **only** from BrightClean's own documents, cites the exact passage for every claim, "
    "and says *I don't know* instead of guessing. BrightClean is a fictional sample company."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Chunking:** each policy document is split into its sections.\n"
        "2. **Hybrid retrieval:** keyword search (BM25) and meaning-based search "
        "(vector embeddings) each rank the sections; the rankings are merged.\n"
        "3. **Relevance gate:** if nothing is close enough, the assistant says it doesn't know "
        "without calling the model.\n"
        "4. **Grounded answer:** Claude answers from the top sections only, with "
        "API-verified citations."
    )
    st.header("Documents")
    for source in sorted({c.source for c in assistant.retriever.chunks}):
        st.markdown(f"- `{source}`")

if "question" not in st.session_state:
    st.session_state.question = ""

cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_input("Your question", key="question", placeholder="e.g. Do you bring your own supplies?")

if question:
    try:
        with st.spinner("Searching the documents…"):
            answer, hits = assistant.ask(question)
    except (MissingAPIKeyError, anthropic.AuthenticationError):
        st.error("The API key in `.env` is missing or invalid. Set `ANTHROPIC_API_KEY` and restart the app.")
        st.stop()
    except anthropic.APIConnectionError:
        st.error("Couldn't reach the Claude API. Check your internet connection and try again.")
        st.stop()
    except anthropic.APIStatusError as e:
        st.error(f"The Claude API returned an error ({e.status_code}). Please try again.")
        st.stop()

    if answer.answered:
        st.success(answer.text)
    else:
        st.warning(answer.text)

    if answer.citations:
        st.subheader("Sources")
        for c in answer.citations:
            st.markdown(f"**[{c.number}]** `{c.label}`")
            st.markdown(f"> {c.quote}")

    with st.expander("Retrieved sections and scores"):
        for h in hits:
            st.markdown(f"`{h.similarity:.2f}` similarity · `{h.chunk.label}`")
        if answer.skipped_model_call:
            st.caption("No section was relevant enough, so the model wasn't called.")
        elif answer.model:
            st.caption(f"Answered by `{answer.model}`.")
