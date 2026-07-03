"""Streamlit UI for the Email Intelligence Assistant."""

import requests
import streamlit as st

API_URL = "http://localhost:8000"


def check_backend() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        return False
    else:
        return response.status_code == 200


def ask(query: str, chat_history: list[dict[str, str]]) -> tuple[str, list[dict]]:
    """Send a query and chat history to the backend, return (answer, sources)."""
    response = requests.post(
        f"{API_URL}/ask",
        json={"query": query, "chat_history": chat_history},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["answer"], data["sources"]



# UI
st.set_page_config(
    page_title="Email Intelligence Assistant",
    page_icon="✉️",
    layout="centered",
)

st.title("✉️ Email Intelligence Assistant")
st.caption("Ask questions about SMB Freight FZE aviation operations emails.")

if not check_backend():
    st.error(
        "Backend is not running. Start it with: "
        "`uvicorn my_rag_app.api.endpoint:app --reload`"
    )
    st.stop()

# Session state — messages store role/content for display and history
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# Replay existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📧 Sources ({len(message['sources'])} email(s))"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**[{i}]** {src.get('subject', '—')}  \n"
                        f"From: `{src.get('sender', '—')}` · {src.get('date', '—')}  \n"
                        f"_{src.get('snippet', '')}_"
                    )
                    if i < len(message["sources"]):
                        st.divider()

if query := st.chat_input("Ask a question about your emails..."):
    # Build history as plain role/content dicts for the API
    # (exclude the 'sources' key — API only needs text)
    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching emails..."):
            try:
                answer, sources = ask(query, api_history)
            except requests.exceptions.HTTPError as e:
                answer = f"Error: {e.response.json().get('detail', str(e))}"
                sources = []
            except Exception as e:
                answer = f"Error: {e}"
                sources = []

        st.write(answer)

        if sources:
            with st.expander(f"📧 Sources ({len(sources)} email(s))"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(
                        f"**[{i}]** {src.get('subject', '—')}  \n"
                        f"From: `{src.get('sender', '—')}` · {src.get('date', '—')}  \n"
                        f"_{src.get('snippet', '')}_"
                    )
                    if i < len(sources):
                        st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
