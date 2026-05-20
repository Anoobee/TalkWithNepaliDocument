"""
frontend/app.py
---------------
Streamlit UI for the Nepali Document RAG system.
Talks to the FastAPI backend at BACKEND_URL.
"""

import os
import json
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Nepali Document Search",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #ffffff !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Base containers ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
.stApp > header,
.main,
section[data-testid="stMainBlockContainer"],
div[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
}

div { background-color: #ffffff !important; }

/* ── Typography globals ── */
p, span, label, div, h1, h2, h3, h4 {
    color: #111827 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #fafafa !important;
    border-right: 1px solid #e5e7eb !important;
    padding-top: 1rem;
}

section[data-testid="stSidebar"] .stSubheader,
section[data-testid="stSidebar"] [data-testid="stSubheader"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
    margin-bottom: 1.25rem !important;
}

/* Sidebar number inputs and checkboxes */
section[data-testid="stSidebar"] label {
    font-size: 0.875rem !important;
    color: #374151 !important;
    font-weight: 500 !important;
}

/* Number input: full container */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
    overflow: hidden !important;
}

/* The actual text input field */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    color: #111827 !important;
    background-color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}

/* +/- step buttons */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background-color: #f3f4f6 !important;
    color: #374151 !important;
    border: none !important;
    border-left: 1px solid #e5e7eb !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background-color: #e8f0fe !important;
    color: #1a73e8 !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
    fill: currentColor !important;
}

/* Checkbox */
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    font-size: 0.875rem !important;
    color: #374151 !important;
    font-weight: 400 !important;
}

/* ── Chat message container ── */
[data-testid="stChatMessageContainer"] {
    padding-bottom: 10rem !important;
    background: #ffffff !important;
}

/* ── Chat message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* ── Response text ── */
.chat-response-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9875rem;
    line-height: 1.75;
    color: #111827 !important;
    white-space: pre-wrap;
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
}

/* ── Metrics caption ── */
.stCaption {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #9ca3af !important;
    margin-top: 0.25rem !important;
    letter-spacing: 0.02em;
}

/* ── Chat input bar ── */
footer { visibility: hidden; }

[data-testid="stChatInputContainer"] {
    background-color: #ffffff !important;
    border-top: 1px solid #f3f4f6 !important;
    padding-top: 0.75rem !important;
}
div[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
}

/* Outer wrapper: relative so button can sit outside the box */
div[data-testid="stChatInput"] > div {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    background-color: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 0 0.2rem 0 0 !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
div[data-testid="stChatInput"] > div:focus-within {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.08) !important;
    background-color: #ffffff !important;
}

/* Textarea: borderless, fills the box */
div[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #111827 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.875rem 1.25rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9375rem !important;
    box-shadow: none !important;
    outline: none !important;
    flex: 1 !important;
    width: 100% !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    box-shadow: none !important;
    background-color: transparent !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
}

/* Send button: floated outside the border to the right */
div[data-testid="stChatInput"] button {
    position: absolute !important;
    right: -3rem !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    background: #000000 !important;
    border: none !important;
    box-shadow: 0 1px 4px rgba(26, 115, 232, 0.25) !important;
    padding: 0.45rem 0.55rem !important;
    margin: 0 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, box-shadow 0.15s ease !important;
    flex-shrink: 0 !important;
}
div[data-testid="stChatInput"] button:hover {
    background-color: #1558b0 !important;
    box-shadow: 0 2px 8px rgba(26, 115, 232, 0.35) !important;
}
div[data-testid="stChatInput"] button svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* ── Source cards ── */
.source-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1rem 1.125rem;
    margin-bottom: 0.625rem;
    border: 1px solid #e5e7eb;
    transition: border-color 0.15s ease;
}
.source-card:hover {
    border-color: #c7d7f8;
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.625rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #f3f4f6;
}
.source-doc-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #374151 !important;
    letter-spacing: 0.01em;
}
.source-score {
    background: #eff6ff;
    color: #1a73e8 !important;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid #dbeafe;
}
.source-text {
    color: #4b5563 !important;
    font-size: 0.875rem;
    line-height: 1.65;
    font-family: 'DM Sans', sans-serif;
}

/* ── Expander ── */
.stExpander {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    background: #ffffff !important;
    overflow: hidden !important;
    margin-top: 0.625rem !important;
}
.stExpander > details > summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8375rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    padding: 0.625rem 0.875rem !important;
    background: #fafafa !important;
    border-bottom: 1px solid #f3f4f6 !important;
}
.stExpander > details[open] > summary {
    color: #1a73e8 !important;
}
.stExpander > details > div {
    padding: 0.875rem !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #f3f4f6 !important;
    margin: 0.75rem 0 !important;
}

/* ── Error alerts ── */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid #fecaca !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Configuration")
    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)

    top_k_retrieval = st.number_input(
        "Documents to retrieve",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of candidate documents fetched from the vector store.",
        key="config_top_k_retrieval",
    )
    top_k_context = st.number_input(
        "Documents to use in context",
        min_value=1,
        max_value=10,
        value=5,
        help="Top-ranked documents passed to the language model.",
        key="config_top_k_context",
    )
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    use_streaming = st.checkbox("Stream response", value=True, key="config_streaming")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#9ca3af !important;'>Nepali Document RAG by Anup Aryal</p>",
        unsafe_allow_html=True,
    )

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 0.5rem;'>
        <span style='
            display: inline-block;
            background: #eff6ff;
            border: 1px solid #dbeafe;
            color: #1a73e8 !important;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            padding: 4px 14px;
            border-radius: 20px;
            font-family: "DM Sans", sans-serif;
        '>Retrieval-Augmented Generation</span>
    </div>
    <h1 style='
        text-align: center;
        color: #111827 !important;
        font-family: "DM Sans", sans-serif;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.035em;
        line-height: 1.2;
        margin-bottom: 0.625rem;
    '>Nepali Document Search</h1>
    <p style='
        text-align: center;
        color: #6b7280 !important;
        font-family: "DM Sans", sans-serif;
        font-size: 0.9375rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        line-height: 1.6;
    '>Ask questions about your documents in Nepali or English</p>
    """,
    unsafe_allow_html=True,
)

# ── Conversation History ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(
                f'<div class="chat-response-text">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            if msg.get("metrics"):
                m = msg["metrics"]
                st.caption(
                    f"⏱ Retrieval {m['retrieval']} ms · Generation {m['generation']} ms · Total {m['total']} ms"
                )

            if msg.get("sources"):
                with st.expander(
                    f"📄 {len(msg['sources'])} source(s) referenced", expanded=False
                ):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"""
                            <div class="source-card">
                              <div class="source-header">
                                <span class="source-doc-label">Document {i + 1} &nbsp;·&nbsp; {src['id']}</span>
                                <span class="source-score">score {src['relevance']:.4f}</span>
                              </div>
                              <div class="source-text">{src['text']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    payload = {
        "query": prompt.strip(),
        "top_k_retrieval": st.session_state.config_top_k_retrieval,
        "top_k_context": st.session_state.config_top_k_context,
    }

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        metrics_placeholder = st.empty()
        sources_placeholder = st.empty()

        full_answer = ""
        sources = []
        metrics_data = None

        if st.session_state.config_streaming:
            # ── Streaming mode ──
            try:
                with requests.post(
                    f"{BACKEND_URL}/query/stream",
                    json=payload,
                    stream=True,
                    timeout=120,
                ) as resp:
                    if resp.status_code != 200:
                        st.error(f"Backend error {resp.status_code}: {resp.text}")
                    else:
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            text = line.decode("utf-8")
                            if not text.startswith("data: "):
                                continue
                            content = text[6:]

                            if content == "[DONE]":
                                break
                            elif content.startswith("[ERROR]"):
                                st.error(f"Error: {content[7:]}")
                                break
                            elif content.startswith("[SOURCES]"):
                                try:
                                    sources = json.loads(content[9:])
                                except ValueError:
                                    pass
                            else:
                                decoded_content = content.replace("\\n", "\n")
                                full_answer += decoded_content
                                answer_placeholder.markdown(
                                    f'<div class="chat-response-text">{full_answer}▌</div>',
                                    unsafe_allow_html=True,
                                )

                        answer_placeholder.markdown(
                            f'<div class="chat-response-text">{full_answer}</div>',
                            unsafe_allow_html=True,
                        )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Unable to reach the backend service. Please check that it is running."
                )

        else:
            # ── Non-streaming mode ──
            with st.spinner("Searching and generating response…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query", json=payload, timeout=120
                    )
                    if resp.status_code != 200:
                        st.error(
                            f"Backend error {resp.status_code}: {resp.json().get('detail', resp.text)}"
                        )
                    else:
                        data = resp.json()
                        full_answer = data["answer"]
                        sources = data.get("sources", [])
                        metrics_data = {
                            "retrieval": data["retrieval_time_ms"],
                            "generation": data["generation_time_ms"],
                            "total": data["total_time_ms"],
                        }
                        answer_placeholder.markdown(
                            f'<div class="chat-response-text">{full_answer}</div>',
                            unsafe_allow_html=True,
                        )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Unable to reach the backend service. Please check that it is running."
                    )

        if metrics_data:
            metrics_placeholder.caption(
                f"⏱ Retrieval {metrics_data['retrieval']} ms · Generation {metrics_data['generation']} ms · Total {metrics_data['total']} ms"
            )

        if sources:
            with sources_placeholder.expander(
                f"📄 {len(sources)} source(s) referenced", expanded=False
            ):
                for i, src in enumerate(sources):
                    st.markdown(
                        f"""
                        <div class="source-card">
                          <div class="source-header">
                            <span class="source-doc-label">Document {i + 1} &nbsp;·&nbsp; {src['id']}</span>
                            <span class="source-score">score {src['relevance']:.4f}</span>
                          </div>
                          <div class="source-text">{src['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
            "metrics": metrics_data,
        }
    )
