"""
ai/llm.py
---------
Calls Gemini (gemini-2.5-flash) with the retrieved context and user query.
Supports both standard and streaming responses.
"""

from __future__ import annotations
import os
import google.generativeai as genai
from typing import Iterator

_gemini_configured = False


def _configure():
    global _gemini_configured
    if not _gemini_configured:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key! Please set either GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable in your .env file or system environment."
            )
        genai.configure(api_key=api_key)
        _gemini_configured = True


SYSTEM_PROMPT = """You are a helpful assistant for a Nepali company document search system.
You are given retrieved document passages relevant to the user's query.
Answer the query accurately and concisely using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Respond in the same language as the query (Nepali or English).
Always cite which document IDs you used in your answer."""


def build_prompt(query: str, context_docs: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Doc {i+1} | ID: {doc['id']} | Score: {doc['relevance']:.4f}]\n{doc['text']}"
        for i, doc in enumerate(context_docs)
    )
    return f"""Context Documents:
{context_block}

---
User Query: {query}

Answer:"""


def generate_answer(query: str, context_docs: list[dict]) -> str:
    """Blocking generation."""
    _configure()
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = build_prompt(query, context_docs)
    response = model.generate_content(prompt)
    return response.text


def stream_answer(query: str, context_docs: list[dict]) -> Iterator[str]:
    """Streaming generation — yields text chunks."""
    _configure()
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = build_prompt(query, context_docs)
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"[ERROR streaming response: {str(e)}]"
