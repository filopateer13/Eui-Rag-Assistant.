import os
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY")))

SYSTEM_PROMPT = (
    "You are the official AI assistant for Egypt Informatics University (EUI). "
    "Be helpful, accurate, and concise. You assist with IT support and general "
    "university inquiries."
)

# How many recent chat turns to feed back to the model so follow-up
# questions keep context. Kept deliberately small -- not an answer-length
# limit (max_tokens is untouched), just how much history rides along.
MAX_HISTORY_MESSAGES = 6  # ~3 user/assistant exchanges


def get_answer(question: str, history: list[dict] | None = None) -> str:
    """
    Send the question to the Groq model, including recent conversation
    history so follow-up questions keep context.

    Raises on failure -- the caller (web_chat.py) is responsible for
    catching and displaying errors, so there is only one place in the
    app that handles failures.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])

    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.7,
        max_tokens=8192,  # left exactly as you had it
    )
    return completion.choices[0].message.content
