import os
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY")))

def get_answer(question: str) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are the official AI assistant for Egypt Informatics University (EUI). Be helpful, accurate, and concise."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"عذراً، حدث خطأ: {str(e)}"
