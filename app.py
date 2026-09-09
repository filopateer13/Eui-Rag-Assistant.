import os
import streamlit as st
from fastapi import FastAPI, Request
from pydantic import BaseModel
from groq import Groq  # استخدام Groq السريع والخفيف

# تهيئة FastAPI (لو حابب يستقبل طلبات API)
app = FastAPI()

# تهيئة عميل Groq (بيسحب المفتاح من البيئة أوتوماتيك)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class QueryModel(BaseModel):
    question: str

@app.post("/chat")
def chat_endpoint(data: QueryModel):
    user_input = data.question
    
    # منطق الـ RAG والرد الذكي
    try:
        completion = client.chat.completions.create(
            model="qwen-2.5-32b",  # موديل قوي وسريع جداً على Groq
            messages=[
                {
                    "role": "system",
                    "content": "You are the official AI assistant for Egypt Informatics University (EUI). Be helpful, accurate, and concise."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        bot_answer = completion.choices[0].message.content
    except Exception as e:
        bot_answer = f"عذراً، حدث خطأ في الاتصال بالذكاء الاصطناعي: {str5(e)}"

    return {
        "question": user_input,
        "answer": bot_answer,
        "retrieved_context": "Groq Cloud API Mode"
    }

# الدالة الوسيطة لربطها بملف web_chat.py مباشرة على Streamlit
def get_answer(question: str) -> str:
    query_obj = QueryModel(question=question)
    response_dict = chat_endpoint(query_obj)
    return response_dict.get("answer", "No response generated.")
