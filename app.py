import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline

app = FastAPI(title="EUI Hybrid Smart Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_DIR = "./chroma_db"
MEMORY_DIR = "./memory_db"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device}
)

if not os.path.exists(DB_DIR):
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

if os.path.exists(MEMORY_DIR):
    memory_db = Chroma(persist_directory=MEMORY_DIR, embedding_function=embeddings)
else:
    memory_db = Chroma.from_documents(
        [Document(page_content="EUI Continuous Learning Memory Store Initialized")], 
        embeddings, 
        persist_directory=MEMORY_DIR
    )

pipe = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-1.5B-Instruct", 
    max_new_tokens=200,
    device_map="auto" if torch.cuda.is_available() else None
)

class QueryModel(BaseModel):
    question: str

def extract_and_store_facts(user_input: str):
    """حفظ الحقائق والتصحيحات الخاصة بالجامعة"""
    input_lower = user_input.lower().strip()
    learning_triggers = [
        "is", "it's", "its", "هو", "هي", "عندنا", "يوجد", "فيها", "تحتوي", "تضم",
        "have", "has", "consists of", "contains", "president is", "secretary general is"
    ]
    is_question = input_lower.endswith("?") or input_lower.endswith("؟")
    
    if not is_question and any(trigger in input_lower for trigger in learning_triggers):
        fact_doc = Document(page_content=f"FACT: {user_input}")
        memory_db.add_documents([fact_doc])
        print(f"[CONTINUOUS LEARNING] Saved Fact: {user_input}")

def is_eui_related(text: str) -> bool:
    """تحديد ما إذا كان السؤال خاصاً بالجامعة أم سؤالاً عاماً"""
    keywords = [
        "eui", "egypt informatics university", "جامعة", "الجامعة", "عميد", "رئيس", 
        "كلية", "كليات", "مبنى", "مباني", "جامعه", "دكتور", "دكتورة", "مصاريف", "تنسيق",
        "secretary general", "president", "faculty", "faculties", "building", "campus"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

@app.post("/chat")
def chat_endpoint(data: QueryModel):
    user_input = data.question.strip()
    input_lower = user_input.lower()
    is_arabic = any('\u0600' <= char <= '\u06FF' for char in user_input)
    
    # 1. التجميع الذكي للتحيات المباشرة
    greetings_en = ["hi", "hello", "hey", "good morning", "good evening"]
    greetings_ar = ["اه", "أهلا", "اهلا", "مرحبا", "مرحباً", "السلام عليكم", "ازيك", "ازيك؟"]
    
    if input_lower in greetings_en:
        return {"question": data.question, "answer": "Hello! How can I help you today?", "retrieved_context": "Greeting"}
    elif input_lower in greetings_ar:
        return {"question": data.question, "answer": "أهلاً بك! كيف يمكنني مساعدتك اليوم؟", "retrieved_context": "Greeting"}

    # 2. حفظ الحقائق المكتسبة
    extract_and_store_facts(user_input)

    # 3. توجيه السؤال بناءً على ارتباطه بالجامعة
    if is_eui_related(user_input):
        # سؤال خاص بالجامعة -> RAG مع grounding صارم
        memory_results = memory_db.similarity_search(user_input, k=3)
        official_results = db.similarity_search(user_input, k=2)
        combined_docs = memory_results + official_results
        context = "\n---\n".join([doc.page_content for doc in combined_docs])
        
        if is_arabic:
            system_instruction = (
                "أنت المساعد الذكي لجامعة مصر للمعلوماتية (EUI).\n"
                "أجب عن السؤال بناءً على السياق (Context) المرفق فقط باللغة العربية. إذا كان هناك حقيقة FACT اعتمدها فوراً."
            )
        else:
            system_instruction = (
                "You are the official AI assistant for Egypt Informatics University (EUI).\n"
                "Answer using ONLY the provided Context below in English. Prioritize 'FACT:' entries."
            )
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
        ]
        res = pipe(messages, temperature=0.0, do_sample=False)
    else:
        # سؤال عام (نكتة، دردشة، معرفة العامة) -> إجابة من المعرفة العامة للموديل
        context = "General Knowledge Mode"
        if is_arabic:
            system_instruction = "أنت مساعد ذكي ولطيف. أجب عن سؤال المستخدم أو طلبه باللغة العربية بطريقة ممتازة وودودة."
        else:
            system_instruction = "You are a helpful and friendly AI assistant. Respond appropriately and accurately in English."
            
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ]
        res = pipe(messages, temperature=0.7, do_sample=True)

    bot_answer = res[0]['generated_text'][-1]['content']
    
    return {
        "question": data.question,
        "answer": bot_answer,
        "retrieved_context": context
    }
# دالة وسيطة لاستدعاء الـ Chatbot مباشرة من Streamlit بدون الحاجة لـ HTTP Requests
def get_answer(question: str) -> str:
    # إنشاء نموذج مؤقت لتمرير الداتا لنفس منطق الـ endpoint
    from fastapi import Request
    query_obj = QueryModel(question=question)
    response_dict = chat_endpoint(query_obj)
    return response_dict.get("answer", "No response generated.")
