import streamlit as st

# استدعاء دالة المعالجة من ملف app.py مباشرة
try:
  from app import get_answer
except ImportError:
  def get_answer(q):
    return "Backend function not found. Please check app.py."

st.set_page_config(page_title="EUI RAG Chatbot", page_icon="🤖")
st.title("🤖 EUI AI Support Assistant")

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("Ask something about EUI..."):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      try:
        answer = get_answer(prompt)
      except Exception as e:
        answer = f"Error processing query: {e}"
      
      st.markdown(answer)
      st.session_state.messages.append({"role": "assistant", "content": answer})
