import streamlit as st
import requests

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
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={"question": prompt}
                )
                answer = response.json().get("answer", "No response received.")
            except Exception as e:
                answer = f"Error connecting to backend: {e}"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
