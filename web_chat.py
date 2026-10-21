import streamlit as st

try:
    from app import get_answer
except ImportError:
    def get_answer(question, history=None):
        return "Backend function not found. Please check app.py."

st.set_page_config(page_title="EUI RAG Chatbot", page_icon="\U0001F393")
st.title("\U0001F393 EUI AI Support Assistant")

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
                # Pass everything before this new question as history,
                # so follow-up questions keep context.
                answer = get_answer(prompt, st.session_state.messages[:-1])
            except Exception as e:
                # Single place in the app that handles a failed request.
                answer = f"Sorry, something went wrong while processing your question: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
