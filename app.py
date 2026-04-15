import streamlit as st
from LLM import stream_answer

st.set_page_config(
    page_title="Chat with Andres' AI",
    page_icon="🤖",
    layout="centered",
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar — branding + reset
with st.sidebar:
    st.title("Andres Cardenas")
    st.markdown(
        "Hi! I'm LOLA, an AI assistant trained on Andres' professional profile. "
        "Ask me anything about his background, experience, projects, or skills."
    )
    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat area
st.title("Chat with Andres' AI")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if question := st.chat_input("Ask about Andres..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Build (question, answer) pairs from prior turns, excluding the new question
        prior = st.session_state.messages[:-1]
        history_pairs = [
            (prior[i]["content"], prior[i + 1]["content"])
            for i in range(0, len(prior) - 1, 2)
            if prior[i]["role"] == "user" and prior[i + 1]["role"] == "assistant"
        ]

        try:
            answer = st.write_stream(stream_answer(question, history_pairs))
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Something went wrong while contacting the model: {e}")
