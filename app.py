import json
import streamlit as st

from chatbot import ChatBot

st.title('JSON QA')

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

# File uploader
uploaded_file = st.file_uploader("Upload a JSON file", type=['json'])

if uploaded_file is not None:
    try:
        # Read and validate JSON
        data = json.load(uploaded_file)

        # Validate that it's a list of objects
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            st.error("The JSON file must contain a list of objects")
        else:
            # Save file and initialize chatbot
            # Convert data to JSON string to keep in memory
            json_str = json.dumps(data)

            st.session_state.chatbot = ChatBot(
                model="gpt-4o",
                file_path=json_str
            )
            st.success("ChatBot initialized successfully!")
    except json.JSONDecodeError:
        st.error("Invalid JSON file")

# Chat interface
if st.session_state.chatbot is not None:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your data"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get chatbot response
        with st.chat_message("assistant"):
            answer = st.session_state.chatbot.generate_response_from_ai(
                prompt)
            if answer:
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
            else:
                st.write("I'm sorry, I cannot answer your question.")
