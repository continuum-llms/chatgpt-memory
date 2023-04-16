"""
Adapted from https://github.com/avrabyt/MemoryBot
"""

import requests

# Import necessary libraries
import streamlit as st

from chatgpt_memory.environment import OPENAI_API_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT

# Set Streamlit page configuration
st.set_page_config(page_title="🧠MemoryBot🤖", layout="wide")
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None


# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Your AI assistant here! Ask me anything ...",
        label_visibility="hidden",
    )
    return input_text


# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state["conversation_id"] = None


# Set up the Streamlit app layout
st.title("🤖 Chat Bot with 🧠")
st.subheader(" Powered by ChatGPT Memory + Redis Search")


# Session state storage would be ideal
if not OPENAI_API_KEY:
    st.sidebar.warning("API key required to try this app. The API key is not stored in any form.")
elif not (REDIS_HOST and REDIS_PASSWORD and REDIS_PORT):
    st.sidebar.warning(
        "Redis `REDIS_HOST`, `REDIS_PASSWORD`, `REDIS_PORT` are required to try this app. Please set them as env variables properly."
    )


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    # Use the ChatGPTClient object to generate a response
    url = "http://localhost:8000/converse"
    payload = {"message": user_input, "conversation_id": st.session_state.conversation_id}

    response = requests.post(url, json=payload).json()
    # Update the conversation_id with the conversation_id from the response
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = response["conversation_id"]
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response["chat_gpt_answer"])

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = ["\n".join(download_str)]
    if download_str:
        st.download_button("Download", download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
