import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from vector_store import get_vector_store
from api.get_response import get_response

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

vector_store = get_vector_store()

#Streamlit Chatbot
st.title("💬 Promtior Chatbot")
st.caption("🚀 A Streamlit chatbot powered by CHERRO-AI")

chat_history = []
vector_store= []

#session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="I am a bot, how can I help you?")
    ]

if vector_store not in st.session_state:
      st.session_state.vector_store = get_vector_store()

user_input=st.chat_input("Type your message here...")
if user_input is not None and user_input.strip()!="":
    response = get_response(user_input, st.session_state)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
      if isinstance(message,AIMessage):
        with st.chat_message("AI"):
          st.write(message.content)
      else:
        with st.chat_message("Human"):
          st.write(message.content)
