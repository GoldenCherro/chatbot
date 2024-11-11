import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from components.vector_store import VectorStore
from components.chain_rag import get_retriever_chain, get_rag_chain
from langchain_ollama import ChatOllama

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

file_path = "sourceFiles/AIEngineer.pdf"
web_paths = ("https://www.promtior.ai/service","https://www.promtior.ai/")
llm = ChatOllama(model="llama3.1", temperature=0,)

vector_store = VectorStore(file_path, web_paths).get_vector_store()
retriever = get_retriever_chain(vector_store, llm)
rag_chain = get_rag_chain(retriever, llm)

def get_response(user_input: str, chat_history: list[AIMessage | HumanMessage], rag_chain):
    
    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input":user_input
    })
    return response["answer"]

#Streamlit Chatbot
st.title("💬 Promtior Chatbot")
st.caption("🚀 A Streamlit chatbot powered by CHERRO-AI")

chat_history = []
vector_store = []

#session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="I am a bot, how can I help you?")
    ]

if vector_store not in st.session_state:
      st.session_state.vector_store = vector_store

user_input=st.chat_input("Type your message here...")
if user_input is not None and user_input.strip()!="":
    response = get_response(user_input, st.session_state.chat_history, rag_chain)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
      if isinstance(message,AIMessage):
        with st.chat_message("AI"):
          st.write(message.content)
      else:
        with st.chat_message("Human"):
          st.write(message.content)
