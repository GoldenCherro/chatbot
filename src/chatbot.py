import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from chain_rag import chain_RAG
from agent_rag import agent_RAG
from transformer import retrieve_data
from dotenv import load_dotenv
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

index = retrieve_data()

model = chain_RAG(index, llm)

# def call_model(state: State):
#     print(state)
#     if "chat_history" not in state:
#         state["chat_history"] = []
    
#     state["chat_history"].append(HumanMessage(state["input"]))
#     response = model.invoke(state)
#     state["chat_history"].append(AIMessage(response["answer"]))
    
#     return {
#         "chat_history": state["chat_history"], 
#         "context": response["context"],
#         "answer": response["answer"],
#     }

# workflow = StateGraph(state_schema=State)
# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    model,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


st.title("💬 Promtior Chatbot")
st.caption("🚀 A Streamlit chatbot powered by CHERRO-AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Holu, How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

config = {"configurable": {"session_id": "abc123"}}

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    msg = conversational_rag_chain.invoke({"input": prompt}, config=config)
    print(msg["answer"])

    st.session_state.messages.append({"role": "assistant", "content": msg["answer"]})
    st.chat_message("assistant").write(msg["answer"])
