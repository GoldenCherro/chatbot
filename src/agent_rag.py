from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain import hub

def agent_RAG(retriever, llm): 
    memory = MemorySaver()
    prompt = hub.pull("rlm/rag-prompt")

    tool = create_retriever_tool(
        retriever,
        "promtior_info_retriever",
        "Searches and returns information about Promtior.",
    )
    tools = [tool]

    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    return agent_executor
