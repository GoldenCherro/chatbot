from components.vector_store import get_vector_store
from components.chain_rag import get_retriever_chain, get_rag_chain

def get_response(user_input, session):
    retriever = get_retriever_chain(get_vector_store())
    rag_chain = get_rag_chain(retriever)
    response = rag_chain.invoke({
        "chat_history": session.chat_history,
        "input":user_input
    })
    return response["answer"]
