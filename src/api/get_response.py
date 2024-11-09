def get_response(user_input, session, rag_chain):
    
    response = rag_chain.invoke({
        "chat_history": session.chat_history,
        "input":user_input
    })
    return response["answer"]
