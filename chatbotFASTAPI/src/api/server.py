from fastapi import FastAPI
from langserve import add_routes
from components.stateful_chain  import conversational_rag_chain


server_app = FastAPI(
    title="Botsito Server",
    description="A FastAPI server for Botsito",
    version="0.1.0"
)

add_routes(server_app, conversational_rag_chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(server_app, host="localhost", port=8000)
