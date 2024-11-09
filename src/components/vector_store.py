from components.loader import loaded_data
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def get_vector_store():
    docs = loaded_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vector_store = InMemoryVectorStore.from_documents(
        documents=splits, embedding = OllamaEmbeddings(model="llama3.1",)
    )
    return vector_store
