from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from loader import loaded_data
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def retrieve_data():
    docs = loaded_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding = OllamaEmbeddings(model="llama3.1",)
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    return retriever
