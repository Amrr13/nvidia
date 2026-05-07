from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

# Singleton-like patterns to avoid re-initializing models multiple times
_embeddings = None

def get_embeddings():
    """Initializes and returns the embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return _embeddings

def save_to_vector_store(documents):
    """Saves a list of LangChain Documents to the vector store."""
    if not documents:
        return None
        
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=settings.VECTOR_STORE_DIR
    )
    return vectorstore

def get_vector_store():
    """Returns the existing vector store."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=settings.VECTOR_STORE_DIR,
        embedding_function=embeddings
    )
