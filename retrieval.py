from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever
from vector_store import get_vector_store
from config import settings

def get_llm():
    """Returns the LLM instance based on settings."""
    # Note: Requires GEMINI_API_KEY in environment or .env
    return ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0, google_api_key=settings.GEMINI_API_KEY)

def get_qa_chain():
    """
    Constructs and returns the Conversational RAG chain.
    Includes memory/history awareness, strict instructions for source citations,
    and guardrails against hallucinations.
    """
    llm = get_llm()
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 1. History Aware Retriever
    # This prompt helps the LLM reformulate the user's question into a standalone question
    # if it relies on chat history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 2. Q&A Chain
    # This is the strict prompt ensuring source grounding and safety.
    system_prompt = (
        "You are an expert, strict, and highly factual Smart Contract Assistant. "
        "Use the following retrieved context to answer the user's question. "
        "You MUST base your answer entirely on the provided context. "
        "If the answer is not in the context, explicitly state 'I cannot answer this based on the provided document'. "
        "Do not make up or hallucinate any information. "
        "Always refer to the specific clauses, sections, or documents when providing an answer. "
        "\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Final Retrieval Chain
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return retrieval_chain

def summarize_text(text: str) -> str:
    """Optional utility to summarize a given piece of text."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Please provide a concise and factual summary of the following text:\n\n{text}"
    )
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content
