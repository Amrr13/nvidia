import os
import shutil
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

from config import settings
from ingestion import ingest_file
from vector_store import save_to_vector_store
from retrieval import get_qa_chain, summarize_text

def handle_upload(file):
    if file is None:
        return "No file uploaded."
    
    try:
        # Save uploaded file
        filename = os.path.basename(file.name)
        save_path = os.path.join(settings.UPLOAD_DIR, filename)
        shutil.copy(file.name, save_path)
        
        # Process and ingest
        chunks = ingest_file(save_path)
        save_to_vector_store(chunks)
        
        total_chunks = len(chunks)
        if total_chunks > 0:
            lengths = [len(chunk.page_content) for chunk in chunks]
            avg_chunk = sum(lengths) / total_chunks
            min_chunk = min(lengths)
            max_chunk = max(lengths)
            stats = (
                f"Successfully processed {filename}.\n"
                f"- Total chunks: {total_chunks}\n"
                f"- Avg chunk size: {avg_chunk:.0f} characters\n"
                f"- Min chunk size: {min_chunk} characters\n"
                f"- Max chunk size: {max_chunk} characters"
            )
        else:
            stats = f"No content could be extracted from {filename}."
            
        return stats
    except Exception as e:
        return f"Error processing file: {str(e)}"

def handle_summarize(file):
    if file is None:
        return "No file uploaded."
    
    try:
        # We can extract text by ingesting the file and concatenating
        chunks = ingest_file(file.name)
        if not chunks:
            return "No content found in the document."
        
        full_text = "\n".join([chunk.page_content for chunk in chunks])
        
        # Truncate to first 12,000 chars to avoid LLM token limits for simple summary
        text_to_summarize = full_text[:12000]
        
        summary = summarize_text(text_to_summarize)
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def format_chat_history(gradio_history):
    """Converts Gradio chat history to LangChain message format."""
    langchain_history = []
    if not gradio_history:
        return langchain_history
        
    # Check if history is a list of dicts (Gradio 5/6 default)
    if isinstance(gradio_history[0], dict):
        for msg in gradio_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                langchain_history.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_history.append(AIMessage(content=content))
    # Check if history is a list of tuples/lists (Older Gradio format)
    else:
        for item in gradio_history:
            if len(item) == 2:
                human_msg, ai_msg = item
                if human_msg:
                    langchain_history.append(HumanMessage(content=human_msg))
                if ai_msg:
                    langchain_history.append(AIMessage(content=ai_msg))
                    
    return langchain_history

def chat_interface(message, history):
    """
    Handles chat messages, invoking the Conversational RAG chain
    and formatting the response with sources.
    """
    try:
        chain = get_qa_chain()
        formatted_history = format_chat_history(history)
        
        response = chain.invoke({
            "input": message,
            "chat_history": formatted_history
        })
        
        answer = response.get("answer", "I could not generate an answer.")
        context_docs = response.get("context", [])
        
        # Extract unique sources
        sources = set()
        for doc in context_docs:
            source = doc.metadata.get("source", "Unknown Document")
            sources.add(source)
            
        if sources:
            source_list = "\n".join([f"- {s}" for s in sources])
            formatted_answer = f"{answer}\n\n**Sources:**\n{source_list}"
        else:
            formatted_answer = answer
            
        return formatted_answer
    except Exception as e:
        return f"Error during retrieval: {str(e)}"

# Build the Gradio UI
with gr.Blocks(title=settings.APP_NAME, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 📄 {settings.APP_NAME}")
    gr.Markdown("Upload your smart contracts and ask questions about them. The AI will strictly use the uploaded documents and provide source citations.")
    
    with gr.Tabs():
        with gr.TabItem("1. Upload Documents"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="Upload PDF or DOCX file", file_types=[".pdf", ".docx"])
                    upload_button = gr.Button("Process Document", variant="primary")
                with gr.Column(scale=3):
                    upload_output = gr.Textbox(label="Status Logging", lines=5, interactive=False)
            
            upload_button.click(
                handle_upload,
                inputs=file_input,
                outputs=upload_output
            )
            
        with gr.TabItem("2. Document Summary"):
            with gr.Row():
                with gr.Column(scale=2):
                    summary_file_input = gr.File(label="Upload PDF or DOCX file to Summarize", file_types=[".pdf", ".docx"])
                    summary_button = gr.Button("Generate Summary", variant="secondary")
                with gr.Column(scale=3):
                    summary_output = gr.Textbox(label="Document Summary", lines=10, interactive=False)
            
            summary_button.click(
                handle_summarize,
                inputs=summary_file_input,
                outputs=summary_output
            )
            
        with gr.TabItem("3. Chat Assistant"):
            chatbot = gr.ChatInterface(
                fn=chat_interface,
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(placeholder="Ask a question about the uploaded contract...", container=False, scale=7),
                title="Contract Assistant",
                description="Ask questions. Responses will be grounded in the uploaded documents."
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
