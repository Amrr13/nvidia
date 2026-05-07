from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from retrieval import get_qa_chain
from config import settings
import uvicorn

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0",
    description="API for RAG-based Smart Contract Analysis"
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Initialize the QA Chain
qa_chain = get_qa_chain()

# Add LangServe routes
# This exposes the chain at /chat/invoke, /chat/stream, etc.
add_routes(
    app,
    qa_chain,
    path="/chat"
)

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
