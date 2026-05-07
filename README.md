# NVIDIA RAG Application

A Retrieval-Augmented Generation (RAG) application that uses NVIDIA AI endpoints or NeMo-based services to answer user queries using indexed document context.

The project includes document ingestion, vector storage, retrieval logic, a backend API server, and a frontend user interface.

## Features

- Document ingestion pipeline
- Local vector database using ChromaDB
- Retrieval-Augmented Generation workflow
- Backend API service
- Frontend user interface
- NVIDIA API integration
- Configurable environment variables
- Optional pre-built vector store support

## Project Structure

```text
.
├── ingestion.py
├── vector_store.py
├── retrieval.py
├── server.py
├── ui.py
├── config.py
├── requirements.txt
└── data/
    └── vector_store/
```

## File Descriptions

| File / Folder | Description |
|---|---|
| `ingestion.py` | Processes documents and loads them into the vector store. |
| `vector_store.py` | Manages the ChromaDB vector database. |
| `retrieval.py` | Handles retrieval logic and fetches relevant context for user queries. |
| `server.py` | Runs the backend API service. |
| `ui.py` | Runs the frontend user interface. |
| `config.py` | Stores configuration logic and environment variable handling. |
| `requirements.txt` | Contains the Python dependencies required for the project. |
| `data/vector_store/` | Stores the local ChromaDB database and vector index files. |

## Prerequisites

Before running the project, make sure you have:

- Python 3.8 or higher
- Git
- NVIDIA API key
- Virtual environment support

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

On Linux or macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root or set the environment variables directly in your terminal.

Example:

```bash
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

On Windows PowerShell:

```powershell
$env:NVIDIA_API_KEY="your_nvidia_api_key_here"
```

If your `config.py` file uses additional environment variables, add them in the same way.

## Running the Application

### Step 1: Run Document Ingestion

Run this step if you want to process new documents or rebuild the vector store.

```bash
python ingestion.py
```

If a pre-built vector store already exists inside `data/vector_store/`, this step may be optional.

### Step 2: Start the Backend Server

If the backend uses FastAPI and Uvicorn, run:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

If your project starts the server directly from `server.py`, run:

```bash
python server.py
```

### Step 3: Start the Frontend UI

If the UI uses Streamlit, run:

```bash
streamlit run ui.py
```

If the UI runs directly as a Python script, run:

```bash
python ui.py
```

## Typical Workflow

```text
Documents
   ↓
ingestion.py
   ↓
ChromaDB vector store
   ↓
retrieval.py
   ↓
server.py backend API
   ↓
ui.py frontend interface
   ↓
User receives answer with retrieved context
```

## Dependencies

The project may require packages such as:

```text
langchain
chromadb
fastapi
uvicorn
streamlit
nvidia-ai-endpoints
python-dotenv
```

For the complete dependency list, check:

```bash
requirements.txt
```

## Configuration

Main configuration is handled in:

```text
config.py
```

This file may include:

- API key loading
- Model configuration
- Vector database path
- NVIDIA endpoint settings
- Retrieval parameters

## Notes

- Do not commit your API keys to GitHub.
- Keep sensitive credentials inside `.env` or system environment variables.
- Re-run `ingestion.py` after adding or modifying source documents.
- Make sure the backend server is running before launching the frontend UI.
- If the UI cannot connect to the backend, check the API host and port settings.

## Troubleshooting

### Missing API Key

If you see an authentication error, confirm that `NVIDIA_API_KEY` is correctly set.

```bash
echo $NVIDIA_API_KEY
```

On Windows PowerShell:

```powershell
echo $env:NVIDIA_API_KEY
```

### Dependency Errors

Reinstall the dependencies:

```bash
pip install -r requirements.txt
```

### Vector Store Not Found

Run the ingestion script:

```bash
python ingestion.py
```

### Backend Not Running

Start the backend before opening the UI:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Security Recommendations

- Store API keys in environment variables.
- Never hard-code credentials in source files.
- Add `.env` to `.gitignore`.
- Avoid uploading vector databases containing private documents unless needed.
- Review the project before making it public.

## Example `.gitignore`

```gitignore
venv/
.env
__pycache__/
*.pyc
data/vector_store/
.DS_Store
```

## License

Add your project license here.

Example:

```text
MIT License
```

## Author

Add your name or team name here.
