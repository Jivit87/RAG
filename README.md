# Local Document RAG Chatbot

A high-performance local Retrieval-Augmented Generation (RAG) system utilizing the lightning-fast Groq API, LangChain, and Chroma vector database to chat directly with your PDF documents.

## Features
- **Fast Generation**: Uses the cutting-edge Groq API (powered by `llama-3.3-70b-versatile`).
- **Local Embeddings**: Fully local, open-source embedding pipeline powered by HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.
- **Memory Buffer**: Retains your conversation context memory for complex follow-up questions.
- **Persistent Vector Store**: Securely builds and queries a local `chroma_db` database, caching embeddings so you don't rebuild them on every run.

## Prerequisites
- Python 3.10+
- A free Groq API Key (get yours at [console.groq.com](https://console.groq.com)).

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```

2. **Create a stable virtual environment:**
   We recommend using a fresh virtual environment.
   ```bash
   python3 -m venv venv_stable
   source venv_stable/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   Create a `.env` file in the root directory and securely add your Groq API Key:
   ```env
   GROQ_API_KEY=gsk_your_api_key_here
   ```

## Usage

1. **Prepare your document:**
   Place your PDF document into the project directory and name it `BOOK.pdf` (or update the `PDF_PATH` configuration inside `RAG.py`).

2. **Run the chatbot:**
   ```bash
   python RAG.py
   ```
   *Note: On your first run, the local HuggingFace embeddings model will transparently download and cache, which may take up to a minute.*

3. **Chat interactively** through the command line prompt. Type `exit` or `quit` to end the session.
