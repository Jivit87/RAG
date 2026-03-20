import os
import sys
from dotenv import load_dotenv

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ------------------- CONFIG -------------------

PDF_PATH = "BOOK.pdf"
PERSIST_DIR = "./chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K = 4

LLM_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------- SYSTEM PROMPT -------------------

SYSTEM_PROMPT = """
You are an expert, multi-tasking AI assistant. You are capable of BOTH answering specific questions using the provided Context from a PDF document AND having normal, friendly conversations with the user.

YOUR DIRECTIVES:
1. **Multi-Tasking:** If the user is just saying hello, asking a personal question, making small talk, or asking about your conversation so far, answer them naturally and conversationally. Do NOT say "it's not in the context".
2. **Contextual Answers:** If the user asks a question about facts, summarize or answer strictly using the provided Context chunks.
3. **General Fallback:** If you truly cannot find a fact in the context, politely state it but continue the conversation smoothly.

=========================================
PDF Context Data:
{context}
=========================================

User Query: {question}

Assistant Response:
"""

# ------------------- STEP 1: LOAD PDF -------------------

def load_pdf():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()

# ------------------- STEP 2: SPLIT TEXT -------------------

def split_docs(docs):
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

# ------------------- STEP 3: VECTOR STORE -------------------

def get_vectorstore(chunks, embeddings):
    if os.path.exists(PERSIST_DIR):
        print("Loading existing DB...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    print("Creating new DB...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

# ------------------- STEP 4: CREATE CHAIN -------------------

def create_chain(vectorstore):
    print("Setting up model...")

    llm = ChatGroq(
        model_name=LLM_MODEL,
        temperature=0.2
    )

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": K}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return chain

# ------------------- MAIN -------------------

def main():
    if not os.path.exists(PDF_PATH):
        print("PDF not found")
        sys.exit()

    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    docs = load_pdf()
    chunks = split_docs(docs)

    vectorstore = get_vectorstore(chunks, embeddings)
    chain = create_chain(vectorstore)
    
    print("\nAsk questions (type 'exit' to quit)\n")

    while True:
        q = input("You: ")

        if q.lower() in ["exit", "quit"]:
            break

        result = chain.invoke({"query": q})
        print("AI:", result["result"], "\n")

if __name__ == "__main__":
    main()