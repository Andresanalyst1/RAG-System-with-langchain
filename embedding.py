from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
)
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:335m")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 6))
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_langchain_md")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
DOCS_URLS = [u.strip() for u in os.getenv("DOCS_URLS", "").split(",") if u.strip()]

LOADER_MAP = {
    ".md":   UnstructuredMarkdownLoader,
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
}


def load_documents():
    docs = []

    # Load files from docs/ directory
    if os.path.isdir(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            ext = os.path.splitext(filename)[1].lower()
            loader_cls = LOADER_MAP.get(ext)
            if loader_cls:
                path = os.path.join(DOCS_DIR, filename)
                print(f"[INFO] Loading file: {filename}")
                docs.extend(loader_cls(path).load())
            else:
                print(f"[WARN] Unsupported file type, skipping: {filename}")

    # Load URLs
    for url in DOCS_URLS:
        print(f"[INFO] Loading URL: {url}")
        docs.extend(WebBaseLoader(url).load())

    return docs


# Load and split all documents
raw_docs = load_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
content = text_splitter.split_documents(raw_docs)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="rag_documents",
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# Add documents only if the collection is empty
if vector_store._collection.count() == 0:
    print(f"[INFO] Adding {len(content)} chunks to vector store...")
    vector_store.add_documents(content)
else:
    print(f"[INFO] Vector store already has {vector_store._collection.count()} chunks.")


retriever = vector_store.as_retriever(
    search_kwargs={"k": RETRIEVER_K}
)
