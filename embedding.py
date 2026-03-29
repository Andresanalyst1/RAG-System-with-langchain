from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# Reading .md file
loader = UnstructuredMarkdownLoader('andres_cardenas_profile.md')
content = loader.load()

# Split the document into chunks for better RAG
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
content = text_splitter.split_documents(content)

embeddings = OllamaEmbeddings(model='mxbai-embed-large:335m')

md_location = './chroma_langchain_md'
add_documents = not os.path.exists(md_location)

vector_store = Chroma(
    collection_name='andres_cardenas_profile.md',
    persist_directory=md_location,
    embedding_function=embeddings
)

# Add documents only if the collection is empty
if vector_store._collection.count() == 0:
    print(f"[INFO] Adding {len(content)} chunks to vector store...")
    vector_store.add_documents(content)
else:
    print(f"[INFO] Vector store already has {vector_store._collection.count()} chunks.")
    

retriever = vector_store.as_retriever(
    search_kwargs={'k': 6}
)