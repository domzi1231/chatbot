# ingest.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import sys

def create_vector_store(file_path):
    # Load document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and save vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print(f"Vector store created with {len(chunks)} chunks")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_text_file>")
        sys.exit(1)
    create_vector_store(sys.argv[1])