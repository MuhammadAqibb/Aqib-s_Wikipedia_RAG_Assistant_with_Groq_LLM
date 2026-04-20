import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def ingest_data():
    print("Loading your documents...")

    # Load all PDFs and text files from a 'data' folder
    documents = []

    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created '{data_folder}' folder. Please add your documents there and run again.")
        return

    # Load PDF files
    for file in os.listdir(data_folder):
        filepath = os.path.join(data_folder, file)
        if file.endswith(".pdf"):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            print(f"Loading TXT: {file}")
            loader = TextLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        print("No documents found in the 'data' folder. Please add PDF or TXT files.")
        return

    print(f"Loaded {len(documents)} pages/documents")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in Pinecone
    index_name = os.getenv("PINECONE_INDEX_NAME")
    print(f"Storing in Pinecone index: {index_name}")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    print("✅ Success! Your data is now in Pinecone.")

if __name__ == "__main__":
    ingest_data()