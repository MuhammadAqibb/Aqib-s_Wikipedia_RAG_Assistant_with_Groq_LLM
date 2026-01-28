import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME, 
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)
retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    """You are a helpful Wikipedia assistant. Answer questions based on the context and conversation history.

Context from Wikipedia: {context}

Question: {question}

Important: If the question refers to something from earlier in the conversation (like "it", "they", "that", etc.), use the conversation history to understand what is being referred to. Answer directly without asking for clarification.

Answer:"""
)

chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)