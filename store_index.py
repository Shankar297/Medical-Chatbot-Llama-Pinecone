from src.helper import load_pdf, text_split, download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


extracted_data = load_pdf("data/")

text_chunks = text_split(extracted_data)

embedding = download_hugging_face_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_documents(text_chunks, embedding, index_name=index_name)