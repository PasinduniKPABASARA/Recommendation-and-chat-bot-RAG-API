import json
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

# 1. Load the structured knowledge base
with open('data/knowledge_base.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Ensure all summaries are strings and handle potential missing values
df['recommendation_summary'] = df['recommendation_summary'].astype(str).fillna('')
documents = df['recommendation_summary'].tolist()
metadatas = df.to_dict('records')

# 2. Initialize the Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create and save the FAISS vector store
print("Creating vector store...")
vector_store = FAISS.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas
)

# Save the vector store locally in the root directory
vector_store.save_local("faiss_index")
print("Vector store created and saved to 'faiss_index'")