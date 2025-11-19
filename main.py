from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not found. Please set it in the.env file.")

# Initialize FastAPI app
app = FastAPI(
    title="Coconut Disease RAG API",
    description="An API to get recommendations for coconut diseases using a RAG system with Gemini.",
    version="1.0.0"
)

# Pydantic model for request body validation
class QueryRequest(BaseModel):
    disease: str

# --- 2. RAG CHAIN SETUP ---
# This section is loaded once when the application starts
try:
    print("Loading models and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY, temperature=0.2, convert_system_message_to_human=True)
    
    # Load the vector store
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 results
    print("Models and vector store loaded successfully.")
except Exception as e:
    print(f"Error loading models or vector store: {e}")
    exit()

# Create the prompt template
template = """
You are an expert agricultural assistant specializing in coconut palm pests and diseases.
Your task is to provide a comprehensive and actionable recommendation for the user's query based ONLY on the following context.
Structure your answer clearly. If there are chemical, organic, and cultural practices, list them under separate headings.
If the information is not in the context, state that you do not have specific recommendations for that issue based on the provided data.
Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = PromptTemplate.from_template(template)

# Create the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Coco Chat (general conversational endpoint) ---
# Chat prompt for Coco Chat (general farmers' questions)
chat_template = """
You are Coco Chat, an expert and friendly agricultural assistant for smallholder coconut farmers.
Answer the user's question concisely and helpfully using plain language. If the question asks for a
recommendation and you can use the knowledge base for grounding, say so and provide actionable steps.
If the information is not available in the knowledge base, be honest and provide general best-practice guidance
or ask for clarification. Keep answers practical and safe.

User: {message}

Assistant:
"""
chat_prompt = PromptTemplate.from_template(chat_template)

# Chat chain for direct conversation (no retrieval)
chat_chain = (
    {"message": RunnablePassthrough()}
    | chat_prompt
    | llm
    | StrOutputParser()
)

# Pydantic model for chat requests
class ChatRequest(BaseModel):
    message: str
    use_knowledge: bool = True

# --- 3. API ENDPOINT ---
@app.post("/get_recommendation")
async def get_recommendation(request: QueryRequest):
    """
    Endpoint to get a recommendation for a given disease.
    Expects a JSON payload with a 'disease' key.
    Example: {"disease": "My coconut leaves have yellow spots"}
    """
    try:
        query = request.disease
        if not query:
            raise HTTPException(status_code=400, detail="The 'disease' field cannot be empty.")
            
        print(f"Received query: {query}")
        
        # Invoke the RAG chain to get the response
        response = rag_chain.invoke(query)
        
        print(f"Generated response: {response}")

        return {"recommendation": response}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# New Coco Chat endpoint
@app.post("/coco_chat")
async def coco_chat(request: ChatRequest):
    """
    Conversational chatbot endpoint named 'Coco Chat'.
    Request body: { "message": "...", "use_knowledge": true }
    If `use_knowledge` is true (default), Coco Chat will attempt to ground answers using the
    project's knowledge base via the RAG chain. If false, it will answer directly using the LLM.
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="The 'message' field cannot be empty.")

        print(f"Coco Chat received: {request.message} (use_knowledge={request.use_knowledge})")

        if request.use_knowledge:
            # Use RAG chain to provide grounded answer
            response = rag_chain.invoke(request.message)
        else:
            # Use direct chat chain for general conversation
            response = chat_chain.invoke(request.message)

        print(f"Coco Chat response: {response}")

        return {"assistant": "Coco Chat", "response": response}

    except HTTPException:
        raise
    except Exception as e:
        print(f"An error occurred in Coco Chat: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "API is running"}

   