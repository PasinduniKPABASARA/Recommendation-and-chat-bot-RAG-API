from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os


# --- 1. INITIALIZATION ---

# Load environment variables (.env locally; on Render, use Dashboard env vars)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not found. Please set GEMINI_API_KEY in the environment.")


# Global variables for models
embeddings = None
llm = None
vector_store = None
retriever = None
rag_chain = None
chat_chain = None


def ensure_models_loaded():
    """
    Lazy-load embeddings, LLM, vector store and chains.
    Only actually loads them once; later calls reuse globals.
    """
    global embeddings, llm, vector_store, retriever, rag_chain, chat_chain

    if rag_chain is not None and chat_chain is not None:
        return  # already loaded

    print("Loading models and vector store lazily...")

    # 1. Embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. LLM
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

    # 3. Vector store + retriever
    if vector_store is None:
        # Make sure the 'faiss_index' directory is in your repo
        # (it should contain the FAISS index files created offline)
        local_path = "faiss_index"
        if not os.path.exists(local_path):
            raise RuntimeError(f"FAISS index directory not found at: {local_path}")
        vs = FAISS.load_local(
            local_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        globals()["vector_store"] = vs

    if retriever is None:
        globals()["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})

    # 4. RAG prompt & chain
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

    globals()["rag_chain"] = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat chain
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

    globals()["chat_chain"] = (
        {"message": RunnablePassthrough()}
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    print("Models and vector store loaded successfully.")


# --- 2. FASTAPI APP ---

app = FastAPI(
    title="Coconut Disease RAG API",
    description="An API to get recommendations for coconut diseases using a RAG system with Gemini.",
    version="1.0.0",
)


# Pydantic model for request body validation
class QueryRequest(BaseModel):
    disease: str


# Pydantic model for chat requests
class ChatRequest(BaseModel):
    message: str
    use_knowledge: bool = True


# --- 3. API ENDPOINTS ---

@app.post("/get_recommendation")
async def get_recommendation(request: QueryRequest):
    """
    Endpoint to get a recommendation for a given disease.
    Expects a JSON payload with a 'disease' key.
    Example: {"disease": "My coconut leaves have yellow spots"}
    """
    try:
        ensure_models_loaded()

        query = request.disease
        if not query:
            raise HTTPException(status_code=400, detail="The 'disease' field cannot be empty.")

        print(f"Received query: {query}")

        response = rag_chain.invoke(query)

        print(f"Generated response: {response}")

        return {"recommendation": response}

    except HTTPException:
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.post("/coco_chat")
async def coco_chat(request: ChatRequest):
    """
    Conversational chatbot endpoint named 'Coco Chat'.
    Request body: { "message": "...", "use_knowledge": true }
    If use_knowledge is true (default), Coco Chat will attempt to ground answers using the
    project's knowledge base via the RAG chain. If false, it will answer directly using the LLM.
    """
    try:
        ensure_models_loaded()

        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="The 'message' field cannot be empty.")

        print(f"Coco Chat received: {request.message} (use_knowledge={request.use_knowledge})")

        if request.use_knowledge:
            response = rag_chain.invoke(request.message)
        else:
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


# --- 4. LOCAL / RENDER ENTRYPOINT ---

if _name_ == "_main_":
    import uvicorn

    port_str = os.environ.get("PORT")
    if port_str is None:
        # Local dev fallback
        print("Warning: PORT environment variable not set. Using 8000 for local development.")
        port = 8000
    else:
        port = int(port_str)
        print(f"Starting server on port: {port}")

    # 'main:app' assumes this file is named main.py
    uvicorn.run("main:app", host="0.0.0.0", port=port)