# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import logging
from dotenv import load_dotenv
import uuid
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for GROQ API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables!")
    raise ValueError("GROQ_API_KEY is required")

logger.info("GROQ_API_KEY found in environment variables")

app = FastAPI()

# Configure CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # preflights can be cached for 24 hours
)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Session storage for message history
sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

def get_session(session_id: str):
    try:
        if session_id not in sessions:
            logger.info(f"Creating new session for {session_id}")
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama3-8b-8192",
                temperature=0.1
            )
            logger.info("Successfully created ChatGroq instance")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Enter your system prompt here"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            logger.info("Created chat prompt template")
            
            chain = prompt | llm
            logger.info("Created chain")
            
            sessions[session_id] = {
                "chain": chain,
                "messages": []
            }
            logger.info("Successfully initialized session with message history")
        return sessions[session_id]
    except Exception as e:
        logger.error(f"Error in get_session: {str(e)}", exc_info=True)
        raise

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Processing request for session {session_id}")
        logger.info(f"Received message: {request.message}")
        
        session = get_session(session_id)
        logger.info("Got session")
        
        # Add RAG context if available
        if vectorstore:
            docs = vectorstore.similarity_search(request.message, k=2)
            context = "\n".join([d.page_content for d in docs])
            input_with_context = f"Context: {context}\n\nQuestion: {request.message}"
            logger.info(f"Added RAG context: {context[:200]}...")
        else:
            input_with_context = request.message
            logger.info("No vectorstore available, using raw message")
        
        try:
            logger.info("Invoking LLM chain")
            logger.info(f"Input to chain: {input_with_context}")
            
            # Add message to history
            session["messages"].append(HumanMessage(content=request.message))
            
            # Invoke chain with history
            response = await session["chain"].ainvoke({
                "input": input_with_context,
                "history": session["messages"]
            })
            
            # Add response to history
            response_text = response.content if hasattr(response, 'content') else str(response)
            session["messages"].append(AIMessage(content=response_text))
            
            logger.info(f"Got response: {response_text[:200]}...")
            
            # Create response with custom headers
            json_response = JSONResponse(
                content={"response": response_text, "session_id": session_id}
            )
            json_response.headers["Access-Control-Allow-Origin"] = "*"
            json_response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            json_response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            
            return json_response
        except Exception as e:
            logger.error(f"Error during LLM chain invocation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/chat")
async def chat_options():
    # Handle OPTIONS preflight request
    response = JSONResponse(content={})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Add to existing vector store
        loader = TextLoader(temp_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        new_store = FAISS.from_documents(chunks, embeddings)
        vectorstore.merge_from(new_store)
        vectorstore.save_local("vectorstore")
        
        os.remove(temp_path)
        return {"message": f"Added {len(chunks)} new chunks to vector store"}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))