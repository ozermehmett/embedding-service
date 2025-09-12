from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import EmbedRequest, EmbedResponse
from app.embedding import embedding_engine
from app.logger import setup_logger
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Logger setup
logger = setup_logger("api")

# Environment variables
SERVICE_NAME = os.getenv('SERVICE_NAME', 'embedding-service')
VERSION = os.getenv('VERSION', '1.0.0')
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))

# FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="Embedding engine service",
    version=VERSION
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info(f"{SERVICE_NAME} v{VERSION} started")
    logger.info(f"API: http://{API_HOST}:{API_PORT}")

@app.post("/embed", response_model=EmbedResponse)
async def create_embeddings(request: EmbedRequest):
    """
    Generate embeddings for text(s)
    
    - **text**: A single string or a list of strings
    """
    
    # Request info
    logger.info(f"Embedding request received")
    
    try:
        embeddings = embedding_engine.get_embeddings(
            texts=request.text,
            model_name=None  # Use default model from .env
        )
        
        dimensions = embedding_engine.get_model_dimensions()
        
        used_model = embedding_engine.default_model
        
        logger.info(f"Embeddings generated successfully")
        
        return EmbedResponse(
            embeddings=embeddings,
            model=used_model,
            dimensions=dimensions
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error while generating embedding: {str(e)}"
        )

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{SERVICE_NAME} shutdown")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server: http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
    