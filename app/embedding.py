from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import os
from dotenv import load_dotenv
from app.logger import setup_logger

# .env dosyasını yükle
load_dotenv()

# Logger setup
logger = setup_logger("embedding")

class EmbeddingEngine:
    def __init__(self):
        self.models = {}
        self.default_model = os.getenv('DEFAULT_MODEL', 'all-MiniLM-L6-v2')
        logger.info(f"Default Model: {self.default_model}")
        self.load_model(self.default_model)

        
    def load_model(self, model_name: str) -> SentenceTransformer:
        """Load and cache the model"""
        if model_name not in self.models:
            logger.info(f"Loading Model: {model_name}")
            try:
                self.models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Model loaded successfully")
            except Exception as e:
                logger.error(f"Model loading failed: {str(e)}")
                raise
        return self.models[model_name]
    
    def get_embeddings(self, texts: Union[str, List[str]], model_name: str = None) -> List[List[float]]:
        """Create embeddings for texts"""
        if model_name is None:
            model_name = self.default_model
            
        if isinstance(texts, str):
            texts = [texts]
                    
        try:            
            model = self.models.get(model_name) or self.load_model(model_name)
            embeddings = model.encode(texts)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error creating embed: {str(e)}")
            raise
    
    def get_model_dimensions(self, model_name: str = None) -> int:
        """Retturn embedding dimensions"""
        if model_name is None:
            model_name = self.default_model
            
        model = self.load_model(model_name)
        dimensions = model.get_sentence_embedding_dimension()
        return dimensions

# Global instance
embedding_engine = EmbeddingEngine()
