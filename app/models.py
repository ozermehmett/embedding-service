from pydantic import BaseModel
from typing import List, Union
import os
from dotenv import load_dotenv

load_dotenv()

class EmbedRequest(BaseModel):
    text: Union[str, List[str]]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
