# Embedding Service

FastAPI-based text embedding service using sentence-transformers.


## Quick Start

```bash
git clone https://github.com/ozermehmett/embedding-service.git
cd embedding-service
docker compose up
```


## Usage

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).


## Config

Create `.env` file:
```env
API_PORT=8000
DEFAULT_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
```


## API

- `POST /embed` - Generate embeddings for text
