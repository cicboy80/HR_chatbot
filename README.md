# HR Q&A Bot (FastAPI + Azure Deployment)

Production-oriented Retrieval-Augmented Generation (RAG) service for querying HR and policy documents. The system ingests PDFs, indexes them into a vector database, and serves context-grounded answers via a FastAPI API deployed on Azure.

## Features

- PDF text extraction and semantic chunking
- Embedding generation using **OpenAI `text-embedding-3-large`**
- Vector storage and retrieval via **Weaviate Cloud**
- Query expansion and passage re-ranking using **GPT-4.1-mini**
- Context-grounded answer generation with source-aware prompts
- Containerised deployment using **Docker**
- Cloud deployment on **Azure Container Apps**
  
__

## Architecture

The system follows a standard RAG pipeline exposed via a FastAPI service:

FastAPI(API layer)
│
├── /upload_pdf → extract → chunk → embed → index in Weaviate
└── /ask_question → retrieve → rerank → answer via GPT

### Modules:
| File | Description |
|------|--------------|
| `pdf_utils.py` | Handles PDF extraction and text chunking |
| `weaviate_utils.py` | Manages vector DB operations |
| `llm_utils.py` | Query expansion, reranking, and embeddings |
| `main.py` | FastAPI route definitions and endpoints |

__

## Setup (Local Deployment)

### 1. Clone the Repository
```bash
git clone https://github.com/cicboy/hr-qa-bot.git
cd HR_chatbot
```

### 2.Create an Environment File

Create a `.env` file (do not commit):

OPENAI_API_KEY=sk-your-openai-key
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API docs at:
http://localhost:8000/docs

## Docker Deployment

### 1. Build the image
```bash
docker build -t hr-qa-bot
```

### 2. Run the Container
```bash
docker run -p 8000:8000 --env-file api_keys.env hr-qa-bot
```

### 3. Test
Open: http://localhost:8000/docs

## Azure Deployment 

The service is designed to run as a stateless container on Azure Container Apps, with secrets injected via environment variables.

### 1. Build and Push to Azure Container Registry
```bash
az acr build --registry <your_registry> --image hr-qa-bot:v1 .
```
### 2. Deploy to Azure Container Apps
```bash

az containerapp create \
    --name hr-qa-bot \
    --resource-group myResourceGroup \
    --image <acr>.azurecr.io/hr-qa-bot:v1 \
    --env-vars OPENAI_API_KEY=<key> WEAVIATE_URL=<url> WEAVIATE_API_KEY=<key>
```
Once deployed, your FastAPI endpoints will be live at:
https://hr-qa-bot.<region>.azurecontainerapps.io

## Example Flow

1. Upload your staff handbook via /upload_pdf

2. Ask: Who Should I contact if I am sick?

3. The API:

- Expands the question using GPT
- Retrieves and re-ranks PDF chunks from Weaviate
- Returns an HR-accurate, context-grounded response derived exclusively from the uploaded document 

### Next Steps (Scaling & Monitoring)

- Add structured tracing and observability (e.g. OpenTelemetry / LangSmith)
- Integrate JWT authentication for secure endpoints
- Implement batch PDF ingestion and async processing
- Connect to Azure Blob Storage for file persistence

## License
MIT License - feel free to use, modify, and build upon this project
