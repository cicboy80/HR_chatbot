# HR Q&A Bot (FastAPI + Azure Deployment)

Production-oriented Retrieval-Augmented Generation (RAG) service for querying HR and policy documents. The system ingests PDFs, indexes them into a vector database, and serves context-grounded answers via a FastAPI API deployed on Azure.

## Features

- PDF text extraction and semantic chunking
- Embedding generation using **OpenAI `text-embedding-3-small`**
- Vector storage and retrieval via **Weaviate Cloud**
- Query expansion (**GPT-4.1-mini**) and passage re-ranking (**GPT-4o-mini**)
- Context-grounded answer generation with source-aware prompts
- Per-IP rate limiting and upload size/page caps to bound API spend
- Containerised deployment using **Docker** (non-root container user)
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

### 2. Create an Environment File

Copy the example file and fill in your real keys (`api_keys.env` is gitignored — never commit it):

```bash
cp api_keys.env.example api_keys.env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

For running the tests: `pip install -r requirements-dev.txt`, then `pytest`.

### 4. Run Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API docs at:
http://localhost:8000/docs

## Docker Deployment

### 1. Build the image
```bash
docker build -t hr-qa-bot .
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

## Configuration

All settings come from environment variables (loaded from `api_keys.env` locally):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | yes | — | OpenAI API key |
| `WEAVIATE_URL` | yes | — | Weaviate Cloud cluster URL |
| `WEAVIATE_API_KEY` | yes | — | Weaviate API key |
| `UPLOAD_RATE_LIMIT` | no | `3/day` | Per-IP limit on PDF uploads |
| `ASK_RATE_LIMIT` | no | `20/hour` | Per-IP limit on questions |
| `MAX_UPLOAD_MB` | no | `25` | Max PDF file size |
| `MAX_PDF_PAGES` | no | `100` | Max pages per PDF |
| `MAX_CHUNKS_PER_UPLOAD` | no | `500` | Max chunks embedded per upload |
| `API_URL` | no | `http://127.0.0.1:8000` | Base URL the Gradio UI uses to reach the API |

## Cost Protection

The app is public (no login), so spend is bounded in layers:

1. **Per-IP rate limits** on `/upload_pdf` and `/ask_question` (see table above). Behind Azure's front end the real client IP is taken from `X-Forwarded-For`.
2. **Upload caps** — file size, page count, and chunks-embedded-per-upload are all limited.
3. **OpenAI hard budget cap (do this!)** — in the [OpenAI dashboard](https://platform.openai.com/settings/organization/limits), set a monthly budget limit. This is the one protection that cannot be bypassed: the API stops serving once the cap is hit.

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
