# 🤖HR Q&A Bot (FastAPI + Azure Deployment)

This project converts the original Hugging Face **Gradio PDF Q&A Bot** into a **production-ready FastAPI API service** with Weaviate vector storage and OpenAI models. It allows users to upload a PDF, automatically index its contents, and query it using natural language.

__

## 🚀 Features

-Extracts and chunks PDF text
-Generates embeddings via OpenAI **'text-embedding-3-large'** model
-Stores vectors in **Weaviate cloud**
-Expands queries and re-ranks retrieved passages with **'gpt-4.1-mini'**
-Returns precise, context-grounded answers
-Containerized via **Docker** for deployment to **Azure**

__

## 🧩 Architecture

FastAPI(API layer)
│
├── /upload_pdf → extract → chunk → embed → index in Weaviate
└── /ask_question → retrieve → rerank → answer via GPT

### 📘Modules:
| File | Description |
|------|--------------|
| `pdf_utils.py` | Handles PDF extraction and text chunking |
| `weaviate_utils.py` | Manages vector DB operations |
| `llm_utils.py` | Query expansion, reranking, and embeddings |
| `main.py` | FastAPI route definitions and endpoints |

__

## ⚙️ Setup (Local Deployment)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/cicboy/hr-qa-bot.git
cd HR_chatbot
```

### 2️⃣ Create an Environment File
OPENAI_API_KEY=sk-your-openai-key
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key

### 3️⃣ Install Dependencies
```bash
pip install -r requirement.txt
```

### 4️⃣ Run Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API docs at:
http://localhost:8000/docs

## 🐳Docker Deployment

### 1️⃣Build the image
```bash
docker build -t hr-qa-bot
```

### 2️⃣Run the Container
```bash
docker run -p 8000:8000 --env-file api_keys.env hr-qa-bot
```

### 3️⃣Test
Open: http://localhost:8000/docs

## ☁️Azure Deployment 

### 1️⃣Build and Push to Azure Container Registry
```bash
az acr build --registry <your_registry> --image hr-qa-bot:v1 .
```
### 2️⃣Deploy to Azure Container Apps
```bash

az containerapp create \
    --name hr-qa-bot \
    --resource-group myResourceGroup \
    --image <acr>.azurecr.io/hr-qa-bot:v1 \
    --env-vars OPENAI_API_KEY=<key> WEAVIATE_URL=<url> WEAVIATE_API_KEY=<key>
```
Once deployed, your FastAPI endpoints will be live at:
https://hr-qa-bot.<region>.azurecontainerapps.io

## 🔍Example Flow

1. Upload your staff handbook via /upload_pdf

2. Ask: Who Should I contact if I am sick?

3. The API:

-Expands the question using GPT
-Retrieves and re-ranks PDF chunks from Weaviate
-Returns an HR-accurate 

### 🧠 Next Steps (Scaling & Monitoring)

-Add LangSmith or OpenTelemetry for trace logging
-Integrate JWT authentication for secure endpoints
-Implement batch PDF ingestion and async processing
-Connect to Azure Blob Storage for file persistence

## ✍️Author

Clyde Cossey
AI Engineer | Machine Learning Developer | RAG & Agent Systems Builder
📧cosseyclyde@gmail.com

## 🪪 License
MIT License - feel free to use, modify, and build upon this project