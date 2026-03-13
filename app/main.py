
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import RedirectResponse
import os, traceback
import requests
import gradio as gr
import atexit
import re
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # goes from app/ -> project root
load_dotenv(BASE_DIR / "api_keys.env")

from app.pdf_utils import extract_text_from_pdf, chunk_text
from app.llm_utils import rerank_chunks_with_llm, client as openai_client
from app.weaviate_utils import connect, insert_chunks, ensure_schema, search_weaviate

app = FastAPI(title="HR Q&A Bot")

# ENV + CONNECTION
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

print("🔍 DEBUG: WEAVIATE_URL =", WEAVIATE_URL)
print("🔍 DEBUG: OPENAI_API_KEY =", "SET" if OPENAI_API_KEY else "MISSING")

if not WEAVIATE_URL:
    raise ValueError("❌ Missing WEAVIATE_URL in environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in environment variables.")
if not WEAVIATE_API_KEY:
    raise ValueError("❌ Missing WEAVIATE_API_KEY in environment variables.")

try:
    client = connect(WEAVIATE_URL, WEAVIATE_API_KEY)
    print("✅ Connected to Weaviate")
    ensure_schema(client)   # create collection
except Exception as e:
    print(f"❌ Failed to connect to Weaviate: {e}")
    client = None

# UPLOAD DIRECTORY
UPLOAD_DIR = Path("/home/uploads") if os.getenv("WEBSITE_SITE_NAME") else Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# API ENDPOINTS
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    """Upload and index a PDF file in Weaviate"""
    try:
        if not client:
            raise HTTPException(status_code=503, detail="Weaviate is not connected")

        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")

        safe_name = Path(file.filename).name
        save_path = UPLOAD_DIR / safe_name

        with open(save_path, "wb") as f:
            f.write(await file.read())

        try:
            text = extract_text_from_pdf(save_path)
        except ValueError as err:
            # pdf_utils uses ValueError for "no extractable text"
            raise HTTPException(status_code=400, detail=str(err))

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in this PDF.")
        
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="PDF produced 0 chunks after processing.")

        # NO MORE SCHEMA WIPE PER UPLOAD
        result = insert_chunks(client, chunks, safe_name)

        return {
            "status": "success",
            "message": f"✅ PDF '{safe_name}' processed successfully.",
            "chunks": len(chunks),
            "inserted": result["inserted"],
            "skipped_existing": result["skipped_existing"],
            "unique_in_upload": result["unique_in_upload"],
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Upload error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_question")
async def ask_question(query: str = Form(...)):
    print("🔥 HIT /ask_question endpoint")
    """Answer a user question using retrieved PDF context."""
    try:
        if not client:
            raise HTTPException(status_code=503, detail="Weaviate is not connected")

        retrieved = search_weaviate(client, query, k=20)
        if not retrieved:
            return {
                "answer": "I couldn't find anything relevant in the uploaded handbook. Try uploading the PDF again or rephrasing your question.",
                "retrieved_docs": [],
                "reranked_docs": [],
            }
        reranked = rerank_chunks_with_llm(query, retrieved)
        top_docs = reranked[:4]

        context = "\n\n---\n\n".join(doc["text"] for doc in top_docs)

        prompt = f"""
You are an HR assistant answering questions from the staff handbook.
Use only the following content to answer accurately and concisely:

{context}

Question: {query}
Answer:
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful HR assistant. "
                        "Answer only from the provided excerpts. "
                        "If the excerpts do not contain the answer, say that you cannot find it in the provided handbook content. "
                        "Do NOT invent or infer policy details that are not present. "
                        "Do NOT wrap the full answer in quotation marks. "
                        "Quote only short phrases when necessary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        # --- DEBUG: show exact characters in logs (Azure will show this) ---
        raw = response.choices[0].message.content.strip()
        print("🔎 RAW LLM OUTPUT repr:", repr(raw))

        # --- Strip straight + curly quotes from the start/end ---
        answer = re.sub(r'^[\"“”‘’]+|[\"“”‘’]+$', '', raw).strip()

        return {
            "answer": answer,
            "retrieved_docs": retrieved,
            "reranked_docs": top_docs,
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# ✅ GRADIO UI (LOCAL API)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def upload_pdf_ui(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF."

    with open(pdf_file.name, "rb") as f:
        files = {"file": f}
        r = requests.post(f"{API_URL}/upload_pdf", files=files)

    if r.status_code != 200:
        return f"❌ {r.text}"
    
    data = r.json()
    if data.get("status") == "success":
        return data.get("message", "✅ PDF processed.")
    return f"❌ {data.get('message', 'Upload failed')}"

def ask_question_ui(question):
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""

    r = requests.post(f"{API_URL}/ask_question", data={"query": question})

    if r.status_code != 200:
        return f"❌ {r.text}", "", ""

    data = r.json()

    answer = data.get("answer", data.get("error", "Unknown error"))
    retrieved_docs = data.get("retrieved_docs", [])
    reranked_docs = data.get("reranked_docs", [])

    retrieved_text = "\n\n---\n\n".join(
        f"Chunk: {int(doc.get('chunk_index') or 0)} | Score: {doc.get('score')}\n{doc.get('text')}"
        for doc in retrieved_docs
    ) if retrieved_docs else "No retrieved docs."

    reranked_text = "\n\n---\n\n".join(
        f"Chunk: {int(doc.get('chunk_index') or 0)} | Score: {doc.get('score')}\n{doc.get('text')}"
        for doc in reranked_docs
    ) if reranked_docs else "No reranked docs."

    return answer, retrieved_text, reranked_text


with gr.Blocks(title="HR Q&A Bot") as gradio_app:
    gr.Markdown("## 🤖 HR Q&A Bot — Upload your HR PDF and ask questions")

    with gr.Tab("📄 Upload PDF"):
        pdf_input = gr.File(label="Upload HR policy PDF")
        upload_btn = gr.Button("Upload & Process")
        upload_output = gr.Textbox(label="Upload Status")
        upload_btn.click(upload_pdf_ui, inputs=pdf_input, outputs=upload_output)

    with gr.Tab("💬 Ask a Question"):
        question_input = gr.Textbox(label="Ask a question about your uploaded document")
        submit_btn = gr.Button("Get Answer")

        answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
        
        with gr.Accordion("Retrieved Docs", open=False):
            retrieved_output = gr.Textbox(label="Retrieved Docs", lines=14, interactive=False)
        with gr.Accordion("Reranked Docs Used", open=False):
            reranked_output = gr.Textbox(label="Reranked Docs Used", lines=14, interactive=False)

        submit_btn.click(
            ask_question_ui,
            inputs=question_input,
            outputs=[answer_output, retrieved_output, reranked_output],
        )

@app.get("/gradio_api/config")
def gradio_config_alias():
    return RedirectResponse(url="/gradio_api/info", status_code=307)

@app.get("/gradio_api/api")
def gradio_api_alias():
    return RedirectResponse(url="/gradio_api/info", status_code=307)

@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

# Mount Gradio into FastAPI
gr.mount_gradio_app(app, gradio_app, path="/ui")

# HEALTH + CLEANUP
@app.get("/health")
def health():
    return {"status": "ok"}


@atexit.register
def close_weaviate():
    if client:
        client.close()
        print("🔒 Weaviate connection closed.")