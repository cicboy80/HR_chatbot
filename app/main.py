from fastapi import FastAPI, UploadFile, Form
import os, traceback
import requests
import gradio as gr
import atexit
from dotenv import load_dotenv
from pathlib import Path

from app.pdf_utils import extract_text_from_pdf, chunk_text
from app.llm_utils import rerank_chunks_with_llm, client as openai_client
from app.weaviate_utils import connect, setup_schema, insert_chunks, search_weaviate

load_dotenv()

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
except Exception as e:
    print(f"❌ Failed to connect to Weaviate: {e}")
    client = None

# UPLOAD DIRECTORY

UPLOAD_DIR = Path("tmp/uploads") if os.getenv("WEBSITE_SITE_NAME") else Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# API ENDPOINTS

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    """Upload and index a PDF file in Weaviate"""
    try:
        if not client:
            return {"status": "error", "message": "Weaviate is not connected"}

        if file is None:
            return {"status": "error", "message": "No file uploaded"}

        save_path = UPLOAD_DIR / file.filename

        with open(save_path, "wb") as f:
            f.write(await file.read())

        text = extract_text_from_pdf(save_path)
        chunks = chunk_text(text)

        # NO MORE SCHEMA WIPE PER UPLOAD
        insert_chunks(client, chunks)

        return {
            "status": "success",
            "message": f"✅ PDF '{file.filename}' processed successfully.",
            "chunks": len(chunks),
        }

    except Exception as e:
        print("❌ Upload error:", e)
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.post("/ask_question")
async def ask_question(query: str = Form(...)):
    """Answer a user question using retrieved PDF context."""
    try:
        if not client:
            return {"error": "Weaviate is not connected"}

        retrieved = search_weaviate(client, query, k=12)
        reranked = rerank_chunks_with_llm(query, retrieved)

        context = "\n\n---\n\n".join(str(x) for x in reranked[:4])

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
                        "Base your answer only on the handbook excerpts provided. "
                        "If the information is unclear, infer carefully but prefer quoting exact text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

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

    return "✅ PDF uploaded." if r.status_code == 200 else f"❌ {r.text}"


def ask_question_ui(question):
    if not question.strip():
        return "⚠️ Please enter a question."

    r = requests.post(f"{API_URL}/ask_question", data={"query": question})
    return r.text if r.status_code == 200 else f"❌ {r.text}"


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

        submit_btn.click(
            ask_question_ui,
            inputs=question_input,
            outputs=answer_output,
        )

# Mount Gradio into FastAPI
gr.mount_gradio_app(app, gradio_app, path="/")

# HEALTH + CLEANUP

@app.get("/health")
def health():
    return {"status": "ok"}


@atexit.register
def close_weaviate():
    if client:
        client.close()
        print("🔒 Weaviate connection closed.")