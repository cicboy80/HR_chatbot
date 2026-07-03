
from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.responses import RedirectResponse
import os
import logging
import requests
import gradio as gr
import re
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

BASE_DIR = Path(__file__).resolve().parent.parent  # goes from app/ -> project root
load_dotenv(BASE_DIR / "api_keys.env")

from app.pdf_utils import extract_text_from_pdf, chunk_text
from app.llm_utils import rerank_chunks_with_llm, client as openai_client
from app.weaviate_utils import connect, insert_chunks, ensure_schema, search_weaviate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hr_chatbot")


# RATE LIMITING (public app: per-IP limits protect API spend)
def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "3/day")
ASK_RATE_LIMIT = os.getenv("ASK_RATE_LIMIT", "20/hour")

limiter = Limiter(key_func=client_ip)

# ENV + CONNECTION
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not WEAVIATE_URL:
    raise ValueError("❌ Missing WEAVIATE_URL in environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in environment variables.")
if not WEAVIATE_API_KEY:
    raise ValueError("❌ Missing WEAVIATE_API_KEY in environment variables.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.weaviate = None
    try:
        app.state.weaviate = connect(WEAVIATE_URL, WEAVIATE_API_KEY)
        ensure_schema(app.state.weaviate)   # create collection once
        logger.info("Connected to Weaviate")
    except Exception:
        logger.exception("Failed to connect to Weaviate")
    yield
    if app.state.weaviate:
        app.state.weaviate.close()
        logger.info("Weaviate connection closed")


app = FastAPI(title="HR Q&A Bot", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def get_weaviate(request: Request):
    wv = getattr(request.app.state, "weaviate", None)
    if not wv:
        raise HTTPException(status_code=503, detail="Weaviate is not connected")
    return wv

# UPLOAD DIRECTORY
UPLOAD_DIR = Path("/home/uploads") if os.getenv("WEBSITE_SITE_NAME") else Path("uploads")
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # e.g. /home not writable for the container user — fall back to a local dir
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# UPLOAD LIMITS (bound the worst-case cost of a single upload)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
MAX_CHUNKS_PER_UPLOAD = int(os.getenv("MAX_CHUNKS_PER_UPLOAD", "500"))


async def save_upload(file: UploadFile, save_path: Path) -> None:
    """Stream the upload to disk, enforcing the PDF magic bytes and size cap."""
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    header = await file.read(5)
    if header != b"%PDF-":
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    size = len(header)
    try:
        with open(save_path, "wb") as f:
            f.write(header)
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.",
                    )
                f.write(chunk)
    except HTTPException:
        save_path.unlink(missing_ok=True)
        raise


def index_pdf(save_path: Path, safe_name: str, wv) -> dict:
    """Extract, chunk, and insert a saved PDF (blocking; run in a threadpool)."""
    try:
        text = extract_text_from_pdf(save_path, max_pages=MAX_PDF_PAGES)
    except ValueError as err:
        # pdf_utils uses ValueError for "no extractable text" / too many pages
        raise HTTPException(status_code=400, detail=str(err))

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in this PDF.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="PDF produced 0 chunks after processing.")

    truncated = len(chunks) > MAX_CHUNKS_PER_UPLOAD
    chunks = chunks[:MAX_CHUNKS_PER_UPLOAD]

    # NO MORE SCHEMA WIPE PER UPLOAD
    result = insert_chunks(wv, chunks, safe_name)

    message = f"✅ PDF '{safe_name}' processed successfully."
    if truncated:
        message += f" (Indexed the first {MAX_CHUNKS_PER_UPLOAD} sections only.)"

    return {
        "status": "success",
        "message": message,
        "chunks": len(chunks),
        "inserted": result["inserted"],
        "skipped_existing": result["skipped_existing"],
        "unique_in_upload": result["unique_in_upload"],
    }


# API ENDPOINTS
@app.post("/upload_pdf")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_pdf(request: Request, file: UploadFile):
    """Upload and index a PDF file in Weaviate"""
    wv = get_weaviate(request)

    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    safe_name = Path(file.filename or "").name
    if (
        not safe_name
        or safe_name.startswith(".")
        or not safe_name.lower().endswith(".pdf")
    ):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = UPLOAD_DIR / f"{uuid4().hex[:8]}-{safe_name}"
    await save_upload(file, save_path)

    try:
        return await run_in_threadpool(index_pdf, save_path, safe_name, wv)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail="Internal error while processing the PDF.")
    finally:
        # the PDF is only needed during ingestion; don't accumulate uploads
        save_path.unlink(missing_ok=True)


@app.post("/ask_question")
@limiter.limit(ASK_RATE_LIMIT)
def ask_question(request: Request, query: str = Form(...)):
    """Answer a user question using retrieved PDF context.

    Plain `def` on purpose: FastAPI runs it in the threadpool, so the
    blocking OpenAI/Weaviate calls don't stall the event loop.
    """
    try:
        wv = get_weaviate(request)

        retrieved = search_weaviate(wv, query, k=20)
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

        raw = response.choices[0].message.content.strip()
        logger.debug("Raw LLM output: %r", raw)

        # --- Strip straight + curly quotes from the start/end ---
        answer = re.sub(r'^[\"“”‘’]+|[\"“”‘’]+$', '', raw).strip()

        return {
            "answer": answer,
            "retrieved_docs": retrieved,
            "reranked_docs": top_docs,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Question answering failed")
        raise HTTPException(status_code=500, detail="Internal error while answering the question.")

# ✅ GRADIO UI (LOCAL API)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def forwarded_ip_headers(request: gr.Request | None) -> dict:
    """Forward the real browser IP on the Gradio -> API self-call so rate
    limits apply per visitor instead of to 127.0.0.1."""
    if request is None:
        return {}
    ip = request.headers.get("x-forwarded-for") or (
        request.client.host if request.client else None
    )
    return {"X-Forwarded-For": ip} if ip else {}


def upload_pdf_ui(pdf_file, request: gr.Request):
    if pdf_file is None:
        return "Please upload a PDF."

    with open(pdf_file.name, "rb") as f:
        files = {"file": f}
        r = requests.post(
            f"{API_URL}/upload_pdf",
            files=files,
            headers=forwarded_ip_headers(request),
            timeout=300,
        )

    if r.status_code != 200:
        return f"❌ {r.text}"

    data = r.json()
    if data.get("status") == "success":
        return data.get("message", "✅ PDF processed.")
    return f"❌ {data.get('message', 'Upload failed')}"

def ask_question_ui(question, request: gr.Request):
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""

    r = requests.post(
        f"{API_URL}/ask_question",
        data={"query": question},
        headers=forwarded_ip_headers(request),
        timeout=120,
    )

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

# HEALTH
@app.get("/health")
def health(request: Request):
    wv = getattr(request.app.state, "weaviate", None)
    connected = False
    if wv:
        try:
            connected = wv.is_ready()
        except Exception:
            connected = False
    return {"status": "ok", "weaviate": "connected" if connected else "disconnected"}
