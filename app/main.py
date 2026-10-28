from fastapi import FastAPI
from app.router import Upload, Ask
from app import rag

app = FastAPI(title="PDF RAG API")
app.include_router(Upload.router)
app.include_router(Ask.router)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_stored": rag.collection.count()}