import os, shutil, tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ingest import pdf_to_chunks
from app import rag

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        chunks = pdf_to_chunks(tmp.name)
        if not chunks:
            raise HTTPException(422, "No text found (scanned PDF? needs OCR)")
        n = rag.add_chunks(chunks, source=file.filename)
    finally:
        os.remove(tmp.name)
    return {"file": file.filename, "chunks_indexed": n}