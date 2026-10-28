from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app import rag
 
router = APIRouter(prefix="/ask", tags=["Ask"])
 
 
class Question(BaseModel):
    question: str
    k: int = 4
 
 
@router.post("")
def ask_question(q: Question):
    if rag.collection.count() == 0:
        raise HTTPException(400, "Upload a PDF first")
    return rag.answer(q.question, q.k)
 