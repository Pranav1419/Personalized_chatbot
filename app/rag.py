import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")        # text -> 384-number vector
db = chromadb.PersistentClient(path="./chroma_store")      # saved on disk
collection = db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
llm = OpenAI(base_url=os.getenv("OPEN_API_URL"), api_key=os.getenv("OPEN_API_KEY"))



def add_chunks(chunks: list[dict], source: str) -> int:
    texts = [c["text"] for c in chunks]
    collection.add(
        ids=[f"{source}_{c['chunk_id']}" for c in chunks],
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        metadatas=[{"source": source, "page": c["page"]} for c in chunks],
    )
    return len(chunks)


def retrieve(question: str, k: int = 4) -> list[dict]:
    q_vec = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


PROMPT = """Answer using the context below. The context may use different words than the question,
so match by meaning. Give the specific fact and cite the page like [page 3].
If nothing in the context relates to the question, say "I don't know based on the document."

Context:
{context}

Question: {question}"""


def answer(question: str, k: int = 4) -> dict:
    hits = retrieve(question, k)
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    resp = llm.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama3.2"),   # set to your provider's model name
        max_tokens=600,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
    )
    return {"answer": resp.choices[0].message.content, "sources": hits}
