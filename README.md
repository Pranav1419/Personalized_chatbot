# PDF Question Answering with RAG

Upload a PDF, ask questions in plain English, and get answers with page citations. Built with **FastAPI**, **ChromaDB** and a local **Llama 3.2** model, so it runs entirely on your own machine with no paid API.

![How the RAG pipeline works]
![pipeline image](<Screenshot 2026-10-28 140410.png>)

## What is this, in plain English?

ChatGPT-style models are like a very good autocomplete trained on the internet. They have never seen your private documents, so if you ask about your company's policy PDF, they can only guess, and the guess can sound confident and still be wrong.

**RAG (Retrieval-Augmented Generation)** fixes this like an open-book exam. Instead of expecting the AI to remember everything, the system first looks up the most relevant pages of your document, then asks the AI to answer using only those pages. If the answer isn't there, it says so.

## How it works

**Part 1: Prepare the PDF (once, on upload)**

1. **Parse**: extract the text from each page (`pypdf`).
2. **Clean**: fix words split across lines, collapse extra whitespace.
3. **Chunk**: cut the text into pieces of 500 characters with 100 characters of overlap, so a sentence isn't cut off from its context.
4. **Embed**: turn each chunk into a vector that captures its meaning (`all-MiniLM-L6-v2`).
5. **Store**: save the chunks, vectors and page numbers in ChromaDB.

**Part 2: Answer a question (every time)**

1. Embed the question the same way.
2. Retrieve the `k` closest chunks (default `k=4`) from ChromaDB.
3. Build a prompt: the retrieved chunks plus the question, with instructions to answer from them or say "I don't know".
4. Send it to the LLM and return the answer together with the source chunks and page numbers.

## Tech stack

| Part | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| PDF parsing | pypdf |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB (persisted to `./chroma_store`) |
| LLM | Llama 3.2 via Ollama (any OpenAI-compatible endpoint works) |

## Quick start

**Prerequisites:** Python 3.11+ and [Ollama](https://ollama.com).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the model
ollama pull llama3.2

# 4. Configure environment variables
copy .env.example .env         # Windows
# cp .env.example .env         # Mac / Linux

# 5. Run the API
uvicorn app.main:app --reload --reload-dir app
```

Open **http://127.0.0.1:8000/docs** for the interactive API page. Upload a PDF at `/upload`, then ask questions at `/ask`.

### Using a different LLM

Only the three values in `.env` change. For example, Gemini's OpenAI-compatible endpoint:

```
OPEN_API_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPEN_API_KEY=your_key
LLM_MODEL=gemini-2.5-flash
```

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/upload` | Upload a PDF (multipart form, field name `file`). Returns how many chunks were indexed. |
| `POST` | `/ask` | Ask a question. Body: `{"question": "...", "k": 4}` |
| `GET` | `/health` | Shows the service is up and how many chunks are stored. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/upload -F "file=@acme_corp_policies.pdf"

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do full-time employees earn per year?"}'
```

The response has this shape (values abridged):

```json
{
  "answer": "Full-time employees accrue 20 days of PTO per year [page 5].",
  "sources": [
    {"text": "5. Annual Paid Time Off (PTO) Framework ...", "source": "acme_corp_policies.pdf", "page": 5, "distance": 0.31}
  ]
}
```

## Project structure

```
.
├── app/
│   ├── main.py             # creates the FastAPI app, /health, includes the routers
│   ├── ingest.py           # parse, clean, chunk
│   ├── rag.py              # embed, vector DB, retrieve, prompt, LLM
│   ├── router/
│   │   ├── Upload.py       # POST /upload
│   │   └── Ask.py          # POST /ask
│   └── evaluation/
│       ├── eval.py         # evaluation script
│       ├── questions.json       # 24 test questions
│       └── questions_hard.json  # 10 harder questions
├── debug_q.py              # inspect what was retrieved for one question
├── acme_corp_policies.pdf  # synthetic 50-page policy manual used for testing
├── requirements.txt
└── .env.example
```

## Evaluation: how I know it works

A demo that works on two or three questions proves very little, so I wrote test questions with known answers and scored the system. The script measures the two jobs separately:

- **Retrieval:** is the right page in the top results? It reports hit@1, hit@3, hit@5 and MRR (the average of 1/rank, so 1.0 means the right page is always first).
- **Answers:** does the LLM's answer contain the expected facts? Questions that have no answer in the document check that the system says "I don't know" instead of inventing one.

Run it with the server stopped:

```bash
python -m app.evaluation.eval --pdf acme_corp_policies.pdf --answers
python -m app.evaluation.eval --questions app/evaluation/questions_hard.json --answers
```

Each run appends its settings and scores to `app/evaluation/history.jsonl`, so changes can be compared.

### Results (chunk size 500, overlap 100, k = 4)

| Question set | Retrieval: right page first | Right page in top 3 | MRR | Answers correct | "I don't know" on unanswerable |
|---|---|---|---|---|---|
| Set 1: 24 questions (22 answerable) | 21 / 22 | 22 / 22 | 0.98 | 23 / 24, then 24 / 24 after the prompt fix | 2 / 2 |
| Set 2: 10 harder questions (8 answerable) | 7 / 8 | 7 / 8 | 0.88 | 9 / 10 | 2 / 2 |

Set 2 contains typos, questions that need two pages, scenario questions and trap questions whose topic sounds covered but isn't. All needed pages were found for both multi-page questions.

### What failed and what I learned

**1. Retrieval was right, but the answer was wrong.** For "How long does a new hire's onboarding mentor stay assigned?" the correct page was ranked first, yet the model answered "I don't know". The document says mentors are matched "for their initial 60 days", and the question said "stay assigned". Sending fewer chunks didn't help, so the problem was the prompt: it told a small model to refuse unless the answer was clearly there. I reworded it to tell the model that the context may use different words, and to refuse only if nothing related is present. Answers went from 23/24 to 24/24, and the unanswerable questions were still refused.

**2. The search never found the right page.** For "Is it okay to save a copy of a production database on my laptop?" the answer is on page 6 ("production databases may never be downloaded locally"), but it wasn't in the top 5. The question says "save... on my laptop" and the document says "downloaded locally", so searching by meaning alone ranked pages about hardware and devices higher. In this case the model's "I don't know" was the correct behaviour for the chunks it received, so the fix belongs in retrieval, not the prompt.

The takeaway: when something fails, first find out whether the search or the model failed, then fix that part.

### Caveats

- The test is one clean, synthetic document and 34 questions, so these numbers show the approach works, not that it is production-ready.
- Set 1 was used while tuning the prompt, and the failures in Set 2 have now been seen. Treat both as development sets; a fresh set of unseen questions is needed for a final number.
- Answer checking is a keyword match, so a correct answer worded unusually can be marked wrong.

## Known limitations

- Scanned PDFs (images with no text layer) are not supported; they would need OCR.
- Every uploaded PDF goes into one shared collection.
- Every page of the test document ends with the same boilerplate text, so many chunks are near-duplicates that compete for the top results.
- No authentication.

## Roadmap

- [ ] Remove repeated boilerplate before chunking and measure the change
- [ ] Hybrid search (keyword + vector) to fix the page-6 miss
- [ ] Compare smaller chunk sizes
- [ ] Docker setup
- [ ] Document management endpoints (list and delete uploaded PDFs)
- [ ] Simple web UI
- [ ] A fresh set of unseen test questions and a second, messier PDF

## Author

Built by Bala Pranav