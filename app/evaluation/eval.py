import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from app import ingest, rag

HERE = Path(__file__).parent
K_VALUES = (1, 3, 5)


# ---------- helpers ----------
def norm(s: str) -> str:
    return s.lower().replace("\u2019", "'")


def keywords_ok(answer: str, keywords: list) -> bool:
    """Every entry must appear in the answer; an entry may be a list of acceptable alternatives."""
    a = norm(answer)
    for entry in keywords:
        alts = [entry] if isinstance(entry, str) else entry
        if not any(norm(x) in a for x in alts):
            return False
    return True


def chunk_settings() -> dict:
    params = inspect.signature(ingest.chunk).parameters
    return {k: params[k].default for k in ("size", "overlap") if k in params}


def reindex(pdf_path: str) -> str:
    """Delete this PDF's old chunks and index it again with the current ingest settings."""
    source = os.path.basename(pdf_path)
    rag.collection.delete(where={"source": source})
    n = rag.add_chunks(ingest.pdf_to_chunks(pdf_path), source=source)
    print(f"Indexed {n} chunks from {source} with settings {chunk_settings()}")
    return source


# ---------- retrieval evaluation (no LLM, fast) ----------
def run_retrieval(questions: list, source, k_ctx: int = 4):
    """expected_page can be one page (int) or several (list). rank = first position of ANY needed page.
    all_found = every needed page is inside the top k_ctx chunks (what the LLM actually receives)."""
    rows = []
    for q in questions:
        if q["expected_page"] is None:
            continue
        needed = q["expected_page"] if isinstance(q["expected_page"], list) else [q["expected_page"]]
        hits = rag.retrieve(q["question"], k=max(K_VALUES))

        def good(h):
            return source is None or h["source"] == source

        rank = next((i for i, h in enumerate(hits, 1) if h["page"] in needed and good(h)), None)
        found = {h["page"] for h in hits[:k_ctx] if good(h)}
        rows.append({
            "question": q["question"],
            "expected_page": q["expected_page"],
            "rank": rank,
            "all_found": set(needed) <= found,
            "retrieved_pages": [h["page"] for h in hits],
            "top_snippet": hits[0]["text"][:90] if hits else "",
        })
    return rows


def retrieval_metrics(rows: list) -> dict:
    n = len(rows)
    m = {f"hit@{k}": sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n for k in K_VALUES}
    m["mrr"] = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    return m


# ---------- answer evaluation (calls the LLM, slow) ----------
def run_answers(questions: list, k: int) -> list:
    rows = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        try:
            text = rag.answer(q["question"], k)["answer"]
            correct = keywords_ok(text, q["answer_contains"])
        except Exception as e:  # keep going if one call fails
            text, correct = f"ERROR {type(e).__name__}: {e}", False
        rows.append({
            "question": q["question"],
            "unanswerable": q["expected_page"] is None,
            "correct": correct,
            "seconds": round(time.time() - start, 1),
            "answer": text,
        })
        print(f"  [{i}/{len(questions)}] {'PASS' if correct else 'FAIL'}  {q['question']}", flush=True)
    return rows


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="PDF to (re)index before evaluating")
    ap.add_argument("--answers", action="store_true", help="also test LLM answers")
    ap.add_argument("--k", type=int, default=4, help="k used for answer generation (your API default is 4)")
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    source = reindex(args.pdf) if args.pdf else None

    if rag.collection.count() == 0:
        sys.exit("The index is empty. Run with --pdf path/to/file.pdf, or run from the folder that has chroma_store.")

    rows = run_retrieval(questions, source, args.k)
    metrics = retrieval_metrics(rows)

    print(f"\nRETRIEVAL ({len(rows)} questions)")
    print(f"{'#':>2}  {'page':>7}  {'rank':>4}  {'retrieved pages':<18} question")
    for i, r in enumerate(rows, 1):
        rank = r["rank"] if r["rank"] else "MISS"
        print(f"{i:>2}  {str(r['expected_page']):>7}  {rank:>4}  {str(r['retrieved_pages']):<18} {r['question']}")
    print()
    for k in K_VALUES:
        print(f"hit@{k}: {metrics[f'hit@{k}']:.0%}")
    print(f"MRR:   {metrics['mrr']:.2f}")

    multi = [r for r in rows if isinstance(r["expected_page"], list)]
    if multi:
        print(f"Multi-page questions with ALL needed pages in the top {args.k}: {sum(r['all_found'] for r in multi)}/{len(multi)}")
        for r in multi:
            if not r["all_found"]:
                print(f"- missing a page: {r['question']}\n    needed {r['expected_page']}, got {r['retrieved_pages'][:args.k]}")

    misses = [r for r in rows if not r["rank"]]
    if misses:
        print("\nMISSES (what the retriever put first instead):")
        for r in misses:
            print(f"- {r['question']}\n    expected page {r['expected_page']}, got {r['retrieved_pages']}\n    top chunk: {r['top_snippet']!r}")

    answer_rows, answer_acc = [], None
    if args.answers:
        print(f"\nANSWERS (k={args.k}, this can take a while with a local model)")
        answer_rows = run_answers(questions, args.k)
        answer_acc = sum(r["correct"] for r in answer_rows) / len(answer_rows)
        refusals = [r for r in answer_rows if r["unanswerable"]]
        print(f"\nAnswer accuracy: {answer_acc:.0%}")
        if refusals:
            print(f"Correctly said 'I don't know' on unanswerable questions: {sum(r['correct'] for r in refusals)}/{len(refusals)}")
        for r in answer_rows:
            if not r["correct"]:
                print(f"- FAIL: {r['question']}\n    answer: {r['answer'][:200]!r}")

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question_set": Path(args.questions).name,
        "chunking": chunk_settings(),
        "answer_k": args.k if args.answers else None,
        "llm": os.getenv("LLM_MODEL") if args.answers else None,
        "metrics": {**metrics, **({"answer_accuracy": answer_acc} if answer_acc is not None else {})},
    }
    (HERE / "results.json").write_text(
        json.dumps({**summary, "retrieval": rows, "answers": answer_rows}, indent=2), encoding="utf-8")
    with open(HERE / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\nSaved evaluation/results.json and added a line to evaluation/history.jsonl")


if __name__ == "__main__":
    main()