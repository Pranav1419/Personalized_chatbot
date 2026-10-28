from app import rag

QUESTION = "How long does a new hire's onboarding mentor stay assigned?"
FACT = "60 days"  # the fact the answer should contain

hits = rag.retrieve(QUESTION, 4)

print("=== CHUNKS SENT TO THE LLM (k=4) ===")
for i, h in enumerate(hits, 1):
    has_fact = "YES" if FACT in h["text"] else "no"
    print(f"\n#{i} page {h['page']}  distance {h['distance']:.3f}  contains '{FACT}': {has_fact}")
    print(h["text"])

print("\n=== ANSWER WITH k=4 ===")
print(rag.answer(QUESTION, 4)["answer"])

print("\n=== ANSWER WITH k=2 ===")
print(rag.answer(QUESTION, 2)["answer"])