import json
from datasets import load_dataset

langs = ["te", "hi", "bn"]

for lang in langs:
    output_path = f"miracl_{lang}_corpus.jsonl"

    # # ✅ Correct way to load MIRACL
    # miracl_corpus = load_dataset(
    #     "miracl/miracl-corpus",
    #     lang,
    #     split="train"
    # )

    corpus = load_dataset("miracl/miracl-corpus", lang)

    # Write JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            record = {
                "_id": doc["docid"],
                "title": doc.get("title", "") or "",
                "text": doc["text"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved corpus to: {output_path}")
