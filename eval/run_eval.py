"""Evaluate the assistant on eval/questions.jsonl.

Retrieval only (free, no API calls):
    uv run python eval/run_eval.py

End to end with Claude (uses your API key, about 24 short requests):
    uv run python eval/run_eval.py --llm

Answerable questions ("expect" set) check that the right chunk is retrieved in
the top 3, and with --llm that the answer contains the key fact. Unanswerable
questions ("expect": null) check that the assistant says it doesn't know.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.answer import generate_answer, too_weak  # noqa: E402
from rag.pipeline import Assistant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also generate answers with Claude")
    args = parser.parse_args()

    cases = [json.loads(line) for line in Path(__file__).with_name("questions.jsonl").read_text().splitlines() if line]
    assistant = Assistant()
    hit3 = answerable = 0
    llm_ok = 0
    rows = []
    for case in cases:
        hits = assistant.retriever.search(case["q"], k=5)
        top_sim = max(h.similarity for h in hits)
        ids = [h.chunk.id for h in hits]
        if case["expect"]:
            answerable += 1
            ok = case["expect"] in ids[:3]
            hit3 += ok
            status = "hit@3" if ok else f"MISS (got {ids[0]})"
        else:
            status = "gated" if too_weak(hits) else "to model"
        line = f"{top_sim:.2f}  {status:<40} {case['q']}"
        if args.llm:
            answer = generate_answer(case["q"], hits)
            if case["expect"]:
                good = answer.answered and all(k.lower() in answer.text.lower() for k in case["keywords"])
            else:
                good = not answer.answered
            llm_ok += good
            line += f"\n      {'OK ' if good else 'BAD'} {answer.text[:140]}"
        rows.append(line)

    print("\n".join(rows))
    print(f"\nRetrieval hit@3: {hit3}/{answerable}")
    if args.llm:
        print(f"End-to-end correct: {llm_ok}/{len(cases)}")


if __name__ == "__main__":
    main()
