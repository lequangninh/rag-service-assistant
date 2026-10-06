from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .answer import Answer, generate_answer
from .retrieve import Hit, Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "brightclean"
CACHE_DIR = ROOT / ".index"

load_dotenv(ROOT / ".env")


class Assistant:
    def __init__(self, data_dir: Path = DATA_DIR, top_k: int = 5):
        self.retriever = Retriever(data_dir, CACHE_DIR)
        self.top_k = top_k

    def ask(self, question: str) -> tuple[Answer, list[Hit]]:
        hits = self.retriever.search(question, k=self.top_k)
        return generate_answer(question, hits), hits
