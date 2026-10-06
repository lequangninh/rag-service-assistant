"""Hybrid retrieval: BM25 keyword search + dense vector search, fused with RRF.

Keyword search catches exact terms ("lockout", "Zone B"); vector search catches
paraphrases ("what if nobody opens the door?"). Reciprocal Rank Fusion combines
both rankings without having to calibrate their very different score scales.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, load_corpus

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9$%]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its "
    "me my of on or our so than that the their them then there these they this "
    "to was we what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0
        df = Counter(term for d in docs for term in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, counts in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
            for term in query:
                tf = counts.get(term, 0)
                if tf:
                    out[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    similarity: float  # cosine similarity to the query (0..1)
    bm25: float


class Retriever:
    """Builds (or loads a cached) index over a folder of Markdown files."""

    def __init__(self, data_dir: Path, cache_dir: Path):
        from fastembed import TextEmbedding  # heavy import, keep it lazy

        self.chunks = load_corpus(data_dir)
        if not self.chunks:
            raise ValueError(f"No Markdown documents found in {data_dir}")
        self._embedder = TextEmbedding(EMBED_MODEL)
        self.vectors = self._load_or_build_vectors(cache_dir)
        self.bm25 = BM25([tokenize(f"{c.section} {c.text}") for c in self.chunks])

    def _passage(self, chunk: Chunk) -> str:
        return f"{chunk.doc_title}. {chunk.section}. {chunk.text}"

    def _load_or_build_vectors(self, cache_dir: Path) -> np.ndarray:
        fingerprint = hashlib.sha256(
            json.dumps([EMBED_MODEL] + [c.to_dict() for c in self.chunks]).encode()
        ).hexdigest()[:16]
        cache = cache_dir / f"vectors-{fingerprint}.npy"
        if cache.exists():
            return np.load(cache)
        vectors = np.array(list(self._embedder.passage_embed(self._passage(c) for c in self.chunks)))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
        return vectors

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = np.array(list(self._embedder.query_embed(query)))[0]
        q /= np.linalg.norm(q)
        sims = self.vectors @ q
        bm25 = self.bm25.scores(tokenize(query))

        vec_rank = {i: r for r, i in enumerate(np.argsort(-sims))}
        bm_rank = {i: r for r, i in enumerate(np.argsort(-bm25))}
        fused = {
            i: 1 / (RRF_K + vec_rank[i] + 1) + (1 / (RRF_K + bm_rank[i] + 1) if bm25[i] > 0 else 0)
            for i in range(len(self.chunks))
        }
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            Hit(chunk=self.chunks[i], score=fused[i], similarity=float(sims[i]), bm25=float(bm25[i]))
            for i in best
        ]
