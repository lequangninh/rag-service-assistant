from types import SimpleNamespace

from rag.answer import NOT_FOUND, generate_answer
from rag.chunking import Chunk, chunk_markdown
from rag.retrieve import BM25, Hit, tokenize

DOC = """# Acme – Pricing

Intro paragraph.

## Prices

A standard visit costs $100.

## Payment

We accept cards only.
"""


def test_chunks_follow_sections():
    chunks = chunk_markdown("pricing.md", DOC)
    assert [c.section for c in chunks] == ["Overview", "Prices", "Payment"]
    assert chunks[1].id == "pricing:prices"
    assert chunks[1].label == "pricing.md › Prices"
    assert chunks[0].doc_title == "Acme – Pricing"


def test_bm25_prefers_matching_chunk():
    docs = [tokenize("we accept cards only"), tokenize("a standard visit costs $100")]
    scores = BM25(docs).scores(tokenize("do you accept cards?"))
    assert scores[0] > scores[1]


def _hit(text: str, label: str, similarity: float = 0.8) -> Hit:
    source, section = label.split(" › ")
    chunk = Chunk(id=label, source=source, doc_title="T", section=section, text=text)
    return Hit(chunk=chunk, score=1.0, similarity=similarity, bm25=1.0)


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_weak_retrieval_skips_model_call():
    client = FakeClient(response=None)
    answer = generate_answer("Who is the CEO?", [_hit("x", "a.md › A", similarity=0.3)], client=client)
    assert not answer.answered and answer.skipped_model_call
    assert client.calls == []


def test_no_shared_keyword_skips_model_call():
    # Nonsense input can pass the similarity gate; zero BM25 overlap must still stop it.
    client = FakeClient(response=None)
    hit = _hit("x", "a.md › A", similarity=0.6)
    hit.bm25 = 0.0
    answer = generate_answer("asdkjh qwe zzz", [hit], client=client)
    assert not answer.answered and answer.skipped_model_call
    assert client.calls == []


def test_citations_are_numbered_and_deduplicated():
    cit = SimpleNamespace(document_index=1, cited_text="We accept cards only. ")
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="We only take cards.", citations=[cit]),
            SimpleNamespace(type="text", text=" Cash is not accepted.", citations=[cit]),
        ],
    )
    hits = [_hit("A standard visit costs $100.", "pricing.md › Prices"), _hit("We accept cards only.", "pricing.md › Payment")]
    client = FakeClient(response)
    answer = generate_answer("Can I pay cash?", hits, client=client)

    assert answer.text == "We only take cards. [1] Cash is not accepted. [1]"
    assert len(answer.citations) == 1
    assert answer.citations[0].label == "pricing.md › Payment"
    sent = client.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in sent] == ["document", "document", "text"]
    assert all(b["citations"] == {"enabled": True} for b in sent[:2])


def test_not_found_reply_is_marked_unanswered():
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        content=[SimpleNamespace(type="text", text=f"{NOT_FOUND} Please contact support.", citations=None)],
    )
    answer = generate_answer("Do you mow lawns?", [_hit("x", "a.md › A")], client=FakeClient(response))
    assert not answer.answered
