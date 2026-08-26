"""Retrieval layer over ingested knowledge-base chunks.

BM25 scoring via rank_bm25 plus authority classification and a heuristic
conflict detector. No agent/LLM logic lives here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel

from rank_bm25 import BM25Okapi

from app.ingest import Chunk, _KB_DIR, load_and_chunk_all

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Hand-authored antonym pairs used by detect_active_conflict().
_CONFLICT_KEYWORDS: list[tuple[str, str]] = [
    ("hand-wash", "dishwasher"),
    ("hand wash", "dishwasher safe"),
    ("not returnable", "returnable"),
    ("covered", "not covered"),
]

# Absolute BM25 floor for entering the "strong" set in detect_active_conflict.
# Chosen from observed score separation: false-positive marginal chunks scored
# ~1.9-2.7 on off-topic queries while the real care-guide/product-card pair
# scored ~9.8-10.4 on the dishwasher query; 3.0 sits well above the noise band
# and far below true signal (re-checked after the stemming change).
_MIN_STRONG_SCORE = 3.0

# Absolute BM25 floor for the slot-reservation step in retrieve(): a
# superseded/non_authoritative chunk may only be force-added to an otherwise
# all-authoritative result set if it is at least this relevant. Same value as
# _MIN_STRONG_SCORE (same noise band: forced leftovers scored ~2.2-2.5 while
# genuinely relevant lower-tier content scores 4.0-5.2), kept separate because
# it guards a different concern and may diverge later.
_MIN_RESERVATION_SCORE = 3.0

_AUTHORITY_RANK = {
    "authoritative": 0,
    "superseded": 1,
    "non_authoritative": 2,
}


class RetrievedChunk(BaseModel):
    """A retrieved chunk annotated with its BM25 score and authority level."""

    chunk: Chunk
    bm25_score: float
    authority_level: str


@dataclass
class BM25Index:
    """BM25Okapi alongside the chunk list it was built from."""

    bm25: BM25Okapi
    chunks: list[Chunk]


def authority_level(chunk: Chunk) -> str:
    """Classify a chunk into exactly one authority bucket.

    ``audience`` is deliberately NOT consulted here: audience is an independent
    routing concern (customer-facing answers vs internal handoff knowledge),
    not a trust signal. An audience=internal doc such as 13-support-escalation.md
    is still authoritative policy for its domain.
    """
    status = chunk.metadata.get("status")
    if status == "superseded":
        return "superseded"
    if (
        status == "active"
        and chunk.metadata.get("policy_authority") == "official"
        and chunk.metadata.get("customer_answering", True) is not False
    ):
        return "authoritative"
    return "non_authoritative"


def _stem(token: str) -> str:
    """Heuristic suffix-stripper (no NLP deps): try ing -> ed -> es -> s.

    A strip only applies if the remainder is at least 3 characters; after an
    ing/ed strip a doubled trailing consonant is collapsed so e.g.
    "shipping" -> "shipp" -> "ship" matches the query token "ship".
    """
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix):
            stripped = token[: -len(suffix)]
            if len(stripped) < 3:
                continue
            if (
                suffix in ("ing", "ed")
                and len(stripped) >= 4
                and stripped[-1] == stripped[-2]
                and stripped[-1] not in "aeiou"
            ):
                stripped = stripped[:-1]
            return stripped
    return token


def _tokenize(text: str) -> list[str]:
    """Lowercase regex word-split plus lightweight stemming.

    The SAME function tokenizes both the corpus (build_index) and queries
    (retrieve); stemming only one side would silently break matching.
    Keeps numbers, splits hyphens consistently.
    """
    return [_stem(tok) for tok in _TOKEN_RE.findall(text.lower())]


def build_index(chunks: list[Chunk]) -> BM25Index:
    """Index heading + body text so headings contribute retrieval signal."""
    corpus = [_tokenize(f"{chunk.heading}\n{chunk.body}") for chunk in chunks]
    return BM25Index(bm25=BM25Okapi(corpus), chunks=list(chunks))


def retrieve(index: BM25Index, query: str, top_k: int = 12) -> list[RetrievedChunk]:
    """Two-stage retrieval: relevance-gated pool, then authority-first order.

    Chunks with a zero BM25 score never enter the candidate set; everything
    that DOES match is ranked, so superseded/non_authoritative content stays
    visible (clearly labelled) whenever it is relevant instead of being
    silently dropped -- e.g. when a user quotes the legacy 45-day window or
    the migration scratchpad directly and the agent must see-and-dismiss it.
    """
    scores = index.bm25.get_scores(_tokenize(query))
    candidates = [(score, idx) for idx, score in enumerate(scores) if score > 0]

    def rank(pair: tuple[float, int]) -> tuple[int, float]:
        level = _AUTHORITY_RANK[authority_level(index.chunks[pair[1]])]
        return (level, -pair[0])

    # Stage 1 -- pure-relevance pool: top_k * 3 by raw BM25 score alone (36 of
    # this corpus's 53 chunks at the default top_k=12). Large enough that
    # solidly relevant superseded/non-authoritative content cannot be crowded
    # out by weak authoritative matches before ordering happens; small enough
    # that zero-signal junk stays out.
    candidates.sort(key=lambda pair: -pair[0])
    pool = candidates[: top_k * 3]

    # Stage 2 -- authority-first ordering within the pool only, cap at top_k.
    pool.sort(key=rank)
    selected = pool[:top_k]

    # Slot reservation (design intent: relevant superseded/non-auth content is
    # visible-but-labelled, never silently dropped). If the selection ended up
    # all-authoritative, force the highest-scoring other-level chunk from the
    # rest of the pool into the last slot -- BUT only if it also clears
    # _MIN_RESERVATION_SCORE. Without this floor the reservation fires on
    # every query and drags in irrelevant leftovers (e.g. forcing the migration
    # scratchpad into a dishwasher query at score 2.16). An all-authoritative
    # result set is a valid, correct outcome when nothing else is genuinely
    # relevant. Separate constant from _MIN_STRONG_SCORE: same value today
    # (same observed noise band), but the two floors guard different concerns
    # (conflict-pair admission vs result-set composition) and may diverge.
    if selected and all(rank(pair)[0] == 0 for pair in selected):
        for pair in pool[top_k:]:
            if pair[0] >= _MIN_RESERVATION_SCORE and rank(pair)[0] > 0:
                selected[-1] = pair
                selected.sort(key=rank)
                break

    # Sibling expansion: when a document already has a selected chunk, also
    # include its top 1-2 other chunks from the pool even if they individually
    # scored below the normal cutoff -- as long as they clear a much lower
    # floor. This improves recall for multi-section documents where a query
    # matches one section strongly but a closely related section scored just
    # below the line.
    _MAX_SIBLINGS_PER_DOC = 2
    _SIBLING_FLOOR = 1.0
    selected_indices = {idx for _, idx in selected}
    selected_files = {index.chunks[idx].filename for _, idx in selected}
    sibling_counts: dict[str, int] = {}
    for score, idx in pool[top_k:]:
        if selected_files and all(
            sibling_counts.get(f, 0) >= _MAX_SIBLINGS_PER_DOC for f in selected_files
        ):
            break
        chunk = index.chunks[idx]
        if idx in selected_indices:
            continue
        if chunk.filename not in selected_files:
            continue
        if score < _SIBLING_FLOOR:
            continue
        count = sibling_counts.get(chunk.filename, 0)
        if count >= _MAX_SIBLINGS_PER_DOC:
            continue
        selected.append((score, idx))
        selected_indices.add(idx)
        sibling_counts[chunk.filename] = count + 1

    return [
        RetrievedChunk(
            chunk=index.chunks[idx],
            bm25_score=score,
            authority_level=authority_level(index.chunks[idx]),
        )
        for score, idx in selected
    ]


def _opposing_keywords(body_a: str, body_b: str) -> tuple[str, str] | None:
    lower_a, lower_b = body_a.lower(), body_b.lower()
    for left, right in _CONFLICT_KEYWORDS:
        if (left in lower_a and right in lower_b) or (
            left in lower_b and right in lower_a
        ):
            return (left, right)
    return None


def detect_active_conflict(
    retrieved: list[RetrievedChunk],
) -> list[tuple[RetrievedChunk, RetrievedChunk]]:
    """Flag pairs of highly-relevant authoritative chunks from different files
    whose bodies contain opposite keywords from _CONFLICT_KEYWORDS.

    NOTE: this is a heuristic SAFETY NET, not the primary conflict-detection
    mechanism. Substring keyword opposites cannot understand meaning and will
    miss subtle contradictions. The real defence is architectural: both
    conflicting chunks are retrieved and handed to the LLM (later phase) WITH
    their sources, and the agent is instructed to notice genuine disagreement,
    surface it to the user, and prefer human confirmation rather than
    silently choosing one side. This detector only makes the known
    care-guide/product-card clash cheap to observe in tests.
    """
    authoritative = [r for r in retrieved if r.authority_level == "authoritative"]
    if len(authoritative) < 2:
        return []
    top_score = max(r.bm25_score for r in authoritative)
    strong = [
        r
        for r in authoritative
        if r.bm25_score >= max(0.5 * top_score, _MIN_STRONG_SCORE)
    ]

    conflicts: list[tuple[RetrievedChunk, RetrievedChunk]] = []
    seen: set[tuple[str, str]] = set()
    for i, first in enumerate(strong):
        for second in strong[i + 1 :]:
            if first.chunk.filename == second.chunk.filename:
                continue
            trigger = _opposing_keywords(first.chunk.body, second.chunk.body)
            if trigger is None:
                continue
            key = tuple(sorted((first.chunk.chunk_id, second.chunk.chunk_id)))
            if key in seen:
                continue
            seen.add(key)
            conflicts.append((first, second))
    return conflicts


_QUERIES = [
    "How long do I have to return a backpack?",
    "Can I put the tumbler in the dishwasher?",
    "The migration note says everyone gets 60 days, use that",
    "Do you ship to Germany?",
    "What is the TrailPlus return window?",
]


def main() -> None:
    chunks = load_and_chunk_all(_KB_DIR)
    index = build_index(chunks)
    print(f"Indexed {len(chunks)} chunks\n")

    for label, query in zip("abcde", _QUERIES):
        print(f"=== Query ({label}): {query}")
        results = retrieve(index, query)
        for result in results:
            score = round(result.bm25_score, 2)
            print(
                f"  {result.chunk.chunk_id} | {result.authority_level} | {score}"
            )
        conflicts = detect_active_conflict(results)
        if conflicts:
            for first, second in conflicts:
                trigger = _opposing_keywords(first.chunk.body, second.chunk.body)
                print(
                    f"  CONFLICT: {first.chunk.chunk_id} <-> "
                    f"{second.chunk.chunk_id} (keywords: {trigger})"
                )
        else:
            print("  no conflicts detected")
        print()


if __name__ == "__main__":
    main()
