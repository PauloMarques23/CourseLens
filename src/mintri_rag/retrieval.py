import os
import re
from dataclasses import dataclass
from typing import Any

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi

from .paths import CHROMA_COLLECTION, CHROMA_DIR, CHUNKS_PATH
from .storage import load_chunks


load_dotenv()


RETRIEVAL_MODES = ("hybrid", "dense", "lexical")
DEFAULT_RETRIEVAL_MODE = "hybrid"
RRF_K = 60

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


@dataclass
class SearchResult:
    chunk: dict[str, Any]
    score: float


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()


def default_retrieval_mode() -> str:
    mode = os.getenv("RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE).strip().lower()
    return mode if mode in RETRIEVAL_MODES else DEFAULT_RETRIEVAL_MODE


def configure_llama_index() -> HuggingFaceEmbedding:
    embed_model = HuggingFaceEmbedding(
        model_name=embedding_model_name(),
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embed_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
    )
    Settings.embed_model = embed_model
    Settings.llm = None
    return embed_model


def _chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _chroma_collection(reset: bool = False):
    client = _chroma_client()
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(CHROMA_COLLECTION)


def _vector_store(reset: bool = False) -> ChromaVectorStore:
    return ChromaVectorStore(chroma_collection=_chroma_collection(reset=reset))


def _node_from_chunk(chunk: dict[str, Any]) -> TextNode:
    return TextNode(
        id_=chunk["id"],
        text=chunk["text"],
        metadata={
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        },
        excluded_embed_metadata_keys=["source", "chunk_id"],
        excluded_llm_metadata_keys=["source", "chunk_id"],
    )


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


class RAGIndex:
    def __init__(self, index: VectorStoreIndex, chunks: list[dict[str, Any]]):
        self.index = index
        self.chunks = chunks
        self._chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
        self._chunk_ids = [chunk["id"] for chunk in chunks]
        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    @classmethod
    def load(cls) -> "RAGIndex":
        configure_llama_index()
        collection = _chroma_collection(reset=False)
        if collection.count() == 0:
            raise FileNotFoundError("The Chroma vector store is empty. Rebuild the index first.")
        chunks = load_chunks(CHUNKS_PATH)
        if not chunks:
            raise FileNotFoundError(
                "chunks.jsonl is missing or empty. Rebuild the index first."
            )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return cls(index=index, chunks=chunks)

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [
            (node.node.node_id, float(node.score or 0.0))
            for node in nodes
            if node.score is not None
        ]

    def _lexical_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = scores.argsort()[::-1][:top_k]
        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str | None = None,
    ) -> list[SearchResult]:
        chosen_mode = (mode or default_retrieval_mode()).lower()
        if chosen_mode not in RETRIEVAL_MODES:
            chosen_mode = DEFAULT_RETRIEVAL_MODE

        if chosen_mode == "dense":
            ranked = self._dense_search(query, top_k)
        elif chosen_mode == "lexical":
            ranked = self._lexical_search(query, top_k)
        else:
            candidate_pool = max(top_k * 4, 20)
            dense = self._dense_search(query, candidate_pool)
            lexical = self._lexical_search(query, candidate_pool)
            ranked = _reciprocal_rank_fusion([dense, lexical])[:top_k]

        return [self._make_result(chunk_id, score) for chunk_id, score in ranked]

    def _make_result(self, chunk_id: str, score: float) -> SearchResult:
        chunk = self._chunk_by_id.get(chunk_id)
        if chunk is None:
            chunk = {
                "id": chunk_id,
                "source": "unknown",
                "chunk_id": "",
                "text": "",
            }
        return SearchResult(chunk=chunk, score=score)


def build_and_save_index(chunks: list[dict[str, Any]]) -> RAGIndex:
    if not chunks:
        raise ValueError("No chunks found. Run ingestion after adding documents.")

    embed_model = configure_llama_index()
    vector_store = _vector_store(reset=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    nodes = [_node_from_chunk(chunk) for chunk in chunks]
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return RAGIndex(index=index, chunks=chunks)
