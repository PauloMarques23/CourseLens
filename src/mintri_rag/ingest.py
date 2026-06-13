from .chunking import chunk_text
from .loaders import discover_documents, load_document
from .paths import CHUNKS_PATH, RAW_DATA_DIR
from .retrieval import build_and_save_index
from .storage import save_jsonl


def ingest_corpus(chunk_size: int = 220, overlap: int = 45) -> dict:
    documents = discover_documents(RAW_DATA_DIR)
    chunks = []

    for document_path in documents:
        text = load_document(document_path)
        relative_source = document_path.relative_to(RAW_DATA_DIR).as_posix()
        for chunk_id, chunk in enumerate(chunk_text(text, chunk_size, overlap), start=1):
            chunks.append(
                {
                    "id": f"{relative_source}::chunk-{chunk_id}",
                    "source": relative_source,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    save_jsonl(CHUNKS_PATH, chunks)
    if chunks:
        build_and_save_index(chunks)

    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_path": str(CHUNKS_PATH),
    }
