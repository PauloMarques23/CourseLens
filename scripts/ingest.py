import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.mintri_rag.ingest import ingest_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents and build the RAG index.")
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=45)
    args = parser.parse_args()

    report = ingest_corpus(chunk_size=args.chunk_size, overlap=args.overlap)
    print(
        f"Indexed {report['chunk_count']} chunks from {report['document_count']} documents."
    )


if __name__ == "__main__":
    main()

