from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
REPORTS_DIR = PROJECT_ROOT / "reports"

CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
CHROMA_DIR = VECTOR_STORE_DIR / "chroma"
CHROMA_COLLECTION = "courselens_rag"
