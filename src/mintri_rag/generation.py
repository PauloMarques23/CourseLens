import os
import re

from dotenv import load_dotenv


load_dotenv()


def configured_provider() -> str:
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "local"


def configured_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


def generate_answer(query: str, results: list) -> str:
    if not results:
        return "I could not find relevant information in the indexed corpus."

    if configured_provider() == "gemini":
        try:
            return _gemini_answer(query, results)
        except Exception as exc:
            fallback = _extractive_answer(query, results)
            return f"{fallback}\n\nNote: Gemini generation failed, so the local fallback was used. Error: {exc}"

    return _extractive_answer(query, results)


def _gemini_answer(query: str, results: list) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = (
        "You are CourseLens, a RAG assistant for Information Retrieval and Text Mining students.\n"
        "Answer the user question using only the provided retrieved context.\n"
        "Write in the same language as the question when possible.\n"
        "Cite evidence with [S1], [S2], etc.\n"
        "If the context is incomplete, say what is missing instead of inventing details.\n"
        "If the retrieved context is unrelated, say that the indexed corpus does not contain enough evidence.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{_format_context(results)}"
    )
    response = client.models.generate_content(
        model=configured_model(),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text or "Gemini returned an empty answer."


def _extractive_answer(query: str, results: list) -> str:
    query_terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z]{3,}", query)
        if token.lower() not in {"what", "why", "how", "the", "and", "are", "for"}
    }

    selected = []
    for source_number, result in enumerate(results, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", result.chunk["text"])
        scored = []
        for sentence in sentences:
            tokens = {token.lower() for token in re.findall(r"[A-Za-z]{3,}", sentence)}
            score = len(tokens & query_terms)
            if score:
                scored.append((score, sentence.strip()))
        scored.sort(reverse=True)
        for _, sentence in scored[:2]:
            if sentence and sentence not in selected:
                selected.append(f"- {sentence} [S{source_number}]")
        if len(selected) >= 5:
            break

    if not selected:
        selected = [
            f"- {result.chunk['text'][:350].strip()} [S{i}]"
            for i, result in enumerate(results[:3], start=1)
        ]

    return "Based on the retrieved material:\n\n" + "\n".join(selected[:5])


def _format_context(results: list) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[S{i}] Source: {chunk['source']} | Chunk: {chunk['chunk_id']}\n{chunk['text']}"
        )
    return "\n\n".join(blocks)
