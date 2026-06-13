import re

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(texts: list[str], top_n: int = 10) -> list[str]:
    texts = [text for text in texts if text.strip()]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1000,
    )
    matrix = vectorizer.fit_transform(texts)
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = scores.argsort()[::-1][:top_n]
    return [terms[index] for index in ranked]


def summarize_results(texts: list[str], max_sentences: int = 5) -> str:
    sentences = []
    for text in texts:
        sentences.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if len(sentence.strip()) > 40
        )
    if not sentences:
        return "No summary available."

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    scores = matrix.sum(axis=1).A1
    ranked = scores.argsort()[::-1][:max_sentences]
    selected = sorted(ranked)
    return " ".join(sentences[index] for index in selected)


def cluster_topics(texts: list[str], max_topics: int = 3) -> list[dict]:
    texts = [text for text in texts if len(text.split()) > 20]
    if len(texts) < 3:
        return []

    topic_count = min(max_topics, len(texts))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(texts)

    model = KMeans(n_clusters=topic_count, random_state=42, n_init="auto")
    labels = model.fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()

    rows = []
    for topic_id in range(topic_count):
        center = model.cluster_centers_[topic_id]
        top_terms = [terms[index] for index in center.argsort()[::-1][:6]]
        rows.append(
            {
                "topic": topic_id + 1,
                "keywords": ", ".join(top_terms),
                "chunks": int((labels == topic_id).sum()),
            }
        )
    return rows

