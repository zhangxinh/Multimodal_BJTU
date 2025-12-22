import logging
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from . import config, pdf_utils, storage
from .clients import get_text_client
from .embeddings import TextEmbedder

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class PaperManager:
    def __init__(self, embedder: TextEmbedder | None = None):
        self.embedder = embedder or TextEmbedder()
        config.PAPERS_DIR.mkdir(exist_ok=True)

    def add_paper(self, pdf_path: str, topics: List[str] | None = None) -> Dict:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"{pdf_path} does not exist")
        topics = [t.strip() for t in (topics or []) if t.strip()]

        pages = pdf_utils.extract_text_by_page(str(path))
        chosen_topics = self._classify_topics(pages, topics)
        target_dir = self._ensure_topic_dir(chosen_topics[0] if chosen_topics else "uncategorized")
        dest_path = target_dir / path.name
        if path.resolve() != dest_path.resolve():
            dest_path = self._resolve_collision(dest_path)
            shutil.move(str(path), dest_path)
        else:
            dest_path = path

        doc_text = " ".join(pages)
        trimmed_text = doc_text[:5000]
        paper_embedding = self.embedder.embed([trimmed_text])[0]
        summary = trimmed_text[:500]

        paper_index = storage.load_index(config.PAPER_INDEX_PATH)
        paper_entry = {
            "path": str(dest_path),
            "topics": chosen_topics or ["uncategorized"],
            "summary": summary,
            "embedding": paper_embedding,
        }
        paper_index = [p for p in paper_index if p.get("path") != str(dest_path)]
        paper_index.append(paper_entry)
        storage.save_index(config.PAPER_INDEX_PATH, paper_index)

        chunk_index = storage.load_index(config.CHUNK_INDEX_PATH)
        chunk_index = [c for c in chunk_index if c.get("paper_path") != str(dest_path)]
        chunks = pdf_utils.chunk_pages(pages)
        embeddings = self.embedder.embed([c[1] for c in chunks])
        for (page_number, text), embedding in zip(chunks, embeddings):
            chunk_index.append(
                {
                    "paper_path": str(dest_path),
                    "page": page_number,
                    "text": text,
                    "embedding": embedding,
                }
            )
        storage.save_index(config.CHUNK_INDEX_PATH, chunk_index)

        return {"path": str(dest_path), "topics": chosen_topics, "chunks_indexed": len(chunks)}

    def batch_organize(self, source_dir: str, topics: List[str] | None = None) -> List[Dict]:
        dir_path = Path(source_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"{source_dir} does not exist")
        effective_topics = [t.strip() for t in (topics or []) if t.strip()]
        if not effective_topics:
            effective_topics = self._known_topics()
        if not effective_topics:
            effective_topics = ["uncategorized"]
        results: List[Dict] = []
        for pdf in dir_path.rglob("*.pdf"):
            try:
                results.append(self.add_paper(str(pdf), effective_topics))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to add %s: %s", pdf, exc)
        return results

    def search_papers(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
        paper_index = storage.load_index(config.PAPER_INDEX_PATH)
        if not paper_index:
            return []
        ref_dim = len(paper_index[0].get("embedding", [])) or None
        query_embedding = self.embedder.embed([query], target_dim=ref_dim)[0]
        scored = []
        for entry in paper_index:
            score = cosine_similarity(query_embedding, entry.get("embedding", []))
            scored.append(
                {
                    "path": entry.get("path"),
                    "topics": entry.get("topics", []),
                    "summary": entry.get("summary", ""),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_chunks(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
        chunk_index = storage.load_index(config.CHUNK_INDEX_PATH)
        if not chunk_index:
            return []
        ref_dim = len(chunk_index[0].get("embedding", [])) or None
        query_embedding = self.embedder.embed([query], target_dim=ref_dim)[0]
        scored = []
        for entry in chunk_index:
            score = cosine_similarity(query_embedding, entry.get("embedding", []))
            scored.append(
                {
                    "paper_path": entry.get("paper_path"),
                    "page": entry.get("page"),
                    "text": entry.get("text"),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _classify_topics(self, pages: List[str], topics: List[str]) -> List[str]:
        if not topics:
            return ["uncategorized"]
        document_sample = " ".join(pages)[:4000]
        prompt = (
            "Given the candidate topics, choose the best fitting ones for the paper content. "
            "Return a comma separated list of topics from the provided options only."
        )
        try:
            client = get_text_client()
            response = client.chat.completions.create(
                model=config.TEXT_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Topics: {topics}\n\nPaper content preview:\n{document_sample}",
                    },
                ],
                temperature=0,
            )
            answer = response.choices[0].message.content or ""
            selected = [t.strip() for t in answer.split(",") if t.strip() and t.strip() in topics]
            if selected:
                return selected
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM classification failed, using heuristic: %s", exc)
        return [self._keyword_match(document_sample, topics)]

    def _keyword_match(self, text: str, topics: List[str]) -> str:
        lowered = text.lower()
        scores: List[Tuple[int, str]] = []
        for topic in topics:
            score = lowered.count(topic.lower())
            scores.append((score, topic))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1] if scores else topics[0]

    def _ensure_topic_dir(self, topic: str) -> Path:
        target = config.PAPERS_DIR / topic
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _known_topics(self) -> List[str]:
        # 收集 papers/ 下已有的主题目录名称（跳过文件和 uncategorized）
        topics: List[str] = []
        for entry in config.PAPERS_DIR.iterdir():
            if entry.is_dir() and entry.name != "uncategorized":
                topics.append(entry.name)
        return topics

    def _resolve_collision(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        parent = path.parent
        for idx in range(1, 1000):
            candidate = parent / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Cannot resolve name for {path}")
