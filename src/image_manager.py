import base64
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List

from . import config, storage
from .clients import get_vision_client
from .embeddings import TextEmbedder
from .paper_manager import cosine_similarity

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


class ImageManager:
    def __init__(self, embedder: TextEmbedder | None = None):
        self.embedder = embedder or TextEmbedder()
        config.IMAGES_DIR.mkdir(exist_ok=True)

    def index_images(self, image_dir: str | None = None) -> List[Dict]:
        directory = Path(image_dir) if image_dir else config.IMAGES_DIR
        if not directory.exists():
            raise FileNotFoundError(f"{directory} does not exist")

        index = storage.load_index(config.IMAGE_INDEX_PATH)
        indexed_paths = {item["path"] for item in index}
        new_entries: List[Dict] = []

        for image_path in directory.rglob("*"):
            if image_path.is_dir() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if str(image_path) in indexed_paths:
                continue
            caption = self._caption_image(str(image_path))
            embedding = self.embedder.embed([caption])[0]
            entry = {"path": str(image_path), "caption": caption, "embedding": embedding}
            index.append(entry)
            new_entries.append(entry)

        if new_entries:
            storage.save_index(config.IMAGE_INDEX_PATH, index)
        return new_entries

    def search_images(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
        index = storage.load_index(config.IMAGE_INDEX_PATH)

        # 自动补充新图片的索引，已索引的跳过
        directory = config.IMAGES_DIR
        indexed_paths = {item.get("path") for item in index}
        has_new_files = any(
            str(p) not in indexed_paths
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not index or has_new_files:
            self.index_images(str(directory))
            index = storage.load_index(config.IMAGE_INDEX_PATH)

        if not index:
            return []
        ref_dim = len(index[0].get("embedding", [])) or None
        query_embedding = self.embedder.embed([query], target_dim=ref_dim)[0]
        scored = []
        for entry in index:
            score = cosine_similarity(query_embedding, entry.get("embedding", []))
            scored.append(
                {
                    "path": entry.get("path"),
                    "caption": entry.get("caption", ""),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _caption_image(self, image_path: str) -> str:
        prompt = (
            "Describe the image briefly (<=40 words) focusing on what a user might search for. "
            "Do not add extra commentary."
        )
        try:
            client = get_vision_client()
            image_b64, mime_type = self._encode_image(image_path)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ]
            response = client.chat.completions.create(
                model=config.VISION_MODEL, messages=[{"role": "user", "content": content}]
            )
            caption = response.choices[0].message.content or ""
            return caption.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision caption failed for %s: %s", image_path, exc)
            return Path(image_path).stem

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(image_path)
        return b64, (mime_type or "image/png")
