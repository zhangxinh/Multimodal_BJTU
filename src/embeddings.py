import hashlib
import logging
import math
from typing import Iterable, List

from . import config
from .clients import get_embed_client, get_text_client

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, prefer_remote: bool | None = None, hash_dims: int = 256):
        self.prefer_remote = prefer_remote if prefer_remote is not None else config.PREFER_REMOTE_EMBEDDING
        self.hash_dims = hash_dims
        self._remote_available = None

    def embed(self, texts: List[str], target_dim: int | None = None) -> List[List[float]]:
        original_dim = self.hash_dims
        if target_dim:
            self.hash_dims = target_dim
        try:
            if self.prefer_remote and self._remote_available is not False:
                try:
                    return self._remote_embed(texts)
                except Exception as exc:  # noqa: BLE001
                    if self._remote_available is not False:
                        logger.warning(
                            "Remote embedding unavailable (%s). Using local hash embeddings. "
                            "Set PREFER_REMOTE_EMBEDDING=0 to silence this message.",
                            exc,
                        )
                    self._remote_available = False
            return [self._hash_embed(text) for text in texts]
        finally:
            self.hash_dims = original_dim

    def _remote_embed(self, texts: List[str]) -> List[List[float]]:
        client = get_embed_client()
        response = client.embeddings.create(model=config.TEXT_EMBED_MODEL, input=texts)
        self._remote_available = True
        if not getattr(response, "data", None):
            raise RuntimeError("No embedding data received")
        embeddings: List[List[float]] = []
        for item in response.data:
            emb = getattr(item, "embedding", None)
            if not emb:
                raise RuntimeError("Embedding item missing embedding field")
            embeddings.append(emb)
        return embeddings

    def _hash_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.hash_dims
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self.hash_dims
            vec[bucket] += 1.0
        return self._normalize(vec)

    def _tokenize(self, text: str) -> Iterable[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return (tok for tok in cleaned.split() if tok)

    def _normalize(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]
