"""Dense (bge-small) and sparse (BM25) embeddings via fastembed, both CPU/ONNX."""

from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from app.config import Settings


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: models.SparseVector


class Embedder:
    def __init__(self, settings: Settings) -> None:
        cache = str(settings.embed_cache_dir)
        # DEBUG: on Windows without symlink rights the HF download logs WinError 1314 and then
        # falls back to another mirror. If models never load, delete .cache/fastembed and rerun
        # `python manage.py setup`.
        self._dense = TextEmbedding(settings.dense_model, cache_dir=cache)
        self._sparse = SparseTextEmbedding(settings.sparse_model, cache_dir=cache)
        self.dense_dim = len(next(iter(self._dense.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        dense = self._dense.embed(texts)
        sparse = self._sparse.embed(texts)
        return [_pack(d, s) for d, s in zip(dense, sparse, strict=True)]

    def embed_query(self, text: str) -> Embedding:
        # query_embed differs from embed: bge adds its query instruction and BM25 weights
        # query terms uniformly. Using embed() for queries silently degrades ranking.
        dense = next(iter(self._dense.query_embed(text)))
        sparse = next(iter(self._sparse.query_embed(text)))
        return _pack(dense, sparse)


def _pack(dense, sparse) -> Embedding:
    return Embedding(
        dense=dense.tolist(),
        sparse=models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
    )
