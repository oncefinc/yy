"""
文本嵌入层 — bge-small-zh-v1.5 ONNX 模型
首次运行自动下载（~40MB），之后完全离线
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger("memory.embedder")

# BGE 模型要求的查询前缀
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 单例
_embedder: Optional["Embedder"] = None


def get_embedder() -> "Embedder":
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


class Embedder:
    """文本嵌入器，封装 fastembed / ONNX 推理"""

    def __init__(self):
        self._model = None
        self._dim = EMBEDDING_DIM
        self._ready = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """加载模型（首次调用自动触发）"""
        if self._ready:
            return
        try:
            from fastembed import TextEmbedding
            logger.info(f"正在加载嵌入模型: {EMBEDDING_MODEL}")
            self._model = TextEmbedding(
                model_name=EMBEDDING_MODEL,
                max_length=512,
                threads=2,  # 台式机 16G 保守设置
            )
            # 预热一下，顺便触发模型下载
            _ = list(self._model.embed(["预热"]))
            self._ready = True
            logger.info("嵌入模型加载完成")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise RuntimeError(f"无法加载嵌入模型 {EMBEDDING_MODEL}: {e}") from e

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        将文本列表转为向量矩阵 (N, 512)

        Args:
            texts: 待编码文本
            is_query: 是否为查询（查询需要加前缀）
        """
        if not self._ready:
            self.load()
        if not texts:
            return np.array([]).reshape(0, self._dim)

        # 查询加 BGE 前缀，文档不加
        if is_query:
            texts = [f"{_QUERY_PREFIX}{t}" for t in texts]
        else:
            texts = [t for t in texts]  # 不加前缀

        embeddings = list(self._model.embed(texts))
        result = np.array(embeddings, dtype=np.float32)

        # L2 归一化（cosine 相似度 = 归一化后内积）
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = result / norms

        return result

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """编码单条文本，返回 (512,)"""
        return self.encode([text], is_query=is_query)[0]
