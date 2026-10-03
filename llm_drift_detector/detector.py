from __future__ import annotations

import re
from typing import Sequence

import numpy as np


class LLMDriftDetector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_model: bool = False):
        self.model_name = model_name
        self.use_model = use_model
        self.model = None

    def _fallback_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        tokens_by_text = [re.findall(r"\w+", str(text).lower()) for text in texts]
        vocab = sorted({token for tokens in tokens_by_text for token in tokens})
        vocab_index = {token: i for i, token in enumerate(vocab)}

        if not vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)

        rows = []
        for tokens in tokens_by_text:
            vector = np.zeros(len(vocab), dtype=np.float32)
            for token in tokens:
                index = vocab_index.get(token)
                if index is not None:
                    vector[index] += 1.0
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)

    def _get_model(self):
        if self.model is None:
            if not self.use_model:
                self.model = "fallback"
                return self.model

            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError:
                self.model = "fallback"
                return self.model

            self.model = SentenceTransformer(self.model_name)
        return self.model

    def get_embeddings(self, texts: Sequence[str] | str | None) -> np.ndarray:
        if texts is None:
            return np.empty((0, 0), dtype=np.float32)

        if isinstance(texts, str):
            texts = [texts]

        sanitized = [str(text).strip() for text in texts if str(text).strip()]
        if not sanitized:
            return np.empty((0, 0), dtype=np.float32)

        model = self._get_model()
        if model == "fallback":
            return self._fallback_embeddings(sanitized)

        embeddings = model.encode(sanitized)
        return np.asarray(embeddings, dtype=np.float32)

    def calculate_drift(
        self,
        reference_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> float:
        ref = np.asarray(reference_embeddings, dtype=np.float32)
        tar = np.asarray(target_embeddings, dtype=np.float32)

        if ref.size == 0 or tar.size == 0:
            return 0.0

        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if tar.ndim == 1:
            tar = tar.reshape(1, -1)

        if ref.shape[1] != tar.shape[1]:
            max_dim = max(ref.shape[1], tar.shape[1])
            ref = np.pad(ref, ((0, 0), (0, max_dim - ref.shape[1])), mode="constant")
            tar = np.pad(tar, ((0, 0), (0, max_dim - tar.shape[1])), mode="constant")

        ref_mean = np.mean(ref, axis=0)
        tar_mean = np.mean(tar, axis=0)

        ref_norm = np.linalg.norm(ref_mean)
        tar_norm = np.linalg.norm(tar_mean)

        if np.isclose(ref_norm, 0.0) or np.isclose(tar_norm, 0.0):
            return 0.0

        similarity = np.dot(ref_mean, tar_mean) / (ref_norm * tar_norm)
        drift = 1.0 - float(np.clip(similarity, -1.0, 1.0))
        return float(np.clip(drift, 0.0, 2.0))
