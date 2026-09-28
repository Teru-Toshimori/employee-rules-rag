from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_PATH,
)


class LocalEmbeddingService:
    """
    Sentence Transformersを使用し、
    ローカル環境で文章をベクトル化する。
    """

    def __init__(self) -> None:
        if not EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Embeddingモデルが見つかりません。\n\n"
                f"確認先:\n{EMBEDDING_MODEL_PATH}"
            )

        print(
            "Embeddingモデルを読み込みます: "
            f"{EMBEDDING_MODEL_PATH}"
        )

        self.model = SentenceTransformer(
            str(EMBEDDING_MODEL_PATH),
            local_files_only=True,
        )

        print(
            "Embeddingモデルの読み込みが完了しました。"
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        文書をベクトル化する。
        """
        cleaned_texts = self._validate_texts(
            texts
        )

        passages = [
            f"passage: {text}"
            for text in cleaned_texts
        ]

        embeddings = self.model.encode(
            passages,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.astype(
            np.float32
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        質問文をベクトル化する。
        """
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "質問が空です。"
            )

        embedding = self.model.encode(
            [
                f"query: {cleaned_query}"
            ],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(
            np.float32
        )

    @staticmethod
    def _validate_texts(
        texts: Sequence[str],
    ) -> list[str]:
        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            raise ValueError(
                "ベクトル化する文章がありません。"
            )

        return cleaned_texts