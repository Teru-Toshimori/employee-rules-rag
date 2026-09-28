from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_TIMEOUT,
)


class OpenAIEmbeddingService:
    """
    OpenAI Embedding APIを使用して、
    文書と質問をベクトル化するサービス。
    """

    def __init__(self) -> None:
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_EMBEDDING_MODEL
        self.timeout = OPENAI_TIMEOUT

    def _create_client(self) -> OpenAI:
        """
        OpenAIクライアントを生成する。
        """

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEYが設定されていません。"
            )

        return OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        複数の文書をOpenAI Embedding APIで
        ベクトル化する。
        """

        cleaned_texts = self._validate_texts(
            texts
        )

        client = self._create_client()

        response = client.embeddings.create(
            model=self.model,
            input=cleaned_texts,
        )

        embeddings = np.array(
            [
                item.embedding
                for item in response.data
            ],
            dtype=np.float32,
        )

        return self._normalize_embeddings(
            embeddings
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        質問をOpenAI Embedding APIで
        ベクトル化する。
        """

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "質問が空です。"
            )

        client = self._create_client()

        response = client.embeddings.create(
            model=self.model,
            input=[cleaned_query],
        )

        embedding = np.array(
            [
                response.data[0].embedding
            ],
            dtype=np.float32,
        )

        return self._normalize_embeddings(
            embedding
        )

    @staticmethod
    def _normalize_embeddings(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        ベクトルをL2正規化する。

        FAISSのIndexFlatIPで
        コサイン類似度として扱えるようにする。
        """

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        normalized = embeddings / norms

        return normalized.astype(
            np.float32
        )

    @staticmethod
    def _validate_texts(
        texts: Sequence[str],
    ) -> list[str]:
        """
        空文字などを除外する。
        """

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