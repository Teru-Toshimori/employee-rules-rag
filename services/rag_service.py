from __future__ import annotations

from typing import Any

from config import SEARCH_TOP_K


class RagService:
    """
    OpenAIを使用したRAG処理を共通化するサービス。

    処理:
    1. 質問をEmbedding
    2. FAISSから関連チャンクを検索
    3. OpenAIへ質問と検索結果を送信
    4. 回答と参照情報を返す
    """

    NOT_FOUND_MESSAGE = (
        "社員規則からは確認できません"
    )

    def __init__(
        self,
        embedding_service,
        vector_store,
        openai_client,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.openai_client = openai_client

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        質問を受け取り、
        RAG検索と回答生成を行う。

        戻り値:
        {
            "answer": str,
            "references": list,
            "found": bool,
        }
        """

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "質問が空です。"
            )

        # ====================================================
        # 1. 質問をEmbedding
        # ====================================================

        query_vector = (
            self.embedding_service.embed_query(
                cleaned_question
            )
        )

        # ====================================================
        # 2. FAISS検索
        # ====================================================

        search_results = (
            self.vector_store.search(
                query_vector=query_vector,
                top_k=SEARCH_TOP_K,
            )
        )

        # ====================================================
        # 3. 検索結果なし
        # ====================================================

        if not search_results:
            return {
                "answer": self.NOT_FOUND_MESSAGE,
                "references": [],
                "found": False,
            }

        # ====================================================
        # 4. OpenAIで回答生成
        # ====================================================

        answer = (
            self.openai_client.generate_answer(
                question=cleaned_question,
                contexts=search_results,
            )
        )

        # ====================================================
        # 5. 回答可能か判定
        # ====================================================

        normalized_answer = answer.strip()

        found = (
            normalized_answer
            != self.NOT_FOUND_MESSAGE
        )

        # ====================================================
        # 6. 結果
        # ====================================================

        return {
            "answer": normalized_answer,
            "references": (
                search_results
                if found
                else []
            ),
            "found": found,
        }