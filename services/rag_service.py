from __future__ import annotations

from typing import Any

from config import (
    SEARCH_SCORE_THRESHOLD,
    SEARCH_TOP_K,
)


class RagService:
    """
    RAG処理を共通化するサービス。

    PySide6 GUIやLark Botなど、
    どのインターフェースからでも
    同じRAG処理を利用できるようにする。

    回答生成には、
    OllamaまたはOpenAIを使用できる。
    """

    def __init__(
        self,
        embedding_service,
        vector_store,
        ollama_client=None,
        openai_client=None,
        llm_provider: str = "ollama",
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.openai_client = openai_client
        self.llm_provider = llm_provider

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
                "answer": (
                    "社員規則からは確認できません"
                ),
                "references": [],
                "found": False,
            }

        # ====================================================
        # 4. Ollama版のみ類似度判定
        # ====================================================

        if self.llm_provider == "ollama":

            best_score = (
                search_results[0]["score"]
            )

            if (
                best_score
                < SEARCH_SCORE_THRESHOLD
            ):
                return {
                    "answer": (
                        "社員規則からは確認できません"
                    ),
                    "references": search_results,
                    "found": False,
                }

        # ====================================================
        # 5. 回答生成
        # ====================================================

        answer = self._generate_answer(
            question=cleaned_question,
            search_results=search_results,
        )

        # ====================================================
        # 6. 回答可能か判定
        # ====================================================

        not_found_message = (
            "社員規則からは確認できません"
        )

        normalized_answer = (
            answer.strip()
        )

        found = (
            normalized_answer
            != not_found_message
        )

        # ====================================================
        # 7. 結果
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

    def _generate_answer(
        self,
        question: str,
        search_results: list[dict],
    ) -> str:
        """
        設定されたLLMを使用して回答を生成する。
        """

        # ====================================================
        # OpenAI
        # ====================================================

        if self.llm_provider == "openai":

            if self.openai_client is None:
                raise RuntimeError(
                    "OpenAIClientが設定されていません。"
                )

            return (
                self.openai_client.generate_answer(
                    question=question,
                    contexts=search_results,
                )
            )

        # ====================================================
        # Ollama
        # ====================================================

        if self.llm_provider == "ollama":

            if self.ollama_client is None:
                raise RuntimeError(
                    "OllamaClientが設定されていません。"
                )

            return (
                self.ollama_client.generate_answer(
                    question=question,
                    search_results=search_results,
                )
            )

        # ====================================================
        # 未対応
        # ====================================================

        raise ValueError(
            f"未対応のLLMプロバイダーです: "
            f"{self.llm_provider}"
        )