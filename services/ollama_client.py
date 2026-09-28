from __future__ import annotations

import re
from typing import Any

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    OLLAMA_TIMEOUT,
)


class OllamaClient:
    """
    Ollama APIを使用し、
    検索した社員規則をもとに回答を生成する。
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL_NAME,
        timeout: int = OLLAMA_TIMEOUT,
    ) -> None:
        self.generate_url = (
            f"{base_url.rstrip('/')}/api/generate"
        )
        self.model_name = model_name
        self.timeout = timeout

    def generate_answer(
        self,
        question: str,
        search_results: list[dict[str, Any]],
    ) -> str:
        """
        質問と検索結果をOllamaへ送り、
        社員規則に基づく回答を生成する。
        """
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("質問が空です。")

        if not search_results:
            raise ValueError(
                "回答生成に使用する検索結果がありません。"
            )

        prompt = self._build_prompt(
            question=cleaned_question,
            search_results=search_results,
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
            },
        }

        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.ConnectionError as error:
            raise RuntimeError(
                "Ollamaへ接続できませんでした。"
                "Ollamaが起動しているか確認してください。"
            ) from error

        except requests.Timeout as error:
            raise RuntimeError(
                "Ollamaの回答生成がタイムアウトしました。"
            ) from error

        except requests.HTTPError as error:
            raise RuntimeError(
                "Ollama APIでエラーが発生しました。\n"
                f"ステータスコード: {response.status_code}\n"
                f"応答: {response.text}"
            ) from error

        try:
            data = response.json()

        except requests.JSONDecodeError as error:
            raise RuntimeError(
                "Ollamaから不正なJSONが返されました。"
            ) from error

        raw_answer = str(
            data.get("response", "")
        ).strip()

        if not raw_answer:
            raise RuntimeError(
                "Ollamaから回答を取得できませんでした。"
            )

        answer = self._remove_thinking_trace(
            raw_answer
        )

        if not answer:
            raise RuntimeError(
                "思考部分を除去した後の回答が空です。"
            )

        return answer

    @staticmethod
    def _build_prompt(
        question: str,
        search_results: list[dict[str, Any]],
    ) -> str:
        """
        社員規則と質問からプロンプトを作成する。
        """
        context_parts: list[str] = []

        for rank, result in enumerate(
            search_results,
            start=1,
        ):
            article = result.get(
                "article",
                "条文名不明",
            )

            start_page = result.get(
                "start_page",
                result.get("page", "不明"),
            )

            end_page = result.get(
                "end_page",
                start_page,
            )

            text = str(
                result.get("text", "")
            ).strip()

            if not text:
                continue

            if start_page == end_page:
                page_text = f"{start_page}ページ"
            else:
                page_text = (
                    f"{start_page}～{end_page}ページ"
                )

            context_parts.append(
                f"【検索順位{rank}："
                f"{article}／{page_text}】\n"
                f"{text}"
            )

        if not context_parts:
            raise ValueError(
                "有効な検索結果本文がありません。"
            )

        context = "\n\n".join(context_parts)

        return f"""
あなたは社員就業規則を案内する社内AIです。

以下の社員規則だけを根拠に回答してください。

回答ルール:
- 最も質問に直接関係する条文を優先する
- 条文の第1項、第2項などを順番に確認する
- 質問が「いつ付与されるか」の場合は、
  初回付与の条件と時期を優先して回答する
- 関係の薄い条文は回答へ含めない
- 社員規則にない内容を推測しない
- 2～4文程度で簡潔に回答する
- 根拠となる条文名とページ番号を最後に記載する
- 規則から確認できない場合は、
  「社員規則からは確認できません」と回答する
- 英語の説明や分析過程は出力しない
- <think>タグや思考内容は出力しない

社員規則:
{context}

質問:
{question} /no_think

最終回答のみを出力してください。
""".strip()

    @staticmethod
    def _remove_thinking_trace(
        answer: str,
    ) -> str:
        """
        Qwen3の思考内容が回答本文へ混入した場合に除去する。

        対応例:

        <think>
        思考内容
        </think>
        最終回答

        または:

        Thinking...
        思考内容
        ...done thinking.
        最終回答

        または:

        思考内容
        </think>
        最終回答
        """
        cleaned_answer = answer.strip()

        # <think>...</think> を削除
        cleaned_answer = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned_answer,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # 開始タグなしで </think> のみ存在する場合
        if "</think>" in cleaned_answer.lower():
            parts = re.split(
                r"</think>",
                cleaned_answer,
                flags=re.IGNORECASE,
            )

            cleaned_answer = parts[-1].strip()

        # Ollama CLI風のThinking表示を削除
        cleaned_answer = re.sub(
            r"^Thinking\.\.\..*?"
            r"(?:\.\.\.done thinking\.)",
            "",
            cleaned_answer,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        return cleaned_answer