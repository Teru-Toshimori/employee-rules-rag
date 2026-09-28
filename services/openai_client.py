from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_TIMEOUT,
)


class OpenAIClient:
    """
    OpenAI APIを使用して回答を生成するクライアント。
    """

    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_CHAT_MODEL
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

    def generate_answer(
        self,
        question: str,
        contexts: list[dict],
    ) -> str:
        """
        質問と検索された社員規則の条文から回答を生成する。
        """

        if not question.strip():
            raise ValueError(
                "質問が入力されていません。"
            )

        if not contexts:
            return "社員規則からは確認できません"

        context_text = self._build_context(contexts)

        prompt = f"""
あなたは社員規則について回答するAIアシスタントです。

以下の社員規則の内容だけを根拠として、
ユーザーの質問に日本語で簡潔に回答してください。

社員規則に回答の根拠がない場合は、
「社員規則からは確認できません」
と回答してください。

推測や一般知識による回答はしないでください。

【社員規則】
{context_text}

【質問】
{question}
""".strip()

        client = self._create_client()

        response = client.responses.create(
            model=self.model,
            input=prompt,
        )

        answer = response.output_text.strip()

        if not answer:
            return "社員規則からは確認できません"

        return answer

    @staticmethod
    def _build_context(
        contexts: list[dict],
    ) -> str:
        """
        FAISS検索結果をOpenAIへ渡す文章に変換する。
        """

        context_parts = []

        for index, context in enumerate(
            contexts,
            start=1,
        ):
            title = context.get(
                "title",
                "",
            )
            text = context.get(
                "text",
                "",
            )
            page = context.get(
                "page",
                "",
            )

            part = (
                f"[資料{index}]\n"
                f"条文: {title}\n"
                f"ページ: {page}\n"
                f"内容:\n{text}"
            )

            context_parts.append(part)

        return "\n\n".join(context_parts)