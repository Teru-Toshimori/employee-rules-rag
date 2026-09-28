from __future__ import annotations

import traceback

from bot.reply_formatter import (
    format_lark_reply,
)


class MessageHandler:
    """
    Lark Botから受け取ったメッセージを処理するクラス。

    文字列を受け取り、
    RagServiceへ質問を渡して、
    Lark返信用文字列を返す。
    """

    def __init__(
        self,
        rag_service,
    ) -> None:
        self.rag_service = (
            rag_service
        )

    def handle_text_message(
        self,
        text: str,
    ) -> str:
        """
        テキストメッセージを処理し、
        Bot返信用文字列を返す。
        """

        # ========================================================
        # 入力チェック
        # ========================================================

        if not isinstance(
            text,
            str,
        ):
            return (
                "テキスト形式で質問を入力してください。"
            )

        question = (
            text.strip()
        )

        if not question:
            return (
                "質問内容を入力してください。"
            )

        # ========================================================
        # RAG処理
        # ========================================================

        try:
            result = (
                self.rag_service.ask(
                    question
                )
            )

            # ====================================================
            # RagService戻り値チェック
            # ====================================================

            if not isinstance(
                result,
                dict,
            ):
                raise RuntimeError(
                    "RagServiceから不正な形式の"
                    "結果が返されました。"
                )

            # ====================================================
            # Lark返信形式へ変換
            # ====================================================

            return format_lark_reply(
                result
            )

        except Exception as error:
            print()
            print("=" * 60)
            print("RAG処理エラー")
            print("=" * 60)

            print(
                f"エラー型: "
                f"{type(error).__name__}"
            )

            print(
                f"エラー内容: "
                f"{error}"
            )

            print()
            print("詳細:")

            traceback.print_exc()

            # ====================================================
            # ユーザー向けメッセージ
            # ====================================================

            return (
                "【エラー】\n"
                "現在、質問への回答を生成できません。\n"
                "しばらくしてからもう一度お試しください。"
            )