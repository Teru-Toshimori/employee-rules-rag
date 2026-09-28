from __future__ import annotations

import asyncio
import os
import traceback

from lark_oapi.channel import (
    FeishuChannel,
)
from lark_oapi.core.const import (
    LARK_DOMAIN,
)

from bot.message_handler import (
    MessageHandler,
)


class LarkBot:
    """
    LarkとローカルRAGを接続するBot。

    Larkから受信したテキストメッセージを
    MessageHandlerへ渡し、
    生成された回答をLarkへ返信する。
    """

    def __init__(
        self,
        message_handler: MessageHandler,
    ) -> None:
        self.message_handler = (
            message_handler
        )

        # ========================================================
        # 環境変数
        # ========================================================

        self.app_id = (
            os.getenv(
                "LARK_APP_ID",
                "",
            ).strip()
        )

        self.app_secret = (
            os.getenv(
                "LARK_APP_SECRET",
                "",
            ).strip()
        )

        if not self.app_id:
            raise RuntimeError(
                "環境変数 LARK_APP_ID "
                "が設定されていません。"
            )

        if not self.app_secret:
            raise RuntimeError(
                "環境変数 LARK_APP_SECRET "
                "が設定されていません。"
            )

        # ========================================================
        # Lark Channel
        # ========================================================
        #
        # Lark Developerは
        # https://open.larksuite.com
        # を使用するため、
        # LARK_DOMAINを明示する。
        #
        # ========================================================

        self.channel = FeishuChannel(
            app_id=self.app_id,
            app_secret=self.app_secret,
            domain=LARK_DOMAIN,
        )

        # ========================================================
        # イベント登録
        # ========================================================

        self.channel.on(
            "message",
            self._on_message,
        )

        self.channel.on(
            "error",
            self._on_error,
        )

    async def _on_message(
        self,
        message,
    ) -> None:
        """
        Larkからメッセージを受信したときの処理。
        """

        # ========================================================
        # テキスト取得
        # ========================================================

        try:
            text = (
                message.content_text
                or ""
            ).strip()

        except Exception as error:
            self._print_error(
                "メッセージ取得エラー",
                error,
            )
            return

        # ========================================================
        # 空メッセージ
        # ========================================================

        if not text:
            return

        print()
        print("=" * 60)
        print("Larkメッセージ受信")
        print("=" * 60)

        print(
            f"質問: {text}"
        )

        # ========================================================
        # RAG処理
        # ========================================================
        #
        # MessageHandler側でRAGの例外を処理する。
        #
        # asyncioイベントループをブロックしないよう
        # 別スレッドで実行する。
        #
        # ========================================================

        try:
            reply_text = (
                await asyncio.to_thread(
                    self.message_handler.handle_text_message,
                    text,
                )
            )

        except Exception as error:
            self._print_error(
                "MessageHandler実行エラー",
                error,
            )

            reply_text = (
                "【エラー】\n"
                "現在、質問への回答を生成できません。\n"
                "しばらくしてからもう一度お試しください。"
            )

        print(
            f"回答: {reply_text}"
        )

        # ========================================================
        # Larkへ返信
        # ========================================================

        try:
            await self.channel.send(
                message.conversation.chat_id,
                {
                    "markdown": reply_text,
                },
                {
                    "reply_to": (
                        message.message_id
                    ),
                },
            )

        except Exception as error:
            # ----------------------------------------------------
            # ここでエラーになった場合は
            # Larkそのものへの送信に失敗しているため、
            # ユーザーへ追加メッセージを送ろうとしない。
            # ----------------------------------------------------

            self._print_error(
                "Lark返信送信エラー",
                error,
            )

    async def _on_error(
        self,
        error,
    ) -> None:
        """
        Lark Channelエラー。
        """

        self._print_error(
            "Lark接続エラー",
            error,
        )

    def _print_error(
        self,
        title: str,
        error,
    ) -> None:
        """
        エラー情報をコンソールへ表示する。

        App Secretなどの認証情報は
        ログへ出力しない。
        """

        print()
        print("=" * 60)
        print(title)
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

        traceback.print_exception(
            type(error),
            error,
            error.__traceback__,
        )

    async def run(
        self,
    ) -> None:
        """
        Lark Botを起動する。
        """

        print("=" * 60)
        print("Lark Bot 起動")
        print("=" * 60)

        print(
            f"Lark API Domain: "
            f"{LARK_DOMAIN}"
        )

        print(
            "Lark WebSocketへ接続します..."
        )

        try:
            await self.channel.connect()

        except Exception as error:
            self._print_error(
                "Lark WebSocket接続エラー",
                error,
            )

            raise