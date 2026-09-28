from __future__ import annotations

import json
import os

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)
from lark_oapi.core.const import (
    LARK_DOMAIN,
)


class LarkReplyClient:
    """
    Lark OpenAPIを使用して、
    受信したメッセージへBotとして返信するクライアント。
    """

    def __init__(self) -> None:
        self.app_id = os.getenv(
            "LARK_APP_ID",
            "",
        ).strip()

        self.app_secret = os.getenv(
            "LARK_APP_SECRET",
            "",
        ).strip()

        if not self.app_id:
            raise RuntimeError(
                "環境変数 LARK_APP_ID が設定されていません。"
            )

        if not self.app_secret:
            raise RuntimeError(
                "環境変数 LARK_APP_SECRET が設定されていません。"
            )

        self.client = (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .domain(LARK_DOMAIN)
            .build()
        )

    def reply(
        self,
        message_id: str,
        text: str,
    ) -> None:
        """
        指定されたLarkメッセージへ返信する。
        """

        if not message_id.strip():
            raise ValueError(
                "message_idが空です。"
            )

        if not text.strip():
            raise ValueError(
                "返信内容が空です。"
            )

        content = json.dumps(
            {
                "text": text,
            },
            ensure_ascii=False,
        )

        request = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type("text")
                .content(content)
                .build()
            )
            .build()
        )

        response = (
            self.client.im.v1.message.reply(
                request
            )
        )

        if not response.success():
            raise RuntimeError(
                "Larkへの返信に失敗しました。"
                f" code={response.code},"
                f" msg={response.msg}"
            )