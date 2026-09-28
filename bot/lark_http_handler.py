from __future__ import annotations

import hmac
import json
import os
from typing import Any


class LarkHttpHandler:
    """
    LarkからHTTPで受信したイベントを解析するクラス。
    """

    @staticmethod
    def verify_token(
        body: dict[str, Any],
    ) -> bool:
        """
        LarkイベントのVerification Tokenを検証する。
        """

        expected_token = os.getenv(
            "LARK_VERIFICATION_TOKEN",
            "",
        ).strip()

        if not expected_token:
            raise RuntimeError(
                "環境変数 "
                "LARK_VERIFICATION_TOKEN "
                "が設定されていません。"
            )

        header = body.get(
            "header",
            {},
        )

        received_token = str(
            header.get(
                "token",
                "",
            )
        ).strip()

        if not received_token:
            return False

        return hmac.compare_digest(
            received_token,
            expected_token,
        )

    @staticmethod
    def extract_message(
        body: dict[str, Any],
    ) -> dict[str, str] | None:
        header = body.get(
            "header",
            {},
        )

        event_type = header.get(
            "event_type",
            "",
        )

        if event_type != "im.message.receive_v1":
            return None

        event = body.get(
            "event",
            {},
        )

        message = event.get(
            "message",
            {},
        )

        message_type = message.get(
            "message_type",
            "",
        )

        if message_type != "text":
            return None

        message_id = (
            message.get(
                "message_id",
                "",
            )
            or ""
        ).strip()

        chat_id = (
            message.get(
                "chat_id",
                "",
            )
            or ""
        ).strip()

        content = message.get(
            "content",
            "",
        )

        text = LarkHttpHandler._extract_text(
            content
        )

        if not text:
            return None

        if not chat_id:
            return None

        if not message_id:
            return None

        return {
            "text": text,
            "chat_id": chat_id,
            "message_id": message_id,
        }

    @staticmethod
    def _extract_text(
        content: Any,
    ) -> str:
        if isinstance(
            content,
            dict,
        ):
            return (
                str(
                    content.get(
                        "text",
                        "",
                    )
                ).strip()
            )

        if not isinstance(
            content,
            str,
        ):
            return ""

        try:
            content_data = json.loads(
                content
            )

        except json.JSONDecodeError:
            return content.strip()

        if not isinstance(
            content_data,
            dict,
        ):
            return ""

        return (
            str(
                content_data.get(
                    "text",
                    "",
                )
            ).strip()
        )