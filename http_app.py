from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import lark_oapi as lark
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)

from lark_oapi.core.model.raw_request import (
    RawRequest,
)

from bot.lark_http_handler import (
    LarkHttpHandler,
)
from bot.lark_reply_client import (
    LarkReplyClient,
)
from bot.message_handler import (
    MessageHandler,
)
from config import (
    OPENAI_CHUNKS_PATH,
    OPENAI_FAISS_INDEX_PATH,
)
from services.openai_client import (
    OpenAIClient,
)
from services.openai_embedding import (
    OpenAIEmbeddingService,
)
from services.rag_service import (
    RagService,
)
from services.vector_store import (
    FaissVectorStore,
)


app = FastAPI(
    title="社員規則AI API",
    version="1.0.0",
)


message_handler: MessageHandler | None = None
lark_reply_client: LarkReplyClient | None = None
lark_event_dispatcher = None


# ============================================================
# Larkイベント重複防止
# ============================================================

# message_id -> 登録時刻（monotonic）
processed_message_ids: dict[str, float] = {}

# 同時に同じmessage_idが届いた場合の競合を防ぐ
processed_message_ids_lock = asyncio.Lock()

# 処理済みmessage_idを保持する時間
# 30分
MESSAGE_ID_TTL_SECONDS = 30 * 60


def cleanup_processed_message_ids(
    current_time: float,
) -> None:
    """
    保持期限を過ぎたmessage_idを削除する。

    processed_message_ids_lockを取得した状態で
    呼び出すことを前提とする。
    """

    expired_message_ids = [
        message_id
        for message_id, registered_at
        in processed_message_ids.items()
        if (
            current_time - registered_at
            > MESSAGE_ID_TTL_SECONDS
        )
    ]

    for message_id in expired_message_ids:
        processed_message_ids.pop(
            message_id,
            None,
        )


async def register_message_id(
    message_id: str,
) -> bool:
    """
    message_idを処理中として登録する。

    初回:
        True

    すでに登録済み:
        False

    Lock内で確認と登録をまとめて行うことで、
    同じmessage_idがほぼ同時に到着した場合でも
    1件だけがRAG処理へ進むようにする。
    """

    async with processed_message_ids_lock:
        current_time = time.monotonic()

        cleanup_processed_message_ids(
            current_time
        )

        if message_id in processed_message_ids:
            return False

        processed_message_ids[
            message_id
        ] = current_time

        return True


def validate_environment_variables() -> None:
    """
    本番HTTP版で必要な環境変数を確認する。
    """

    required_variables = [
        "OPENAI_API_KEY",
        "LARK_APP_ID",
        "LARK_APP_SECRET",
        "LARK_VERIFICATION_TOKEN",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(
            variable,
            "",
        ).strip()
    ]

    if missing_variables:
        missing_text = ", ".join(
            missing_variables
        )

        raise RuntimeError(
            "必須環境変数が設定されていません: "
            f"{missing_text}"
        )

    print(
        "必須環境変数チェック完了"
    )


def initialize_message_handler() -> MessageHandler:
    """
    OpenAI版RAGを初期化する。
    """

    print("=" * 60)
    print("社員規則AI HTTP版")
    print("OpenAI RAG")
    print("=" * 60)

    print(
        "[1/4] OpenAI Embeddingを"
        "準備しています..."
    )

    embedding_service = (
        OpenAIEmbeddingService()
    )

    print(
        "      OpenAI Embedding準備完了"
    )

    print(
        "[2/4] OpenAI用FAISSインデックスを"
        "確認しています..."
    )

    if not OPENAI_FAISS_INDEX_PATH.exists():
        raise RuntimeError(
            "OpenAI用FAISSインデックスが"
            "見つかりません。\n"
            f"{OPENAI_FAISS_INDEX_PATH}"
        )

    if not OPENAI_CHUNKS_PATH.exists():
        raise RuntimeError(
            "OpenAI用チャンクデータが"
            "見つかりません。\n"
            f"{OPENAI_CHUNKS_PATH}"
        )

    vector_store = FaissVectorStore(
        index_path=OPENAI_FAISS_INDEX_PATH,
        chunks_path=OPENAI_CHUNKS_PATH,
    )

    print(
        "      OpenAI用FAISS確認完了"
    )

    print(
        "[3/4] OpenAI回答クライアントを"
        "準備しています..."
    )

    openai_client = (
        OpenAIClient()
    )

    print(
        "      OpenAI回答クライアント準備完了"
    )

    print(
        "[4/4] OpenAI版RAGを"
        "準備しています..."
    )

    rag_service = RagService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        openai_client=openai_client,
    )

    print(
        "      OpenAI版RAG準備完了"
    )

    print("=" * 60)
    print("社員規則AI HTTP版 起動準備完了")
    print("=" * 60)

    return MessageHandler(
        rag_service=rag_service
    )


def initialize_lark_event_dispatcher():
    """
    Lark公式SDKのイベントDispatcherを初期化する。

    URL Verificationでは、
    SDK側でVerification Tokenを検証し、
    challengeを返却する。
    """

    verification_token = os.getenv(
        "LARK_VERIFICATION_TOKEN",
        "",
    ).strip()

    dispatcher = (
        lark.EventDispatcherHandler
        .builder(
            "",
            verification_token,
        )
        .build()
    )

    return dispatcher


@app.on_event("startup")
def startup_event() -> None:
    global message_handler
    global lark_reply_client
    global lark_event_dispatcher

    # --------------------------------------------------------
    # 必須環境変数チェック
    # --------------------------------------------------------

    validate_environment_variables()

    # --------------------------------------------------------
    # OpenAI RAG初期化
    # --------------------------------------------------------

    message_handler = (
        initialize_message_handler()
    )

    # --------------------------------------------------------
    # Lark公式SDK Dispatcher初期化
    # --------------------------------------------------------

    lark_event_dispatcher = (
        initialize_lark_event_dispatcher()
    )

    print(
        "Lark Event Dispatcher初期化完了"
    )

    # --------------------------------------------------------
    # Lark返信クライアント初期化
    # --------------------------------------------------------

    app_id = os.getenv(
        "LARK_APP_ID",
        "",
    ).strip()

    app_secret = os.getenv(
        "LARK_APP_SECRET",
        "",
    ).strip()

    if app_id and app_secret:
        lark_reply_client = (
            LarkReplyClient()
        )

        print(
            "Lark返信クライアント初期化完了"
        )

    else:
        print(
            "Lark認証情報が未設定のため、"
            "Lark返信機能は無効です。"
        )


@app.get("/")
def root() -> dict[str, str]:
    """
    APIルート。
    """

    return {
        "status": "ok",
        "message": (
            "社員規則AI API is running"
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    """
    ヘルスチェック。
    """

    return {
        "status": "healthy",
    }


@app.post("/lark/events")
async def lark_events(
    request: Request,
):
    """
    Larkイベント受信用エンドポイント。
    """

    raw_body = await request.body()

    try:
        body = json.loads(
            raw_body.decode("utf-8")
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON",
        )

    # --------------------------------------------------------
    # Lark URL Verification
    # --------------------------------------------------------

    challenge = body.get(
        "challenge"
    )

    if challenge:
        if lark_event_dispatcher is None:
            raise RuntimeError(
                "Lark Event Dispatcherが"
                "初期化されていません。"
            )

        print(
            "Lark URL Verificationを受信しました。"
        )

        raw_request = RawRequest()

        raw_request.uri = str(
            request.url
        )

        raw_request.headers = {
            key: value
            for key, value
            in request.headers.items()
        }

        raw_request.body = raw_body

        sdk_response = (
            lark_event_dispatcher.do(
                raw_request
            )
        )

        if sdk_response.status_code != 200:
            print(
                "Lark URL Verificationの"
                "検証に失敗しました。"
            )

            raise HTTPException(
                status_code=401,
                detail=(
                    "Lark URL Verification failed"
                ),
            )

        print(
            "Lark URL Verificationに成功しました。"
        )

        return Response(
            content=sdk_response.content,
            status_code=sdk_response.status_code,
            media_type="application/json",
        )

    # --------------------------------------------------------
    # 通常イベントのVerification Token検証
    # --------------------------------------------------------

    if not LarkHttpHandler.verify_token(
        body
    ):
        print(
            "Lark Verification Tokenの"
            "検証に失敗しました。"
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid verification token"
            ),
        )

    # --------------------------------------------------------
    # Larkメッセージ解析
    # --------------------------------------------------------

    message = (
        LarkHttpHandler.extract_message(
            body
        )
    )

    if message is None:
        print(
            "処理対象外のLarkイベントを"
            "受信しました。"
        )

        return {
            "status": "ignored",
        }

    # --------------------------------------------------------
    # message_id取得
    # --------------------------------------------------------

    message_id = message[
        "message_id"
    ]

    # message_idそのものはログへ出力しない。
    # 同一イベントの再送確認用として
    # 末尾6文字だけ記録する。
    message_id_suffix = (
        message_id[-6:]
        if len(message_id) >= 6
        else "short-id"
    )

    # 質問本文はログへ出力しない
    print(
        "Lark HTTPメッセージを受信しました。 "
        f"message_id_suffix={message_id_suffix}"
    )

    # --------------------------------------------------------
    # 重複イベントチェック
    # --------------------------------------------------------

    is_new_message = await register_message_id(
        message_id
    )

    if not is_new_message:
        print(
            "重複Larkイベントを検出しました。 "
            "RAG処理と返信をスキップします。 "
            f"message_id_suffix={message_id_suffix}"
        )

        return {
            "status": "duplicate_ignored",
        }

    print(
        "Larkイベントを処理対象として登録しました。 "
        f"message_id_suffix={message_id_suffix}"
    )

    # --------------------------------------------------------
    # RAG回答生成
    # --------------------------------------------------------

    if message_handler is None:
        raise RuntimeError(
            "MessageHandlerが"
            "初期化されていません。"
        )

    reply_text = await asyncio.to_thread(
        message_handler.handle_text_message,
        message["text"],
    )

    # 回答本文はログへ出力しない
    print(
        "RAG回答生成が完了しました。 "
        f"message_id_suffix={message_id_suffix}"
    )

    # --------------------------------------------------------
    # Larkへ返信
    # --------------------------------------------------------

    is_test_message = (
        message_id.startswith(
            "test-"
        )
    )

    if is_test_message:
        print(
            "疑似テストイベントのため"
            "Larkへの返信をスキップしました。"
        )

    elif lark_reply_client is None:
        print(
            "Lark返信クライアントが"
            "初期化されていないため"
            "返信をスキップしました。"
        )

    else:
        await asyncio.to_thread(
            lark_reply_client.reply,
            message_id,
            reply_text,
        )

        print(
            "Larkへの返信が完了しました。 "
            f"message_id_suffix={message_id_suffix}"
        )

    # 回答本文はHTTPレスポンスへ含めない
    return {
        "status": "ok",
    }