from __future__ import annotations

import asyncio

from config import (
    PDF_PATH,
)

from bot.lark_bot import (
    LarkBot,
)
from bot.message_handler import (
    MessageHandler,
)
from services.embedding import (
    LocalEmbeddingService,
)
from services.index_cache import (
    is_index_cache_valid,
)
from services.ollama_client import (
    OllamaClient,
)
from services.rag_service import (
    RagService,
)
from services.system_check import (
    run_startup_checks,
)
from services.vector_store import (
    FaissVectorStore,
)


def initialize_message_handler() -> MessageHandler:
    """
    Lark Bot用のローカルRAGを初期化する。
    """

    print("=" * 60)
    print("社員規則AI Lark Bot")
    print("=" * 60)

    # ========================================================
    # 1. 動作環境チェック
    # ========================================================

    print(
        "[1/4] 動作環境を確認しています..."
    )

    run_startup_checks()

    print(
        "      動作環境チェック完了"
    )

    # ========================================================
    # 2. Embedding
    # ========================================================

    print(
        "[2/4] Embeddingモデルを"
        "読み込んでいます..."
    )

    embedding_service = (
        LocalEmbeddingService()
    )

    # ========================================================
    # 3. FAISS
    # ========================================================

    print(
        "[3/4] FAISSインデックスを"
        "確認しています..."
    )

    if not is_index_cache_valid(
        PDF_PATH
    ):
        raise RuntimeError(
            "FAISSインデックスがありません。\n"
            "先にGUI版を起動して"
            "インデックスを作成してください。"
        )

    vector_store = (
        FaissVectorStore()
    )

    print(
        "      保存済みFAISS"
        "インデックスを利用します。"
    )

    # ========================================================
    # 4. Ollama
    # ========================================================

    print(
        "[4/4] Ollamaを準備しています..."
    )

    ollama_client = (
        OllamaClient()
    )

    print(
        "      Ollama準備完了"
    )

    # ========================================================
    # RagService
    # ========================================================

    rag_service = RagService(
        embedding_service=(
            embedding_service
        ),
        vector_store=(
            vector_store
        ),
        ollama_client=(
            ollama_client
        ),
    )

    # ========================================================
    # MessageHandler
    # ========================================================

    return MessageHandler(
        rag_service=rag_service
    )


async def main() -> None:
    """
    Lark Botを起動する。
    """

    try:
        message_handler = (
            initialize_message_handler()
        )

        lark_bot = LarkBot(
            message_handler=(
                message_handler
            )
        )

        await lark_bot.run()

    except KeyboardInterrupt:
        print()
        print(
            "Lark Botを終了します。"
        )

    except Exception as error:
        import traceback

        print()
        print("=" * 60)
        print("Lark Bot 起動エラー")
        print("=" * 60)

        print(
            f"エラー型: {type(error).__name__}"
        )

        print(
            f"エラー内容: {repr(error)}"
        )

        print()
        print("詳細:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(
        main()
    )