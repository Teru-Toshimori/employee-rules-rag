import sys

from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from config import (
    OLLAMA_MODEL_NAME,
    PDF_PATH,
)

from services.chunker import (
    split_pages_into_articles,
)
from services.embedding import (
    LocalEmbeddingService,
)
from services.index_cache import (
    is_index_cache_valid,
    save_index_metadata,
)
from services.ollama_client import (
    OllamaClient,
)
from services.pdf_reader import (
    extract_pdf_pages,
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
from ui.main_window import (
    MainWindow,
)


def initialize_rag():
    """
    RAGエンジンを初期化する。

    処理:
    1. 起動前チェック
    2. Embeddingモデル読み込み
    3. FAISSインデックス確認
    4. Ollamaクライアント準備
    5. RagService作成
    """

    print("=" * 60)
    print("完全ローカルRAG 起動中")
    print("=" * 60)

    print(f"対象PDF: {PDF_PATH}")
    print()

    # ========================================================
    # 1. 起動前チェック
    # ========================================================

    print(
        "[1/4] 動作環境を確認しています..."
    )

    run_startup_checks()

    print(
        "      動作環境チェック完了"
    )

    # ========================================================
    # 2. Embeddingモデル
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

    vector_store = (
        FaissVectorStore()
    )

    print(
        "[3/4] FAISSインデックスを"
        "確認しています..."
    )

    if is_index_cache_valid(
        PDF_PATH
    ):
        print(
            "      保存済みFAISS"
            "インデックスを利用します。"
        )

    else:
        print(
            "      インデックスを"
            "新しく作成します。"
        )

        create_index(
            embedding_service=(
                embedding_service
            ),
            vector_store=(
                vector_store
            ),
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
    # 5. RagService
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

    print()
    print("=" * 60)
    print("完全ローカルRAG 起動完了")
    print("=" * 60)

    return (
        rag_service,
        embedding_service,
        vector_store,
    )


def create_index(
    embedding_service,
    vector_store,
) -> None:
    """
    PDFからFAISSインデックスを作成する。
    """

    print(
        "      PDFを読み込んでいます..."
    )

    pages = extract_pdf_pages(
        PDF_PATH
    )

    print(
        f"      PDF読み込み完了 "
        f"({len(pages)}ページ)"
    )

    print(
        "      条文を分割しています..."
    )

    chunks = (
        split_pages_into_articles(
            pages
        )
    )

    print(
        f"      条文分割完了 "
        f"({len(chunks)}チャンク)"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "      ベクトル化しています..."
    )

    vectors = (
        embedding_service
        .embed_documents(
            texts
        )
    )

    print(
        f"      ベクトル化完了 "
        f"({len(vectors)}件)"
    )

    total_count = (
        vector_store.create_index(
            vectors=vectors,
            chunks=chunks,
        )
    )

    print(
        f"      FAISS登録完了 "
        f"({total_count}件)"
    )

    save_index_metadata(
        pdf_path=PDF_PATH,
        chunk_count=len(chunks),
    )

    print(
        "      インデックス情報を"
        "保存しました。"
    )


def show_startup_error(
    error_message: str,
) -> None:
    """
    起動時に発生した技術的なエラーを、
    ユーザー向けの案内メッセージへ変換する。
    """

    # ========================================================
    # Ollamaに接続できない
    # ========================================================

    if (
        "[OLLAMA_CONNECTION_ERROR]"
        in error_message
    ):
        message = (
            "ローカルAI（Ollama）に接続できません。\n\n"
            "このツールを使用するには、"
            "Ollamaのインストールと起動が必要です。\n\n"
            "【確認手順】\n"
            "1. Ollamaがインストールされているか確認\n"
            "2. Ollamaを起動\n"
            "3. このアプリを再起動\n\n"
            "PowerShellで確認する場合:\n"
            "ollama --version"
        )

    # ========================================================
    # Ollamaモデルがない
    # ========================================================

    elif (
        "[OLLAMA_MODEL_NOT_FOUND]"
        in error_message
    ):
        message = (
            "回答生成用AIモデルが"
            "インストールされていません。\n\n"
            f"必要なモデル:\n"
            f"{OLLAMA_MODEL_NAME}\n\n"
            "PowerShellを開いて、"
            "次のコマンドを実行してください。\n\n"
            f"ollama pull {OLLAMA_MODEL_NAME}\n\n"
            "ダウンロード完了後、"
            "このアプリを再起動してください。"
        )

    # ========================================================
    # Ollamaタイムアウト
    # ========================================================

    elif (
        "[OLLAMA_TIMEOUT]"
        in error_message
    ):
        message = (
            "Ollamaへの接続が"
            "タイムアウトしました。\n\n"
            "Ollamaが正常に起動しているか確認し、"
            "このアプリを再起動してください。"
        )

    # ========================================================
    # Ollamaその他エラー
    # ========================================================

    elif (
        "[OLLAMA_ERROR]"
        in error_message
        or "[OLLAMA_INVALID_RESPONSE]"
        in error_message
    ):
        message = (
            "ローカルAI（Ollama）の"
            "状態確認中にエラーが発生しました。\n\n"
            "Ollamaを再起動してから、"
            "もう一度このアプリを起動してください。"
        )

    # ========================================================
    # Embeddingモデルがない
    # ========================================================

    elif (
        "[EMBEDDING_MODEL_NOT_FOUND]"
        in error_message
        or "[EMBEDDING_MODEL_INVALID]"
        in error_message
    ):
        message = (
            "検索用AIモデルが見つかりません。\n\n"
            "配布フォルダ内に、次のフォルダが"
            "存在することを確認してください。\n\n"
            "models\\multilingual-e5-base\n\n"
            "アプリ単体ではなく、"
            "配布フォルダを丸ごと使用してください。"
        )

    # ========================================================
    # PDFがない
    # ========================================================

    elif (
        "[PDF_NOT_FOUND]"
        in error_message
    ):
        message = (
            "社員規則PDFが見つかりません。\n\n"
            "配布フォルダ内の"
            "documentsフォルダを確認してください。\n\n"
            f"確認先:\n{PDF_PATH}"
        )

    # ========================================================
    # PDF形式不正
    # ========================================================

    elif (
        "[PDF_INVALID]"
        in error_message
    ):
        message = (
            "社員規則ファイルが"
            "PDF形式ではありません。\n\n"
            "documentsフォルダ内の"
            "社員規則ファイルを確認してください。"
        )

    # ========================================================
    # その他
    # ========================================================

    else:
        message = (
            "アプリの起動中に"
            "エラーが発生しました。\n\n"
            f"{error_message}"
        )

    QMessageBox.critical(
        None,
        "社員規則 AI検索 - 起動エラー",
        message,
    )


def main() -> None:
    """
    GUIアプリケーションを起動する。
    """

    qt_app = QApplication(
        sys.argv
    )

    try:
        (
            rag_service,
            embedding_service,
            vector_store,
        ) = initialize_rag()

    except Exception as error:
        show_startup_error(
            str(error)
        )

        return

    window = MainWindow(
        rag_service=(
            rag_service
        ),
        embedding_service=(
            embedding_service
        ),
        vector_store=(
            vector_store
        ),
    )

    window.show()

    sys.exit(
        qt_app.exec()
    )


if __name__ == "__main__":
    main()