from __future__ import annotations

from config import (
    OPENAI_CHUNKS_PATH,
    OPENAI_FAISS_INDEX_PATH,
    PDF_PATH,
)
from services.chunker import (
    split_pages_into_articles,
)
from services.openai_embedding import (
    OpenAIEmbeddingService,
)
from services.openai_index_cache import (
    is_openai_index_cache_valid,
    save_openai_index_metadata,
)
from services.pdf_reader import (
    extract_pdf_pages,
)
from services.vector_store import (
    FaissVectorStore,
)


def create_openai_index() -> None:
    """
    社員規則PDFからOpenAI Embedding用の
    FAISSインデックスを作成する。
    """

    print("=" * 60)
    print("OpenAI FAISSインデックス作成")
    print("=" * 60)

    print()
    print(f"対象PDF: {PDF_PATH}")

    # ========================================================
    # 1. キャッシュ確認
    # ========================================================

    print()
    print("[1/5] 既存インデックスを確認しています...")

    if is_openai_index_cache_valid(
        PDF_PATH
    ):
        print(
            "      有効なOpenAI用インデックスが"
            "すでに存在します。"
        )
        print()
        print("再作成は不要です。")
        return

    print(
        "      OpenAI用インデックスを"
        "新しく作成します。"
    )

    # ========================================================
    # 2. PDF読み込み
    # ========================================================

    print()
    print("[2/5] PDFを読み込んでいます...")

    pages = extract_pdf_pages(
        PDF_PATH
    )

    print(
        f"      PDF読み込み完了: "
        f"{len(pages)}ページ"
    )

    # ========================================================
    # 3. チャンク分割
    # ========================================================

    print()
    print("[3/5] 条文単位に分割しています...")

    chunks = split_pages_into_articles(
        pages
    )

    print(
        f"      チャンク作成完了: "
        f"{len(chunks)}件"
    )

    if not chunks:
        raise RuntimeError(
            "チャンクを作成できませんでした。"
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # ========================================================
    # 4. OpenAI Embedding
    # ========================================================

    print()
    print(
        "[4/5] OpenAI Embedding APIで"
        "ベクトル化しています..."
    )

    embedding_service = (
        OpenAIEmbeddingService()
    )

    vectors = (
        embedding_service.embed_documents(
            texts
        )
    )

    print(
        f"      ベクトル化完了: "
        f"{vectors.shape}"
    )

    # ========================================================
    # 5. FAISS保存
    # ========================================================

    print()
    print(
        "[5/5] FAISSインデックスを"
        "保存しています..."
    )

    vector_store = FaissVectorStore(
        index_path=OPENAI_FAISS_INDEX_PATH,
        chunks_path=OPENAI_CHUNKS_PATH,
    )

    total_count = (
        vector_store.create_index(
            vectors=vectors,
            chunks=chunks,
        )
    )

    save_openai_index_metadata(
        pdf_path=PDF_PATH,
        chunk_count=len(chunks),
    )

    print(
        f"      FAISS登録完了: "
        f"{total_count}件"
    )

    print()
    print("=" * 60)
    print("OpenAI FAISSインデックス作成完了")
    print("=" * 60)

    print()
    print(
        f"FAISS : "
        f"{OPENAI_FAISS_INDEX_PATH}"
    )

    print(
        f"Chunks: "
        f"{OPENAI_CHUNKS_PATH}"
    )


def main() -> None:
    create_openai_index()


if __name__ == "__main__":
    main()