from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config import (
    CHUNKING_VERSION,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
)


def calculate_file_hash(
    file_path: Path,
) -> str:
    """
    ファイルのSHA-256ハッシュ値を計算する。

    PDFの内容が変更されたか判定するために使用する。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"ファイルが見つかりません: {file_path}"
        )

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def is_index_cache_valid(
    pdf_path: Path,
) -> bool:
    """
    保存済みFAISSインデックスを
    再利用できるか確認する。

    以下が一致している場合のみ再利用する。

    ・PDFの内容
    ・Embeddingモデル
    ・チャンク分割バージョン
    """

    # 必要ファイルがなければキャッシュ無効
    required_files = [
        FAISS_INDEX_PATH,
        CHUNKS_PATH,
        INDEX_METADATA_PATH,
    ]

    if not all(
        path.exists()
        for path in required_files
    ):
        return False

    try:
        with INDEX_METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return False

    current_pdf_hash = calculate_file_hash(
        pdf_path
    )

    saved_pdf_hash = metadata.get(
        "pdf_hash"
    )

    saved_model = metadata.get(
        "embedding_model"
    )

    saved_chunking_version = metadata.get(
        "chunking_version"
    )

    if saved_pdf_hash != current_pdf_hash:
        return False

    if saved_model != EMBEDDING_MODEL_NAME:
        return False

    if (
        saved_chunking_version
        != CHUNKING_VERSION
    ):
        return False

    return True


def save_index_metadata(
    pdf_path: Path,
    chunk_count: int,
) -> None:
    """
    作成したFAISSインデックスの情報を保存する。
    """

    INDEX_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata: dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "pdf_hash": calculate_file_hash(
            pdf_path
        ),
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "chunking_version": (
            CHUNKING_VERSION
        ),
        "chunk_count": chunk_count,
    }

    with INDEX_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )