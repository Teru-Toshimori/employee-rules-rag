from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config import (
    CHUNKING_VERSION,
    OPENAI_CHUNKS_PATH,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_FAISS_INDEX_PATH,
    OPENAI_INDEX_METADATA_PATH,
)


def calculate_file_hash(
    file_path: Path,
) -> str:
    """
    ファイルのSHA-256ハッシュ値を計算する。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"ファイルが見つかりません: {file_path}"
        )

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def is_openai_index_cache_valid(
    pdf_path: Path,
) -> bool:
    """
    OpenAI Embeddingで作成した
    FAISSインデックスを再利用できるか確認する。
    """

    required_files = [
        OPENAI_FAISS_INDEX_PATH,
        OPENAI_CHUNKS_PATH,
        OPENAI_INDEX_METADATA_PATH,
    ]

    if not all(
        path.exists()
        for path in required_files
    ):
        return False

    try:
        with OPENAI_INDEX_METADATA_PATH.open(
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

    if saved_model != OPENAI_EMBEDDING_MODEL:
        return False

    if (
        saved_chunking_version
        != CHUNKING_VERSION
    ):
        return False

    return True


def save_openai_index_metadata(
    pdf_path: Path,
    chunk_count: int,
) -> None:
    """
    OpenAI Embeddingで作成した
    FAISSインデックスの情報を保存する。
    """

    OPENAI_INDEX_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata: dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "pdf_hash": calculate_file_hash(
            pdf_path
        ),
        "embedding_model": (
            OPENAI_EMBEDDING_MODEL
        ),
        "chunking_version": (
            CHUNKING_VERSION
        ),
        "chunk_count": chunk_count,
    }

    with OPENAI_INDEX_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )