from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DOCUMENT_METADATA_PATH


def create_temporary_pdf_copy(
    source_path: Path,
) -> Path:
    """
    選択されたPDFを一時領域へコピーする。

    本番PDFを直接上書きせず、
    まず一時ファイルで検証するために使用する。
    """

    if not source_path.exists():
        raise FileNotFoundError(
            f"選択したPDFが見つかりません: {source_path}"
        )

    if source_path.suffix.lower() != ".pdf":
        raise ValueError(
            "PDFファイルを選択してください。"
        )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="local_rag_pdf_"
        )
    )

    temp_pdf_path = (
        temp_dir
        / source_path.name
    )

    shutil.copy2(
        source_path,
        temp_pdf_path,
    )

    return temp_pdf_path


def replace_pdf(
    source_path: Path,
    destination_path: Path,
) -> None:
    """
    検証済みPDFを本番PDFへ反映する。
    """

    if not source_path.exists():
        raise FileNotFoundError(
            f"PDFが見つかりません: {source_path}"
        )

    if source_path.suffix.lower() != ".pdf":
        raise ValueError(
            "PDFファイルを指定してください。"
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_path,
        destination_path,
    )


def save_document_metadata(
    source_filename: str,
) -> None:
    """
    現在使用している社員規則PDFの
    元ファイル名と更新日時を保存する。
    """

    DOCUMENT_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "source_filename": source_filename,
        "updated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    with DOCUMENT_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_document_metadata() -> dict[str, Any]:
    """
    保存済みの社員規則PDF情報を読み込む。
    """

    if not DOCUMENT_METADATA_PATH.exists():
        return {}

    try:
        with DOCUMENT_METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def cleanup_temporary_pdf(
    temp_pdf_path: Path,
) -> None:
    """
    一時PDFと一時フォルダを削除する。
    """

    try:
        temp_dir = temp_pdf_path.parent

        if temp_dir.exists():
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

    except OSError:
        pass