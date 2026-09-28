from __future__ import annotations

from typing import Any

import requests

from config import (
    EMBEDDING_MODEL_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    PDF_PATH,
)


def check_pdf() -> None:
    """
    社員規則PDFが存在するか確認する。
    """

    if not PDF_PATH.exists():
        raise RuntimeError(
            "[PDF_NOT_FOUND]\n"
            "社員規則PDFが見つかりません。\n\n"
            f"確認先:\n{PDF_PATH}"
        )

    if PDF_PATH.suffix.lower() != ".pdf":
        raise RuntimeError(
            "[PDF_INVALID]\n"
            "社員規則ファイルがPDFではありません。"
        )


def check_embedding_model() -> None:
    """
    ローカルEmbeddingモデルが
    配置されているか確認する。
    """

    if not EMBEDDING_MODEL_PATH.exists():
        raise RuntimeError(
            "[EMBEDDING_MODEL_NOT_FOUND]\n"
            "Embeddingモデルが見つかりません。\n\n"
            f"確認先:\n{EMBEDDING_MODEL_PATH}"
        )

    model_file = (
        EMBEDDING_MODEL_PATH
        / "model.safetensors"
    )

    if not model_file.exists():
        raise RuntimeError(
            "[EMBEDDING_MODEL_INVALID]\n"
            "Embeddingモデルのファイルが不足しています。\n\n"
            f"確認先:\n{model_file}"
        )


def check_ollama_server() -> None:
    """
    Ollamaサーバーが起動しているか確認する。
    """

    url = (
        f"{OLLAMA_BASE_URL.rstrip('/')}"
        "/api/tags"
    )

    try:
        response = requests.get(
            url,
            timeout=5,
        )

        response.raise_for_status()

    except requests.ConnectionError as error:
        raise RuntimeError(
            "[OLLAMA_CONNECTION_ERROR]\n"
            "Ollamaに接続できません。"
        ) from error

    except requests.Timeout as error:
        raise RuntimeError(
            "[OLLAMA_TIMEOUT]\n"
            "Ollamaへの接続がタイムアウトしました。"
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            "[OLLAMA_ERROR]\n"
            "Ollamaの状態確認中に"
            "エラーが発生しました。"
        ) from error


def get_ollama_models() -> list[str]:
    """
    Ollamaに登録されているモデル一覧を取得する。
    """

    url = (
        f"{OLLAMA_BASE_URL.rstrip('/')}"
        "/api/tags"
    )

    try:
        response = requests.get(
            url,
            timeout=5,
        )

        response.raise_for_status()

        data: dict[str, Any] = (
            response.json()
        )

    except requests.RequestException as error:
        raise RuntimeError(
            "[OLLAMA_ERROR]\n"
            "Ollamaのモデル一覧を"
            "取得できませんでした。"
        ) from error

    except ValueError as error:
        raise RuntimeError(
            "[OLLAMA_INVALID_RESPONSE]\n"
            "Ollamaから不正な応答が返されました。"
        ) from error

    models = data.get(
        "models",
        [],
    )

    model_names: list[str] = []

    for model in models:
        if not isinstance(
            model,
            dict,
        ):
            continue

        name = model.get(
            "name"
        )

        if isinstance(
            name,
            str,
        ):
            model_names.append(
                name
            )

    return model_names


def check_ollama_model() -> None:
    """
    使用するLLMモデルが
    インストールされているか確認する。
    """

    model_names = (
        get_ollama_models()
    )

    if OLLAMA_MODEL_NAME in model_names:
        return

    matching_models = [
        model_name
        for model_name in model_names
        if model_name.startswith(
            OLLAMA_MODEL_NAME
        )
    ]

    if matching_models:
        return

    raise RuntimeError(
        "[OLLAMA_MODEL_NOT_FOUND]\n"
        f"{OLLAMA_MODEL_NAME}が"
        "インストールされていません。"
    )


def run_startup_checks() -> None:
    """
    アプリ起動前チェック。
    """

    check_pdf()

    check_embedding_model()

    check_ollama_server()

    check_ollama_model()