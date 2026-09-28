from pathlib import Path
import os
import sys

from dotenv import load_dotenv


def get_base_dir() -> Path:
    """
    開発時とexe実行時の両方で、
    アプリ本体のフォルダを取得する。
    """

    if getattr(sys, "frozen", False):
        return Path(
            sys.executable
        ).resolve().parent

    return Path(
        __file__
    ).resolve().parent


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = get_base_dir()

# ------------------------------------------------------------
# 環境変数
# ------------------------------------------------------------
# ローカル開発時はプロジェクト直下の .env を読み込む。
#
# Cloud RunではCloud Run側に設定されている環境変数を使用する。
# load_dotenv() はデフォルトで既存の環境変数を上書きしない。
# ------------------------------------------------------------

ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)

DOCUMENTS_DIR = (
    BASE_DIR
    / "documents"
)

DATA_DIR = (
    BASE_DIR
    / "data"
)

PDF_PATH = (
    DOCUMENTS_DIR
    / "5_02_社員就業規則_改訂I_20240701.pdf"
)


# ============================================================
# FAISS保存設定
# ============================================================

FAISS_INDEX_PATH = (
    DATA_DIR
    / "index.faiss"
)

CHUNKS_PATH = (
    DATA_DIR
    / "chunks.json"
)

INDEX_METADATA_PATH = (
    DATA_DIR
    / "index_metadata.json"
)

DOCUMENT_METADATA_PATH = (
    DATA_DIR
    / "document_metadata.json"
)


# ============================================================
# Embedding設定
# ============================================================

EMBEDDING_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "multilingual-e5-base"
)

EMBEDDING_MODEL_NAME = (
    "multilingual-e5-base"
)

EMBEDDING_BATCH_SIZE = 8


# ============================================================
# Ollama設定
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).strip()

OLLAMA_MODEL_NAME = (
    "qwen3:4b"
)

OLLAMA_TIMEOUT = 180


# ============================================================
# 検索設定
# ============================================================

SEARCH_TOP_K = 3

SEARCH_SCORE_THRESHOLD = 0.80


# ============================================================
# キャッシュ設定
# ============================================================

CHUNKING_VERSION = (
    "article-v1"
)


# ============================================================
# OpenAI API 設定
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

# 回答生成モデル
OPENAI_CHAT_MODEL = "gpt-5-mini"

# Embeddingモデル
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# OpenAI API タイムアウト（秒）
OPENAI_TIMEOUT = 60

# OpenAI Embedding版FAISS
OPENAI_FAISS_INDEX_PATH = (
    DATA_DIR / "openai_index.faiss"
)

OPENAI_CHUNKS_PATH = (
    DATA_DIR / "openai_chunks.json"
)

OPENAI_INDEX_METADATA_PATH = (
    DATA_DIR / "openai_index_metadata.json"
)