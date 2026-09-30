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
# load_dotenv() は既存の環境変数を上書きしない。
# ------------------------------------------------------------

ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


# ============================================================
# ディレクトリ設定
# ============================================================

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
# 検索設定
# ============================================================

SEARCH_TOP_K = 3


# ============================================================
# チャンク設定
# ============================================================

CHUNKING_VERSION = (
    "article-v1"
)


# ============================================================
# OpenAI API設定
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

# 回答生成モデル
OPENAI_CHAT_MODEL = (
    "gpt-5-mini"
)

# Embeddingモデル
OPENAI_EMBEDDING_MODEL = (
    "text-embedding-3-small"
)

# OpenAI APIタイムアウト（秒）
OPENAI_TIMEOUT = 60


# ============================================================
# OpenAI Embedding版FAISS設定
# ============================================================

OPENAI_FAISS_INDEX_PATH = (
    DATA_DIR
    / "openai_index.faiss"
)

OPENAI_CHUNKS_PATH = (
    DATA_DIR
    / "openai_chunks.json"
)

OPENAI_INDEX_METADATA_PATH = (
    DATA_DIR
    / "openai_index_metadata.json"
)