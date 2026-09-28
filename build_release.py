from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"

APP_NAME = "社員規則AI検索"

APP_DIST_DIR = DIST_DIR / APP_NAME

SOURCE_MODELS_DIR = BASE_DIR / "models"
SOURCE_DOCUMENTS_DIR = BASE_DIR / "documents"
SOURCE_DATA_DIR = BASE_DIR / "data"


def run_pyinstaller() -> None:
    """
    PyInstallerでアプリをビルドする。
    """

    command = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "app.py",
    ]

    print("=" * 60)
    print("PyInstallerビルド開始")
    print("=" * 60)

    subprocess.run(
        command,
        cwd=BASE_DIR,
        check=True,
    )

    print()
    print("PyInstallerビルド完了")


def copy_directory(
    source: Path,
    destination: Path,
) -> None:
    """
    フォルダを配布先へコピーする。

    既存フォルダがある場合は一度削除してからコピーする。
    """

    if not source.exists():
        raise FileNotFoundError(
            f"コピー元フォルダがありません: {source}"
        )

    if destination.exists():
        shutil.rmtree(
            destination
        )

    shutil.copytree(
        source,
        destination,
    )


def copy_runtime_data() -> None:
    """
    models / documents / data を
    exeと同じ階層へコピーする。
    """

    print()
    print("=" * 60)
    print("配布データコピー")
    print("=" * 60)

    copy_directory(
        SOURCE_MODELS_DIR,
        APP_DIST_DIR / "models",
    )

    print("models コピー完了")

    copy_directory(
        SOURCE_DOCUMENTS_DIR,
        APP_DIST_DIR / "documents",
    )

    print("documents コピー完了")

    copy_directory(
        SOURCE_DATA_DIR,
        APP_DIST_DIR / "data",
    )

    print("data コピー完了")


def remove_unnecessary_files() -> None:
    """
    配布先に誤って作られた二重フォルダなどを除去する。
    """

    nested_data = (
        APP_DIST_DIR
        / "data"
        / "data"
    )

    if nested_data.exists():
        shutil.rmtree(
            nested_data
        )

    nested_models = (
        APP_DIST_DIR
        / "models"
        / "models"
    )

    if nested_models.exists():
        shutil.rmtree(
            nested_models
        )

    nested_documents = (
        APP_DIST_DIR
        / "documents"
        / "documents"
    )

    if nested_documents.exists():
        shutil.rmtree(
            nested_documents
        )


def verify_release() -> None:
    """
    配布に必要なファイルが存在するか確認する。
    """

    required_paths = [
        APP_DIST_DIR / f"{APP_NAME}.exe",

        APP_DIST_DIR
        / "models"
        / "multilingual-e5-base"
        / "model.safetensors",

        APP_DIST_DIR
        / "documents",

        APP_DIST_DIR
        / "data"
        / "index.faiss",

        APP_DIST_DIR
        / "data"
        / "chunks.json",

        APP_DIST_DIR
        / "data"
        / "index_metadata.json",

        APP_DIST_DIR
        / "data"
        / "document_metadata.json",
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise RuntimeError(
            "配布ファイルが不足しています。\n\n"
            f"{missing_text}"
        )


def show_release_summary() -> None:
    """
    完成した配布フォルダを表示する。
    """

    print()
    print("=" * 60)
    print("配布パッケージ作成完了")
    print("=" * 60)

    print(
        f"出力先:\n{APP_DIST_DIR}"
    )

    print()
    print("配布時は、このフォルダを丸ごと渡してください。")

    print()
    print("配布先PCで別途必要:")
    print("・Ollama")
    print("・qwen3:4b")


def main() -> None:
    try:
        run_pyinstaller()

        copy_runtime_data()

        remove_unnecessary_files()

        verify_release()

        show_release_summary()

    except (
        FileNotFoundError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print()
        print("=" * 60)
        print("配布パッケージ作成失敗")
        print("=" * 60)
        print(error)


if __name__ == "__main__":
    main()