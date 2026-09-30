from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FaissVectorStore:
    """
    FAISSインデックスと条文チャンクを管理する。

    Windowsの日本語パス問題を避けるため、
    faiss.read_index / write_index で
    ファイルパスを直接扱わず、
    Python経由でバイトデータを読み書きする。
    """

    def __init__(
        self,
        index_path: Path,
        chunks_path: Path,
    ) -> None:
        self.index_path = index_path
        self.chunks_path = chunks_path

    def create_index(
        self,
        vectors: np.ndarray,
        chunks: list[dict[str, Any]],
    ) -> int:
        """
        文書ベクトルをFAISSへ登録し、
        インデックスとチャンク情報を保存する。
        """

        if vectors.ndim != 2:
            raise ValueError(
                "文書ベクトルは2次元配列である必要があります。"
            )

        if len(vectors) != len(chunks):
            raise ValueError(
                "ベクトル数とチャンク数が一致していません。"
            )

        if len(chunks) == 0:
            raise ValueError(
                "登録するチャンクがありません。"
            )

        vectors = vectors.astype(
            np.float32
        )

        dimension = vectors.shape[1]

        # 正規化済みベクトルに対する内積を
        # コサイン類似度として使用する
        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            vectors
        )

        # 保存フォルダ作成
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chunks_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # FAISSインデックス保存
        #
        # faiss.write_index()へ日本語パスを
        # 直接渡すとWindowsで失敗する場合があるため、
        # 一度バイト配列へ変換してPythonで保存する。
        # ====================================================

        serialized_index = (
            faiss.serialize_index(
                index
            )
        )

        with self.index_path.open(
            "wb"
        ) as file:
            file.write(
                serialized_index.tobytes()
            )

        # ====================================================
        # チャンク情報保存
        # ====================================================

        with self.chunks_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return index.ntotal

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        質問ベクトルに近い条文を検索する。
        """

        if top_k <= 0:
            raise ValueError(
                "top_kは1以上にしてください。"
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                "FAISSインデックスがありません。\n"
                f"{self.index_path}"
            )

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                "チャンク情報がありません。\n"
                f"{self.chunks_path}"
            )

        if query_vector.ndim != 2:
            raise ValueError(
                "質問ベクトルは2次元配列である必要があります。"
            )

        # ====================================================
        # FAISSインデックス読み込み
        #
        # Pythonでファイルを開くことで、
        # Windowsの日本語パス問題を回避する。
        # ====================================================

        with self.index_path.open(
            "rb"
        ) as file:
            index_bytes = file.read()

        index_array = np.frombuffer(
            index_bytes,
            dtype=np.uint8,
        )

        index = faiss.deserialize_index(
            index_array
        )

        # ====================================================
        # チャンク読み込み
        # ====================================================

        with self.chunks_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            chunks = json.load(
                file
            )

        query_vector = (
            query_vector.astype(
                np.float32
            )
        )

        actual_top_k = min(
            top_k,
            index.ntotal,
        )

        scores, indices = index.search(
            query_vector,
            actual_top_k,
        )

        results: list[
            dict[str, Any]
        ] = []

        for score, chunk_index in zip(
            scores[0],
            indices[0],
        ):
            if chunk_index < 0:
                continue

            if chunk_index >= len(chunks):
                continue

            chunk = chunks[
                chunk_index
            ]

            results.append(
                {
                    **chunk,
                    "score": float(
                        score
                    ),
                }
            )

        return results