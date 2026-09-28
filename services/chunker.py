from __future__ import annotations

import re
from typing import Any


ARTICLE_HEADING_PATTERN = re.compile(
    r"^第[０-９0-9]+条(?:（[^）\n]+）)?\s*$",
    re.MULTILINE,
)

ARTICLE_NAME_PATTERN = re.compile(
    r"^(第[０-９0-9]+条(?:（[^）\n]+）)?)"
)


def split_pages_into_articles(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    PDF本文を全ページ連結し、条文単位でチャンク分割する。

    目次にある「第○条」は除外し、
    ページをまたぐ条文は1つのチャンクへまとめる。

    Args:
        pages:
            PDFから抽出したページ情報。

            例:
            [
                {
                    "page": 1,
                    "text": "ページ本文"
                }
            ]

    Returns:
        条文単位に分割したチャンク一覧。

        例:
        [
            {
                "chunk_id": 1,
                "page": 7,
                "start_page": 7,
                "end_page": 7,
                "article": "第１条（目的）",
                "text": "第１条（目的）..."
            }
        ]
    """
    if not pages:
        raise ValueError("ページデータがありません。")

    full_text, page_positions = _combine_pages(pages)

    matches = list(
        ARTICLE_HEADING_PATTERN.finditer(full_text)
    )

    if not matches:
        raise ValueError(
            "本文から条文見出しを検出できませんでした。"
        )

    chunks: list[dict[str, Any]] = []
    chunk_id = 1

    for index, match in enumerate(matches):
        start_position = match.start()

        if index + 1 < len(matches):
            end_position = matches[index + 1].start()
        else:
            end_position = len(full_text)

        article_text = _clean_text(
            full_text[start_position:end_position]
        )

        article_name = _extract_article_name(
            article_text
        )

        if article_name is None:
            continue

        # 目次のような短いデータを除外
        if _is_table_of_contents_entry(article_text):
            continue

        start_page = _find_page_number(
            start_position,
            page_positions,
        )

        end_page = _find_page_number(
            max(start_position, end_position - 1),
            page_positions,
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "page": start_page,
                "start_page": start_page,
                "end_page": end_page,
                "article": article_name,
                "text": article_text,
            }
        )

        chunk_id += 1

    if not chunks:
        raise ValueError(
            "有効な条文チャンクを作成できませんでした。"
        )

    return chunks


def _combine_pages(
    pages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, int]]]:
    """
    全ページの文章を結合し、
    各ページが全体テキストのどこにあるかを記録する。

    Returns:
        結合した全文とページ位置情報。
    """
    text_parts: list[str] = []
    page_positions: list[dict[str, int]] = []

    current_position = 0

    for page_data in pages:
        page_number = int(page_data["page"])
        page_text = str(
            page_data["text"]
        ).strip()

        if not page_text:
            continue

        cleaned_page_text = _clean_page_text(
            page_text
        )

        if not cleaned_page_text:
            continue

        page_start = current_position

        text_parts.append(cleaned_page_text)

        current_position += len(cleaned_page_text)

        page_positions.append(
            {
                "page": page_number,
                "start": page_start,
                "end": current_position,
            }
        )

        # ページ同士が直接つながらないように改行を追加
        text_parts.append("\n")
        current_position += 1

    return "".join(text_parts), page_positions


def _find_page_number(
    position: int,
    page_positions: list[dict[str, int]],
) -> int:
    """
    全体テキスト上の文字位置からPDFページ番号を取得する。

    ページ間に追加した改行位置だった場合は、
    直前のページ番号を返す。
    """
    if not page_positions:
        return 0

    previous_page = page_positions[0]["page"]

    for page_data in page_positions:
        if (
            page_data["start"]
            <= position
            < page_data["end"]
        ):
            return page_data["page"]

        if position < page_data["start"]:
            return previous_page

        previous_page = page_data["page"]

    return page_positions[-1]["page"]


def _extract_article_name(
    text: str,
) -> str | None:
    """
    条文名を取得する。

    例:
        第４４条（年次有給休暇）
    """
    match = ARTICLE_NAME_PATTERN.match(text)

    if match is None:
        return None

    return match.group(1)


def _is_table_of_contents_entry(
    text: str,
) -> bool:
    """
    目次に記載されている条文項目かどうかを判定する。

    例:
        第１条（目的）
        7
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return True

    # 見出しとページ番号だけの場合は目次とみなす
    if len(lines) <= 3:
        remaining_lines = lines[1:]

        if remaining_lines and all(
            re.fullmatch(
                r"[０-９0-9]+",
                line,
            )
            for line in remaining_lines
        ):
            return True

    # 見出しを除いた本文が極端に短い場合も除外
    body_text = "".join(lines[1:])

    if len(body_text) < 20:
        return True

    return False


def _clean_page_text(
    text: str,
) -> str:
    """
    ページ単位の不要な情報を除去する。

    ページ番号だけの行や空行を除外する。
    """
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            continue

        # ページ番号だけの行を除外
        if re.fullmatch(
            r"[０-９0-9]+",
            stripped_line,
        ):
            continue

        cleaned_lines.append(stripped_line)

    return "\n".join(cleaned_lines)


def _clean_text(
    text: str,
) -> str:
    """
    条文内の不要な空行や行末空白を整える。
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)