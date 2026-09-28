from __future__ import annotations

import re
from typing import Any


def format_lark_reply(
    result: dict[str, Any],
) -> str:
    """
    RagServiceの結果を、
    Lark Bot向けの返信文に整形する。
    """

    answer = (
        result.get(
            "answer",
            "",
        ).strip()
    )

    references = (
        result.get(
            "references",
            [],
        )
    )

    found = (
        result.get(
            "found",
            False,
        )
    )

    # ========================================================
    # 回答本文を整形
    # ========================================================

    answer = _clean_answer(
        answer
    )

    lines: list[str] = []

    lines.append(
        "【回答】"
    )

    if answer:
        lines.append(
            answer
        )
    else:
        lines.append(
            "回答を生成できませんでした。"
        )

    # ========================================================
    # 規則外質問の場合
    # ========================================================

    if (
        not found
        or _is_not_found_answer(answer)
    ):
        return "\n".join(
            lines
        )

    # ========================================================
    # 参照条文
    # ========================================================

    if references:
        lines.append("")
        lines.append(
            "【参照】"
        )

        for reference in references:
            article = (
                reference.get(
                    "article",
                    "条文情報なし",
                )
            )

            start_page = (
                reference.get(
                    "start_page"
                )
            )

            end_page = (
                reference.get(
                    "end_page"
                )
            )

            page_text = _format_page(
                start_page,
                end_page,
            )

            if page_text:
                lines.append(
                    f"・{article} "
                    f"{page_text}"
                )
            else:
                lines.append(
                    f"・{article}"
                )

    return "\n".join(
        lines
    )


def _clean_answer(
    answer: str,
) -> str:
    """
    回答本文に含まれる参照条文・ページ情報を削除する。

    例:
        第４４条（年次有給休暇）／16～17ページ
        第４４条第９項（16～17ページ）
        (第37条、15ページ)
        第４２条（遅刻・早退及び外出）第1項（16ページ）

    参照情報は【参照】欄にまとめて表示する。
    """

    if not answer:
        return ""

    cleaned = answer.strip()

    # ========================================================
    # 末尾に付く参照表記
    # ========================================================

    patterns = [

        # ----------------------------------------------------
        # 第４４条（年次有給休暇）／16～17ページ
        # 第44条 / 16ページ
        # ----------------------------------------------------
        (
            r"\s*"
            r"第[0-9０-９]+条"
            r"(?:（[^）]*）|\([^)]*\))?"
            r"(?:第?[0-9０-９]+項)?"
            r"\s*[／/]\s*"
            r"[0-9０-９]+"
            r"(?:[～〜~-][0-9０-９]+)?"
            r"\s*ページ"
            r"[。．.]?"
            r"\s*$"
        ),

        # ----------------------------------------------------
        # 第４４条第９項（16～17ページ）
        # 第42条（遅刻・早退及び外出）第1項（16ページ）
        # ----------------------------------------------------
        (
            r"\s*"
            r"第[0-9０-９]+条"
            r"(?:（[^）]*）|\([^)]*\))?"
            r"(?:第?[0-9０-９]+項)?"
            r"\s*"
            r"[（(]"
            r"[0-9０-９]+"
            r"(?:[～〜~-][0-9０-９]+)?"
            r"\s*ページ"
            r"[）)]"
            r"[。．.]?"
            r"\s*$"
        ),

        # ----------------------------------------------------
        # (第37条、15ページ)
        # （第３７条、15ページ）
        # ----------------------------------------------------
        (
            r"\s*"
            r"[（(]"
            r"第[0-9０-９]+条"
            r"(?:（[^）]*）|\([^)]*\))?"
            r"(?:第?[0-9０-９]+項)?"
            r"\s*[、,]\s*"
            r"[0-9０-９]+"
            r"(?:[～〜~-][0-9０-９]+)?"
            r"\s*ページ"
            r"[）)]"
            r"[。．.]?"
            r"\s*$"
        ),
    ]

    # ========================================================
    # パターンを繰り返し除去
    # ========================================================
    #
    # LLMが複数の参照を連続して付ける場合にも対応する。
    # ========================================================

    previous = None

    while previous != cleaned:
        previous = cleaned

        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

    # ========================================================
    # 余分な空白・句読点を整える
    # ========================================================

    cleaned = re.sub(
        r"\s+\n",
        "\n",
        cleaned,
    )

    cleaned = cleaned.strip()

    return cleaned


def _is_not_found_answer(
    answer: str,
) -> bool:
    """
    社員規則から回答できない旨の文章か判定する。

    この場合は検索上位の条文を
    【参照】として表示しない。
    """

    if not answer:
        return True

    normalized = (
        answer
        .replace("。", "")
        .replace("．", "")
        .strip()
    )

    not_found_phrases = [
        "社員規則からは確認できません",
        "社員規則では確認できません",
        "社員規則から確認できません",
        "社員規則には記載されていません",
        "社員規則では確認できない",
        "規則からは確認できません",
        "確認できません",
    ]

    for phrase in not_found_phrases:
        if phrase in normalized:
            return True

    return False


def _format_page(
    start_page: Any,
    end_page: Any,
) -> str:
    """
    ページ番号をLark表示用に整形する。

    例:
        16, 16 -> 16ページ
        16, 17 -> 16～17ページ
    """

    if (
        start_page is None
        or end_page is None
    ):
        return ""

    if start_page == end_page:
        return (
            f"{start_page}ページ"
        )

    return (
        f"{start_page}"
        f"～{end_page}ページ"
    )