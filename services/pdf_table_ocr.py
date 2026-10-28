
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import fitz
from paddleocr import PPStructureV3


# ============================================================
# 設定
# ============================================================

OCR_SCALE = 3.0
MIN_IMAGE_AREA_RATIO = 0.002

ARTICLE_PATTERN = re.compile(
    r"第\s*([0-9０-９]+)\s*条"
)

TABLE_PATTERN = re.compile(
    r"<table\b.*?</table>",
    re.IGNORECASE | re.DOTALL,
)

TR_PATTERN = re.compile(
    r"<tr\b[^>]*>(.*?)</tr>",
    re.IGNORECASE | re.DOTALL,
)

CELL_PATTERN = re.compile(
    r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
    re.IGNORECASE | re.DOTALL,
)

TAG_PATTERN = re.compile(r"<[^>]+>")

DIGIT_TABLE = str.maketrans(
    "０１２３４５６７８９",
    "0123456789",
)


# ============================================================
# 条文番号
# ============================================================

def get_article_number(text: str) -> int | None:
    match = ARTICLE_PATTERN.search(text)

    if match is None:
        return None

    return int(
        match.group(1).translate(DIGIT_TABLE)
    )


# ============================================================
# HTML表 → テキスト
# ============================================================

def html_table_to_text(html: str) -> str:
    """
    PaddleOCRのHTML表を検索用テキストに変換する。

    単純な表は行単位で変換する。
    行列の結合がある複雑な表は別途検証が必要。
    """

    import html as html_module

    rows = []

    for row_html in TR_PATTERN.findall(html):
        cells = []

        for cell_html in CELL_PATTERN.findall(
            row_html
        ):
            text = TAG_PATTERN.sub(
                " ",
                cell_html,
            )

            text = html_module.unescape(text)
            text = " ".join(text.split())
            text = normalize_ocr_table_text(text)

            cells.append(text)

        if cells:
            rows.append(cells)

    if not rows:
        return ""

    lines = ["【OCR抽出表データ】"]

    # 横方向の2行表を縦方向に展開
    # 例: 勤続年数 / 付与日数
    if (
        len(rows) == 2
        and len(rows[0]) == len(rows[1])
        and len(rows[0]) > 2
    ):
        label1 = rows[0][0]
        label2 = rows[1][0]

        for value1, value2 in zip(
            rows[0][1:],
            rows[1][1:],
        ):
            lines.append(
                f"{label1}：{value1} → "
                f"{label2}：{value2}"
            )

    else:
        for row in rows:
            lines.append(
                " | ".join(row)
            )

    return "\n".join(lines)


# ============================================================
# 画像の有無を検出
# ============================================================

def has_significant_image(page: Any) -> bool:
    """
    ページ内に一定面積以上の画像があるか確認。

    小さなアイコン等を除外するため、
    面積比率で判定する。
    """

    page_area = (
        page.rect.width * page.rect.height
    )

    if page_area <= 0:
        return False

    for image_info in page.get_images(full=True):
        xref = image_info[0]

        for rect in page.get_image_rects(xref):
            image_area = (
                rect.width * rect.height
            )

            if (
                image_area / page_area
                >= MIN_IMAGE_AREA_RATIO
            ):
                return True

    return False


# ============================================================
# ページ内の条文見出し位置を取得
# ============================================================

def get_article_positions(
    page: Any,
) -> list[dict[str, Any]]:
    """
    PDFのネイティブ文字情報から
    条文見出しのY座標を取得する。
    """

    result = []

    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue

        for line in block["lines"]:
            spans = line.get("spans", [])

            text = "".join(
                span.get("text", "")
                for span in spans
            ).strip()

            # 本文中の「第○条」参照を誤検出しないよう
            # 見出し行のみ対象とする
            if not re.match(
                r"^第\s*[0-9０-９]+\s*条"
                r"(?:[（(].*[）)])?\s*$",
                text,
            ):
                continue

            article_number = get_article_number(
                text
            )

            if article_number is None:
                continue

            result.append({
                "article_number": article_number,
                "y": float(line["bbox"][1]),
                "text": text,
            })

    return sorted(
        result,
        key=lambda item: item["y"],
    )


# ============================================================
# OCR結果の表データ抽出
# ============================================================

def extract_table_blocks(
    ocr_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    PP-StructureV3の解析結果から
    表ブロックのみを取得する。

    想定するJSON構造:
    res.parsing_res_list
    """

    root = ocr_json.get("res", ocr_json)

    blocks = root.get(
        "parsing_res_list", []
    )

    if not isinstance(blocks, list):
        return []

    tables = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        label = str(
            block.get("block_label", "")
        ).lower()

        content = str(
            block.get("block_content", "")
        )

        if (
            label != "table"
            and "<table" not in content.lower()
        ):
            continue

        html_match = TABLE_PATTERN.search(
            content
        )

        if html_match is None:
            continue

        html = html_match.group(0)

        table_text = html_table_to_text(
            html
        )

        if not table_text:
            continue

        bbox = block.get("block_bbox")

        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
        ):
            bbox = None

        tables.append({
            "text": table_text,
            "html": html,
            "bbox": bbox,
        })

    return tables


# ============================================================
# 表が属する条文を判定
# ============================================================

def find_article_for_table(
    table_bbox: list | tuple | None,
    article_positions: list[dict[str, Any]],
    scale: float,
) -> int | None:
    """
    表の上端より前にある直近の条文見出しを取得。

    OCR座標は画像ピクセル単位、
    PDF座標はポイント単位なので変換する。
    """

    if table_bbox is None:
        return None

    if not article_positions:
        return None

    table_y = float(table_bbox[1]) / scale

    candidates = [
        article
        for article in article_positions
        if article["y"] <= table_y
    ]

    if not candidates:
        return None

    return candidates[-1]["article_number"]


# ============================================================
# PDFから表を抽出
# ============================================================

def extract_pdf_tables(
    pdf_path: str | Path,
    target_pages: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    PDF内の画像表を自動認識する。

    Args:
        pdf_path:
            対象PDF

        target_pages:
            対象ページ番号（1始まり）
            Noneの場合は全ページを確認

    Returns:
        OCR表データ一覧
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDFが見つかりません: {pdf_path}"
        )

    print("PaddleOCRを初期化しています...")

    pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    extracted_tables = []

    with fitz.open(str(pdf_path)) as pdf:

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            if (
                target_pages is not None
                and page_number not in target_pages
            ):
                continue

            if not has_significant_image(page):
                print(
                    f"Page {page_number}: "
                    "画像なし・スキップ"
                )
                continue

            print(
                f"Page {page_number}: OCR実行中..."
            )

            article_positions = (
                get_article_positions(page)
            )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    OCR_SCALE,
                    OCR_SCALE,
                ),
                alpha=False,
            )

            with tempfile.TemporaryDirectory() as tmp:
                image_path = (
                    Path(tmp) / "page.png"
                )

                pix.save(str(image_path))

                results = pipeline.predict(
                    input=str(image_path)
                )

                for result in results:
                    data = result.json

                    if callable(data):
                        data = data()

                    if not isinstance(data, dict):
                        continue

                    tables = extract_table_blocks(
                        data
                    )

                    for table in tables:
                        article_number = (
                            find_article_for_table(
                                table["bbox"],
                                article_positions,
                                OCR_SCALE,
                            )
                        )

                        extracted_tables.append({
                            "page": page_number,
                            "article_number": article_number,
                            "text": table["text"],
                            "bbox": table["bbox"],
                        })

            print(
                f"Page {page_number}: "
                "OCR処理完了"
            )

    return extracted_tables


# ============================================================
# ChunkにOCR表データを追加
# ============================================================

def merge_ocr_tables_into_chunks(
    chunks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    条文番号を基にOCR表データを追加する。

    所属条文を確定できない表は追加しない。
    """

    merged_chunks = [
        dict(chunk)
        for chunk in chunks
    ]

    skipped_tables = []

    article_map = {}

    for chunk in merged_chunks:
        number = get_article_number(
            str(chunk.get("article", ""))
        )

        if number is not None:
            article_map.setdefault(
                number, []
            ).append(chunk)

    for table in tables:
        number = table.get(
            "article_number"
        )

        candidates = article_map.get(
            number, []
        )

        # 同じ条番号が複数ある場合も誤追加を防止
        candidates = [
            chunk
            for chunk in candidates
            if (
                int(chunk.get("start_page", 0))
                <= table["page"]
                <= int(chunk.get("end_page", 0))
            )
        ]

        if len(candidates) != 1:
            skipped_tables.append(table)
            continue

        chunk = candidates[0]
        table_text = table["text"]

        if table_text not in chunk["text"]:
            chunk["text"] += (
                "\n\n" + table_text
            )

            print(
                f"第{number}条にOCR表を追加: "
                f"Page {table['page']}"
            )

    return merged_chunks, skipped_tables
def normalize_ocr_table_text(text: str) -> str:
    """
    OCRで認識した勤続年数表記を補正する。
    """

    text = re.sub(
        r"(\d+)\s*年\s*(\d+)\s*[力ヵかヶカ]月",
        r"\1年\2か月",
        text,
    )

    text = re.sub(
        r"(\d+)\s*[力ヵかヶカ]月",
        r"\1か月",
        text,
    )

    return text