from pathlib import Path

import fitz


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    PDFからページごとに文章を抽出する。

    戻り値:
    [
        {
            "page": 1,
            "text": "1ページ目の文章"
        },
        {
            "page": 2,
            "text": "2ページ目の文章"
        }
    ]
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDFが見つかりません: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"PDFファイルを指定してください: {pdf_path}"
        )

    pages: list[dict] = []

    try:
        with fitz.open(pdf_path) as document:
            if document.page_count == 0:
                raise ValueError("PDFにページがありません。")

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                text = page.get_text("text").strip()

                # 文字を取得できない空ページは除外
                if not text:
                    continue

                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    except fitz.FileDataError as error:
        raise ValueError(
            f"PDFを読み込めませんでした: {pdf_path}"
        ) from error

    if not pages:
        raise ValueError(
            "PDFから文字を抽出できませんでした。"
            "スキャンPDFの場合はOCRが必要です。"
        )

    return pages