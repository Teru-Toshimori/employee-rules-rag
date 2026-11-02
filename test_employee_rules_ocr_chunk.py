
from pathlib import Path
from datetime import datetime
import json

from services.pdf_reader import extract_pdf_pages
from services.chunker import split_pages_into_articles
from services.pdf_table_ocr import (
    extract_pdf_tables,
    merge_ocr_tables_into_chunks,
    get_article_number,
)

# ===== テスト設定 =====
PDF_PATH = Path(
    "documents/5_02_社員就業規則_改訂I_20240701.pdf"
)

OUTPUT_DIR = Path("doc/test_results")
TARGET_ARTICLE = 44


def find_article_chunk(chunks, article_number):
    matches = [
        chunk
        for chunk in chunks
        if get_article_number(str(chunk.get("article", "")))
        == article_number
    ]
    return matches


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.is_file():
        print(f"PDFが見つかりません: {PDF_PATH}")
        return

    print("=== 社員就業規則 OCR・Chunk統合テスト ===")

    start_time = datetime.now()

    # 1. PDF読み込み
    pages = extract_pdf_pages(PDF_PATH)
    print(f"PDFページ数: {len(pages)}")

    # 2. 条文単位でChunk分割
    chunks = split_pages_into_articles(pages)
    print(f"Chunk数: {len(chunks)}")

    # 3. 全ページOCR処理
    tables = extract_pdf_tables(
        pdf_path=PDF_PATH,
        target_pages=None,
    )
    print(f"OCR抽出テーブル数: {len(tables)}")

    # 4. OCR結果をChunkに統合
    merged_chunks, skipped_tables = (
        merge_ocr_tables_into_chunks(chunks, tables)
    )

    print(f"統合対象外テーブル数: {len(skipped_tables)}")

    # 5. 第44条Chunkを比較
    before = find_article_chunk(chunks, TARGET_ARTICLE)
    after = find_article_chunk(merged_chunks, TARGET_ARTICLE)

    print(f"第44条Chunk数（統合前）: {len(before)}")
    print(f"第44条Chunk数（統合後）: {len(after)}")

    article_tables = [
        table
        for table in tables
        if table.get("article_number") == TARGET_ARTICLE
    ]

    merged_count = 0
    if len(after) == 1:
        for table in article_tables:
            if (
                table not in skipped_tables
                and table["text"] in after[0]["text"]
                and (
                    len(before) != 1
                    or table["text"] not in before[0]["text"]
                )
            ):
                merged_count += 1

    print(f"第44条OCR表数: {len(article_tables)}")
    print(f"第44条新規統合確認数: {merged_count}")
# 6. OCR抽出内容を表示・保存
    print("\n=== 第44条 OCR抽出内容 ===")

    ocr_texts = []

    for i, table in enumerate(article_tables, start=1):
        text = table.get("text", "")
        print(f"\n--- Table {i} / Page {table['page']} ---")
        print(text)
        ocr_texts.append(text)

    ocr_output_path = (
        OUTPUT_DIR / "ocr_employee_rules_extracted.txt"
    )

    ocr_output_path.write_text(
        "\n\n".join(ocr_texts),
        encoding="utf-8",
    )

    print(f"OCR抽出内容保存先: {ocr_output_path}")
   
    elapsed = (datetime.now() - start_time).total_seconds()

    # 7. 結果を保存
    summary = {
        "pdf": PDF_PATH.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "ocr_table_count": len(tables),
        "skipped_table_count": len(skipped_tables),
        "article_44_table_count": len(article_tables),
        "article_44_newly_merged_count": merged_count,
        "article_44_merge_success": (
            len(before) == 1
            and len(after) == 1
            and len(article_tables) > 0
            and merged_count == len(article_tables)
        ),
        "elapsed_seconds": round(elapsed, 2),
    }

    output_path = (
        OUTPUT_DIR / "ocr_employee_rules_summary.json"
    )

    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== テスト結果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n結果保存先: {output_path}")


if __name__ == "__main__":
    main()
