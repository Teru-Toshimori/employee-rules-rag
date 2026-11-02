

from pathlib import Path
from datetime import datetime
import json
import re
import traceback

from services.pdf_table_ocr import extract_pdf_tables


# ============================================================
# テスト設定
# ============================================================

PDF_PATH = Path(
    "test_pdfs/RAG汎用性テスト_業務マニュアル.pdf"
)

TARGET_PAGES = None

OUTPUT_DIR = Path("doc/test_results")

JSON_OUTPUT = OUTPUT_DIR / "ocr_output.json"
TXT_OUTPUT = OUTPUT_DIR / "ocr_output.txt"
MD_OUTPUT = OUTPUT_DIR / "02_generic_pdf_ocr.md"


# ============================================================
# OCR結果の簡易構造チェック
# ============================================================

def check_table_structure(text: str) -> dict:
    """
    OCR出力に含まれる表の行・列構造を簡易確認する。

    注意:
        原本PDFとの完全一致を判定するものではない。
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.startswith("【OCR抽出表データ】")
    ]

    # 「|」区切りの表形式を想定
    table_lines = [
        line for line in lines if "|" in line
    ]

    column_counts = [
        len(line.split("|"))
        for line in table_lines
    ]

    consistent_columns = (
        len(column_counts) > 0
        and len(set(column_counts)) == 1
    )

    return {
        "line_count": len(lines),
        "table_line_count": len(table_lines),
        "column_counts": column_counts,
        "consistent_columns": consistent_columns,
    }


# ============================================================
# TXT出力
# ============================================================

def save_txt(tables, elapsed):
    lines = [
        "=== 業務マニュアル PDF OCR抽出テスト ===",
        f"対象PDF: {PDF_PATH.name}",
        f"対象ページ: {TARGET_PAGES}",
        f"抽出テーブル数: {len(tables)}",
        f"処理時間: {elapsed:.2f}秒",
        "",
    ]

    for i, table in enumerate(tables, start=1):
        lines.extend([
            f"=== Table {i} ===",
            f"Page: {table.get('page')}",
            f"Article: {table.get('article_number')}",
            f"BBox: {table.get('bbox')}",
            "",
            str(table.get("text", "")),
            "",
        ])

    TXT_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Markdown出力
# ============================================================

def save_markdown(tables, elapsed, checks):
    today = datetime.now().strftime("%Y年%m月%d日")

    table_detected = len(tables) > 0

    structure_ok = (
        table_detected
        and all(
            c["consistent_columns"]
            for c in checks
        )
    )

    detection_result = "OK" if table_detected else "NG"

    structure_result = (
        "OK（簡易構造確認）"
        if structure_ok
        else "要確認"
    )

    md = [
        "# 業務マニュアルPDF OCR抽出テスト結果報告書",
        "",
        "## 1. テスト目的",
        "",
        "社員就業規則以外のPDFに対して、"
        "画像形式の表データをOCRで抽出できるか確認する。",
        "",
        "## 2. テスト環境",
        "",
        "- 対象システム：社員規則RAGシステム",
        "- OCRエンジン：PaddleOCR（PP-StructureV3）",
        f"- 対象PDF：{PDF_PATH.name}",
        f"- 対象ページ：{TARGET_PAGES}",
        "- テストスクリプト：test_ocr_validation.py",
        f"- 実行日：{today}",
        "",
        "## 3. テスト結果",
        "",
        "| No. | テスト項目 | 実際の結果 | 判定 |",
        "|---|---|---|---|",
        f"| 1 | 表データ検出 | "
        f"{len(tables)}件抽出 | {detection_result} |",
        f"| 2 | 表の行・列構造 | "
        f"区切り文字による簡易確認 | {structure_result} |",
        "| 3 | 文字・数値の正確性 | "
        "原本PDFとの照合が必要 | 要目視確認 |",
        "| 4 | OCR処理 | "
        "例外なく処理完了 | OK |",
        "",
        "## 4. OCR抽出結果",
        "",
        f"- 抽出テーブル数：{len(tables)}件",
        f"- 処理時間：{elapsed:.2f}秒",
        "",
    ]

    for i, table in enumerate(tables, start=1):
        md.extend([
            f"### Table {i}",
            "",
            f"- ページ：{table.get('page')}",
            f"- 条文番号：{table.get('article_number')}",
            "",
            "```text",
            str(table.get("text", "")),
            "```",
            "",
        ])

    md.extend([
        "## 5. 確認事項",
        "",
        "OCR抽出結果について、原本PDFと比較し、"
        "文字・数値・行列の対応関係を目視確認する。",
        "",
        "## 6. 総合判定",
        "",
        (
            "**OCR抽出成功（文字精度は要確認）**"
            if table_detected
            else "**表データ未検出（要調査）**"
        ),
        "",
        "本テストでは、OCRによる表データ抽出を確認した。",
        "ただし、原本との完全一致は未判定である。",
        "",
    ])

    MD_OUTPUT.write_text(
        "\n".join(md),
        encoding="utf-8",
    )


# ============================================================
# メイン処理
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== OCR テスト開始 ===")

    print(f"対象PDF: {PDF_PATH}")
    print(f"対象ページ: {TARGET_PAGES}")

    if not PDF_PATH.is_file():
        print(f"ERROR: PDFが見つかりません: {PDF_PATH}")
        return

    start_time = datetime.now()

    try:
        # OCR実行
        tables = extract_pdf_tables(
            pdf_path=PDF_PATH,
            target_pages=TARGET_PAGES,
        )

        elapsed = (
            datetime.now() - start_time
        ).total_seconds()

        print(f"\n抽出テーブル数: {len(tables)}")
        print(f"処理時間: {elapsed:.2f}秒")

        checks = []

        for i, table in enumerate(tables, start=1):
            print(f"\n=== Table {i} ===")

            print(f"Page: {table.get('page')}")
            print(f"Article: {table.get('article_number')}")
            print(f"BBox: {table.get('bbox')}")

            text = str(table.get("text", ""))

            print(text)

            check = check_table_structure(text)
            checks.append(check)

            print(
                "列構造チェック:",
                check["consistent_columns"]
            )

        # JSON保存
        result = {
            "pdf": str(PDF_PATH),
            "target_pages": TARGET_PAGES,
            "table_count": len(tables),
            "elapsed_seconds": round(elapsed, 2),
            "structure_checks": checks,
            "tables": tables,
        }

        JSON_OUTPUT.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # TXT保存
        save_txt(tables, elapsed)

        # Markdown報告書保存
        save_markdown(tables, elapsed, checks)

        print("\n=== 保存完了 ===")
        print(f"JSON: {JSON_OUTPUT}")
        print(f"TXT : {TXT_OUTPUT}")
        print(f"MD  : {MD_OUTPUT}")

        print("\n=== OCR テスト終了 ===")

    except Exception:
        print("\nERROR: OCRテストで例外が発生しました")
        traceback.print_exc()


if __name__ == "__main__":
    main()
