from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QObject,
    QEvent,
    QThread,
    Signal,
    Slot,
    Qt,
)
from PySide6.QtGui import (
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    PDF_PATH,
)

from services.chunker import (
    split_pages_into_articles,
)
from services.document_manager import (
    cleanup_temporary_pdf,
    create_temporary_pdf_copy,
    load_document_metadata,
    replace_pdf,
    save_document_metadata,
)
from services.index_cache import (
    save_index_metadata,
)
from services.pdf_reader import (
    extract_pdf_pages,
)


# ============================================================
# 質問処理Worker
# ============================================================


class RagWorker(QObject):
    """
    RAGの質問処理を別スレッドで実行する。

    実際のRAG処理はRagServiceへ委譲する。
    """

    finished = Signal(str, list)
    error = Signal(str)

    def __init__(
        self,
        question: str,
        rag_service,
    ) -> None:
        super().__init__()

        self.question = question
        self.rag_service = rag_service

    @Slot()
    def run(self) -> None:
        """
        RagServiceを使用して質問処理を実行する。
        """

        try:
            result = (
                self.rag_service.ask(
                    self.question
                )
            )

            self.finished.emit(
                result["answer"],
                result["references"],
            )

        except Exception as error:
            self.error.emit(
                str(error)
            )


# ============================================================
# PDF更新Worker
# ============================================================


class PdfUpdateWorker(QObject):
    """
    社員規則PDF更新用Worker。

    まず一時ファイルで
    PDF解析・チャンク分割・Embeddingを行い、
    すべて成功した場合のみ本番データへ反映する。
    """

    finished = Signal(int, str)
    error = Signal(str)

    def __init__(
        self,
        source_pdf: Path,
        embedding_service,
        vector_store,
    ) -> None:
        super().__init__()

        self.source_pdf = source_pdf
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    @Slot()
    def run(self) -> None:
        temp_pdf_path: Path | None = None

        try:
            original_filename = (
                self.source_pdf.name
            )

            # ================================================
            # 1. 一時コピー作成
            # ================================================

            temp_pdf_path = (
                create_temporary_pdf_copy(
                    self.source_pdf
                )
            )

            # ================================================
            # 2. 一時PDFを解析
            # ================================================

            pages = extract_pdf_pages(
                temp_pdf_path
            )

            # ================================================
            # 3. 条文分割
            # ================================================

            chunks = (
                split_pages_into_articles(
                    pages
                )
            )

            if not chunks:
                raise ValueError(
                    "社員規則から条文を取得できませんでした。"
                )

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            # ================================================
            # 4. Embedding
            # ================================================

            vectors = (
                self.embedding_service
                .embed_documents(
                    texts
                )
            )

            if len(vectors) != len(chunks):
                raise ValueError(
                    "ベクトル数とチャンク数が一致しません。"
                )

            # ================================================
            # 5. 本番PDFへ反映
            # ================================================

            replace_pdf(
                source_path=temp_pdf_path,
                destination_path=PDF_PATH,
            )

            # ================================================
            # 6. FAISS更新
            # ================================================

            total_count = (
                self.vector_store.create_index(
                    vectors=vectors,
                    chunks=chunks,
                )
            )

            # ================================================
            # 7. キャッシュ情報更新
            # ================================================

            save_index_metadata(
                pdf_path=PDF_PATH,
                chunk_count=len(chunks),
            )

            # ================================================
            # 8. 表示用メタ情報更新
            # ================================================

            save_document_metadata(
                source_filename=(
                    original_filename
                )
            )

            self.finished.emit(
                total_count,
                original_filename,
            )

        except Exception as error:
            self.error.emit(
                str(error)
            )

        finally:
            if temp_pdf_path is not None:
                cleanup_temporary_pdf(
                    temp_pdf_path
                )


# ============================================================
# MainWindow
# ============================================================


class MainWindow(QMainWindow):
    def __init__(
        self,
        rag_service,
        embedding_service,
        vector_store,
    ) -> None:
        super().__init__()

        # RAG質問処理
        self.rag_service = (
            rag_service
        )

        # PDF再Embedding用
        self.embedding_service = (
            embedding_service
        )

        # PDF更新時のFAISS再構築用
        self.vector_store = (
            vector_store
        )

        # 質問Worker
        self.worker_thread: QThread | None = None
        self.worker: RagWorker | None = None

        # PDF更新Worker
        self.update_thread: QThread | None = None
        self.update_worker: PdfUpdateWorker | None = None

        self.setWindowTitle(
            "社員規則 AI検索"
        )

        self.resize(
            950,
            800,
        )

        self._create_ui()

    # ========================================================
    # UI作成
    # ========================================================

    def _create_ui(self) -> None:
        central_widget = QWidget()

        main_layout = QVBoxLayout(
            central_widget
        )

        main_layout.setContentsMargins(
            18,
            18,
            18,
            14,
        )

        main_layout.setSpacing(
            10
        )

        # ====================================================
        # タイトル
        # ====================================================

        title_label = QLabel(
            "社員規則 AI検索"
        )

        title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title_label.setStyleSheet(
            """
            QLabel {
                font-size: 26px;
                font-weight: bold;
                padding: 10px;
            }
            """
        )

        main_layout.addWidget(
            title_label
        )

        # ====================================================
        # 現在のPDF
        # ====================================================

        pdf_layout = QHBoxLayout()

        metadata = (
            load_document_metadata()
        )

        source_filename = (
            metadata.get(
                "source_filename",
                PDF_PATH.name,
            )
        )

        updated_at = (
            metadata.get(
                "updated_at",
                "未記録",
            )
        )

        self.pdf_label = QLabel(
            f"社員規則PDF：{source_filename}\n"
            f"最終更新：{updated_at}"
        )

        self.pdf_update_button = QPushButton(
            "社員規則PDFを更新"
        )

        self.pdf_update_button.clicked.connect(
            self.select_new_pdf
        )

        pdf_layout.addWidget(
            self.pdf_label,
            stretch=1,
        )

        pdf_layout.addWidget(
            self.pdf_update_button
        )

        main_layout.addLayout(
            pdf_layout
        )

        # ====================================================
        # 説明
        # ====================================================

        description_label = QLabel(
            "社員規則について質問を入力してください。"
            "  Ctrl + Enter でも質問できます。"
        )

        main_layout.addWidget(
            description_label
        )

        # ====================================================
        # 質問入力
        # ====================================================

        self.question_input = QTextEdit()

        self.question_input.setPlaceholderText(
            "例：有給休暇はいつ付与されますか？"
        )

        self.question_input.setFixedHeight(
            100
        )

        self.question_input.installEventFilter(
            self
        )

        main_layout.addWidget(
            self.question_input
        )

        # ====================================================
        # ボタン
        # ====================================================

        button_layout = QHBoxLayout()

        self.ask_button = QPushButton(
            "質問する"
        )

        self.ask_button.setFixedHeight(
            42
        )

        self.ask_button.clicked.connect(
            self.ask_question
        )

        self.clear_button = QPushButton(
            "入力をクリア"
        )

        self.clear_button.setFixedHeight(
            42
        )

        self.clear_button.clicked.connect(
            self.clear_question
        )

        button_layout.addWidget(
            self.ask_button
        )

        button_layout.addWidget(
            self.clear_button
        )

        main_layout.addLayout(
            button_layout
        )

        # ====================================================
        # ステータス
        # ====================================================

        self.status_label = QLabel(
            "準備完了"
        )

        self.status_label.setStyleSheet(
            """
            QLabel {
                padding: 6px;
                font-weight: bold;
            }
            """
        )

        main_layout.addWidget(
            self.status_label
        )

        # ====================================================
        # 回答
        # ====================================================

        answer_label = QLabel(
            "回答"
        )

        answer_label.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: bold;
            }
            """
        )

        main_layout.addWidget(
            answer_label
        )

        self.answer_output = QTextEdit()

        self.answer_output.setReadOnly(
            True
        )

        self.answer_output.setPlaceholderText(
            "ここにAIの回答が表示されます。"
        )

        main_layout.addWidget(
            self.answer_output
        )

        # ====================================================
        # 参照条文
        # ====================================================

        reference_label = QLabel(
            "参照した条文"
        )

        reference_label.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: bold;
            }
            """
        )

        main_layout.addWidget(
            reference_label
        )

        self.reference_output = QTextEdit()

        self.reference_output.setReadOnly(
            True
        )

        self.reference_output.setFixedHeight(
            145
        )

        main_layout.addWidget(
            self.reference_output
        )

        # ====================================================
        # モデル情報
        # ====================================================

        embedding_name = (
            EMBEDDING_MODEL_NAME
            .split("/")[-1]
        )

        model_info_label = QLabel(
            f"Embedding: {embedding_name}"
            f"  /  LLM: {OLLAMA_MODEL_NAME}"
        )

        model_info_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
        )

        model_info_label.setStyleSheet(
            """
            QLabel {
                color: #666666;
                font-size: 11px;
            }
            """
        )

        main_layout.addWidget(
            model_info_label
        )

        self.setCentralWidget(
            central_widget
        )

        self.question_input.setFocus()

    # ========================================================
    # Ctrl + Enter
    # ========================================================

    def eventFilter(
        self,
        watched,
        event,
    ) -> bool:
        if (
            watched is self.question_input
            and event.type()
            == QEvent.Type.KeyPress
        ):
            if isinstance(
                event,
                QKeyEvent,
            ):
                if (
                    event.key()
                    in {
                        Qt.Key.Key_Return,
                        Qt.Key.Key_Enter,
                    }
                    and event.modifiers()
                    & Qt.KeyboardModifier.ControlModifier
                ):
                    self.ask_question()

                    return True

        return super().eventFilter(
            watched,
            event,
        )

    # ========================================================
    # PDF更新
    # ========================================================

    @Slot()
    def select_new_pdf(
        self,
    ) -> None:
        """
        新しい社員規則PDFを選択する。
        """

        if self._is_processing():
            return

        file_path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "社員規則PDFを選択",
                str(PDF_PATH.parent),
                "PDFファイル (*.pdf)",
            )
        )

        if not file_path:
            return

        selected_pdf = Path(
            file_path
        )

        reply = QMessageBox.question(
            self,
            "社員規則PDFの更新",
            (
                "社員規則PDFを更新します。\n\n"
                f"{selected_pdf.name}\n\n"
                "ベクトルデータも再作成されます。\n"
                "実行しますか？"
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            ),
            QMessageBox.StandardButton.No,
        )

        if (
            reply
            != QMessageBox.StandardButton.Yes
        ):
            return

        self._start_pdf_update(
            selected_pdf
        )

    def _start_pdf_update(
        self,
        selected_pdf: Path,
    ) -> None:
        """
        PDF更新処理をバックグラウンドで開始する。
        """

        self._set_all_controls_enabled(
            False
        )

        self.status_label.setText(
            "社員規則PDFを更新しています..."
        )

        self.answer_output.setText(
            "社員規則を再ベクトル化しています。"
            "\nしばらくお待ちください。"
        )

        self.reference_output.clear()

        self.update_thread = QThread()

        self.update_worker = (
            PdfUpdateWorker(
                source_pdf=selected_pdf,
                embedding_service=(
                    self.embedding_service
                ),
                vector_store=(
                    self.vector_store
                ),
            )
        )

        self.update_worker.moveToThread(
            self.update_thread
        )

        self.update_thread.started.connect(
            self.update_worker.run
        )

        self.update_worker.finished.connect(
            self._on_pdf_update_finished
        )

        self.update_worker.error.connect(
            self._on_pdf_update_error
        )

        self.update_worker.finished.connect(
            self.update_thread.quit
        )

        self.update_worker.error.connect(
            self.update_thread.quit
        )

        self.update_worker.finished.connect(
            self.update_worker.deleteLater
        )

        self.update_worker.error.connect(
            self.update_worker.deleteLater
        )

        self.update_thread.finished.connect(
            self.update_thread.deleteLater
        )

        self.update_thread.finished.connect(
            self._on_update_thread_finished
        )

        self.update_thread.start()

    @Slot(int, str)
    def _on_pdf_update_finished(
        self,
        total_count: int,
        source_filename: str,
    ) -> None:
        """
        PDF更新成功時。
        """

        metadata = (
            load_document_metadata()
        )

        updated_at = metadata.get(
            "updated_at",
            "未記録",
        )

        self.pdf_label.setText(
            f"社員規則PDF：{source_filename}\n"
            f"最終更新：{updated_at}"
        )

        self.answer_output.clear()
        self.reference_output.clear()

        self.status_label.setText(
            "社員規則PDF 更新完了"
        )

        QMessageBox.information(
            self,
            "更新完了",
            (
                "社員規則PDFを更新しました。\n\n"
                f"ファイル：{source_filename}\n"
                f"登録チャンク数：{total_count}件"
            ),
        )

    @Slot(str)
    def _on_pdf_update_error(
        self,
        error_message: str,
    ) -> None:
        self.status_label.setText(
            "PDF更新エラー"
        )

        QMessageBox.critical(
            self,
            "PDF更新エラー",
            error_message,
        )

    @Slot()
    def _on_update_thread_finished(
        self,
    ) -> None:
        self.update_thread = None
        self.update_worker = None

        self._set_all_controls_enabled(
            True
        )

        if (
            self.status_label.text()
            != "PDF更新エラー"
        ):
            self.status_label.setText(
                "準備完了"
            )

        self.question_input.setFocus()

    # ========================================================
    # 質問
    # ========================================================

    @Slot()
    def clear_question(
        self,
    ) -> None:
        if self._is_processing():
            return

        self.question_input.clear()

        self.question_input.setFocus()

    @Slot()
    def ask_question(
        self,
    ) -> None:
        question = (
            self.question_input
            .toPlainText()
            .strip()
        )

        if not question:
            QMessageBox.warning(
                self,
                "入力エラー",
                "質問を入力してください。",
            )

            return

        if self._is_processing():
            return

        self.ask_button.setText(
            "回答生成中..."
        )

        self._set_all_controls_enabled(
            False
        )

        self.status_label.setText(
            "AIが社員規則を検索しています..."
        )

        self.answer_output.setText(
            "回答を生成しています..."
        )

        self.reference_output.clear()

        # ====================================================
        # RagService Worker
        # ====================================================

        self.worker_thread = QThread()

        self.worker = RagWorker(
            question=question,
            rag_service=(
                self.rag_service
            ),
        )

        self.worker.moveToThread(
            self.worker_thread
        )

        self.worker_thread.started.connect(
            self.worker.run
        )

        self.worker.finished.connect(
            self._on_answer_finished
        )

        self.worker.error.connect(
            self._on_answer_error
        )

        self.worker.finished.connect(
            self.worker_thread.quit
        )

        self.worker.error.connect(
            self.worker_thread.quit
        )

        self.worker.finished.connect(
            self.worker.deleteLater
        )

        self.worker.error.connect(
            self.worker.deleteLater
        )

        self.worker_thread.finished.connect(
            self.worker_thread.deleteLater
        )

        self.worker_thread.finished.connect(
            self._on_question_thread_finished
        )

        self.worker_thread.start()

    @Slot(str, list)
    def _on_answer_finished(
        self,
        answer: str,
        search_results: list[dict[str, Any]],
    ) -> None:
        self.answer_output.setText(
            answer
        )

        self._show_references(
            search_results
        )

        self.status_label.setText(
            "回答生成完了"
        )

    @Slot(str)
    def _on_answer_error(
        self,
        error_message: str,
    ) -> None:
        self.answer_output.clear()

        self.status_label.setText(
            "エラーが発生しました"
        )

        QMessageBox.critical(
            self,
            "処理エラー",
            error_message,
        )

    @Slot()
    def _on_question_thread_finished(
        self,
    ) -> None:
        self.worker_thread = None
        self.worker = None

        self._set_all_controls_enabled(
            True
        )

        self.ask_button.setText(
            "質問する"
        )

        if (
            self.status_label.text()
            != "エラーが発生しました"
        ):
            self.status_label.setText(
                "準備完了"
            )

        self.question_input.setFocus()

    # ========================================================
    # 共通
    # ========================================================

    def _is_processing(
        self,
    ) -> bool:
        question_running = (
            self.worker_thread is not None
            and self.worker_thread.isRunning()
        )

        update_running = (
            self.update_thread is not None
            and self.update_thread.isRunning()
        )

        return (
            question_running
            or update_running
        )

    def _set_all_controls_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.question_input.setEnabled(
            enabled
        )

        self.ask_button.setEnabled(
            enabled
        )

        self.clear_button.setEnabled(
            enabled
        )

        self.pdf_update_button.setEnabled(
            enabled
        )

    def _show_references(
        self,
        search_results: list[dict[str, Any]],
    ) -> None:
        if not search_results:
            self.reference_output.setText(
                "参照条文なし"
            )

            return

        lines: list[str] = []

        for rank, result in enumerate(
            search_results,
            start=1,
        ):
            start_page = result[
                "start_page"
            ]

            end_page = result[
                "end_page"
            ]

            if start_page == end_page:
                page_text = (
                    f"{start_page}ページ"
                )

            else:
                page_text = (
                    f"{start_page}"
                    f"～{end_page}ページ"
                )

            lines.append(
                f"{rank}位: "
                f"{result['article']} "
                f"({page_text}) "
                f"類似度: "
                f"{result['score']:.4f}"
            )

        self.reference_output.setText(
            "\n".join(lines)
        )