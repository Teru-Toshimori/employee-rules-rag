@echo off
chcp 65001 >nul

cd /d C:\Users\shise\workspace\local-rag-test

echo ============================================================
echo 社員規則AI Lark Bot
echo ============================================================
echo.

REM ------------------------------------------------------------
REM Ollamaの起動確認
REM ------------------------------------------------------------

curl -s http://localhost:11434/api/tags >nul 2>&1

if errorlevel 1 (
    echo Ollamaを起動します...

    start "" /B ollama serve

    echo Ollamaの起動を待っています...

    timeout /t 5 /nobreak >nul
) else (
    echo Ollamaは起動済みです。
)

REM ------------------------------------------------------------
REM Lark Bot起動
REM ------------------------------------------------------------

echo Lark Botを起動します...
echo.

".venv\Scripts\python.exe" lark_app.py

echo.
echo Lark Botが終了しました。
pause