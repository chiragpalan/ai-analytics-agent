@echo off
REM =============================================================
REM  AI Analytics Agent — Project Setup Script (Windows)
REM  Run: setup_project.bat  (double-click or run in CMD)
REM =============================================================

set PROJECT=ai-analytics-agent

echo.
echo ===========================================
echo   AI Analytics Agent — Project Setup
echo ===========================================
echo.

echo [1/3] Creating folder structure...

mkdir "%PROJECT%"
mkdir "%PROJECT%\.streamlit"
mkdir "%PROJECT%\assets"
mkdir "%PROJECT%\src"
mkdir "%PROJECT%\src\llm"
mkdir "%PROJECT%\src\data"
mkdir "%PROJECT%\src\metadata"
mkdir "%PROJECT%\src\kpi"
mkdir "%PROJECT%\src\insights"
mkdir "%PROJECT%\src\dashboard"

REM Create __init__.py files
type nul > "%PROJECT%\src\__init__.py"
type nul > "%PROJECT%\src\llm\__init__.py"
type nul > "%PROJECT%\src\data\__init__.py"
type nul > "%PROJECT%\src\metadata\__init__.py"
type nul > "%PROJECT%\src\kpi\__init__.py"
type nul > "%PROJECT%\src\insights\__init__.py"
type nul > "%PROJECT%\src\dashboard\__init__.py"

echo [OK] Folder structure created.

echo.
echo [2/3] Listing created structure:
echo.
dir /s /b "%PROJECT%" 2>nul | findstr /v "__pycache__"

echo.
echo ===========================================
echo   Next steps:
echo ===========================================
echo.
echo 1. Copy all source files into the project folders
echo    (see the README for the full file list)
echo.
echo 2. Open the folder in VS Code:
echo    code %PROJECT%
echo.
echo 3. Create virtual environment (in VS Code terminal):
echo    python -m venv venv
echo    venv\Scripts\activate
echo    pip install -r requirements.txt
echo.
echo 4. Run the app:
echo    streamlit run app.py
echo.
echo ===========================================
pause
