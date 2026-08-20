@echo off
REM GBAC RAG System - Windows Setup Script

echo.
echo ========================================
echo   GBAC RAG System - Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Virtual environment activated
echo.

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To run the app:
echo   1. Run: run.bat
echo   2. Or manually:
echo      - venv\Scripts\activate.bat
echo      - streamlit run gbac_rag_app.py
echo.
pause
