@echo off
REM GBAC RAG System - Windows Run Script

echo.
echo Starting GBAC RAG System...
echo.

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
    echo.
)

REM Run Streamlit
streamlit run gbac_rag_app.py

REM Deactivate on exit
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)
