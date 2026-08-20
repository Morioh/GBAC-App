#!/bin/bash

# Quick run script for GBAC RAG Streamlit App

echo "🚀 Starting GBAC RAG System..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    echo "✅ Virtual environment activated"
fi

# Run Streamlit
streamlit run gbac_rag_app.py

# Deactivate virtual environment on exit
deactivate
