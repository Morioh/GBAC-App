# Configuration Template for GBAC RAG App
# Copy this to config.py and customize as needed

# Database Settings
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "gbac_chunks"

# Embedding Model
# Options: 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (better quality)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Search Defaults
DEFAULT_TOP_K = 10
DEFAULT_BM25_WEIGHT = 0.3
MIN_TOP_K = 5
MAX_TOP_K = 30

# Data File
DEFAULT_JSON_PATH = "gbac_chunks.json"

# UI Settings
PAGE_TITLE = "GBAC RAG System"
PAGE_ICON = "🏢"
AUTO_EXPAND_TOP_N = 3  # Number of top results to auto-expand

# Example Queries
EXAMPLE_QUERIES = [
    "What policies were discussed in 2024?",
    "Building Energy Performance Standards",
    "Electrification and fossil fuel phase-out",
    "Equity concerns and affordability",
    "What did stakeholders say about BEPS?",
    "Innovation Hub updates",
    "Compliance mechanisms discussed",
]

# Advanced Settings
SHOW_SCORE_BREAKDOWN = True  # Show semantic/BM25 scores separately
SHOW_CHAT_HISTORY = True
MAX_HISTORY_ITEMS = 10

# Performance
ENABLE_PROGRESS_BAR = True  # Show progress when loading chunks
CACHE_EMBEDDINGS = True
