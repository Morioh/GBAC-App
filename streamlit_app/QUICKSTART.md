# Quick Start Guide - GBAC RAG Streamlit App

## 🚀 Get Running in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Or use the setup script:
```bash
./setup.sh
```

### Step 2: Prepare Your Data
Make sure you have `gbac_chunks.json` in the same folder as `gbac_rag_app.py`

### Step 3: Run the App
```bash
streamlit run gbac_rag_app.py
```

Or use the run script:
```bash
./run.sh
```

That's it! The app will open in your browser at `http://localhost:8501`

---

## 📖 First-Time Usage

1. **Load Data**: In the sidebar, verify the JSON path and click "Load/Reload Data"
2. **Wait**: First load takes a minute to build the database
3. **Search**: Enter a question and click Search
4. **Explore**: Click on results to read full content

---

## 💡 Quick Examples

Try these queries to test the system:

- `What were the main discussions in 2024 meetings?`
- `Building Energy Performance Standards BEPS`
- `stakeholder concerns about electrification`
- `equity and affordability issues`
- `compliance mechanisms`

---

## ⚙️ Key Settings

**Number of Results**: How many results to show (default: 10)

**BM25 Weight**: Balance between keyword and semantic search
- 0.0 = Pure semantic (meaning-based)
- 0.3 = Balanced (recommended)
- 1.0 = Pure keyword (exact matches)

---

## 🔧 Troubleshooting

**App won't start?**
```bash
pip install streamlit --upgrade
```

**No results found?**
- Check that data loaded successfully (sidebar shows chunk count)
- Try broader queries
- Lower the BM25 weight for more semantic results

**Slow performance?**
- First load is slow (building database)
- Subsequent runs are much faster
- Database cached in `./chroma_db/` folder

---

## 📚 Next Steps

Once comfortable with basic search, explore:
- Adjusting search weights for different query types
- Using the search history
- Comparing results across different date ranges
- Adding your own example queries

For detailed documentation, see `README.md`
