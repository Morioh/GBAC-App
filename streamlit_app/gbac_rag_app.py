"""
GBAC Policy Intelligence — TPP Labs Interface
Georgetown University McCourt School of Public Policy
"""

import streamlit as st
import chromadb

# Load API keys from .env file if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import re

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ==================== Backend Classes (unchanged) ====================

class QueryAnalyzer:
    @staticmethod
    def detect_query_type(query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        attendance_patterns = [
            r'who attended (all|every|each)', r'list (all )?attendees',
            r'who (was|were) (at|present at|in) (all|every|each)',
            r'attendance (across|for) all', r'attendees? (for|across|at) all'
        ]
        if any(re.search(p, query_lower) for p in attendance_patterns):
            result = {'query_type': 'attendance', 'requires_all_instances': True,
                      'target_section': 'Attendees', 'date_filter': None}
            m = re.search(r'in (\d{4})', query_lower)
            if m:
                result['date_filter'] = {'year': int(m.group(1))}
            return result
        aggregation_patterns = [
            r'(all|every) (the )?(meetings?|sessions?|discussions?)', r'(across|in|during) all',
            r'list (all|everything)', r'summarize (all|everything)',
            r'consistently (raised|discussed|mentioned)',
            r'(what|which).*(concerns?|issues?|topics?|decisions?).*(raised?|discussed?)',
        ]
        if any(re.search(p, query_lower) for p in aggregation_patterns):
            result = {'query_type': 'aggregation', 'requires_all_instances': True,
                      'target_section': None, 'date_filter': None}
            if 'action item' in query_lower:
                result['target_section'] = 'Action Items'
            elif 'attendee' in query_lower or 'attendance' in query_lower:
                result['target_section'] = 'Attendees'
            m = re.search(r'in (\d{4})', query_lower)
            if m:
                result['date_filter'] = {'year': int(m.group(1))}
            return result
        if any(w in query_lower for w in ['how many', 'count', 'total number', 'frequency']):
            return {'query_type': 'metrics', 'requires_all_instances': True,
                    'target_section': None, 'date_filter': None}
        return {'query_type': 'standard', 'requires_all_instances': False,
                'target_section': None, 'date_filter': None}


class RAGDatabase:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = None):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        if collection_name is None:
            existing = self.client.list_collections()
            if existing:
                self.collection = existing[0]
            else:
                self.collection = self.client.create_collection(
                    name="gbac_chunks", metadata={"hnsw:space": "cosine"})
        else:
            try:
                self.collection = self.client.get_collection(name=collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"})

    def query(self, query_text: str, n_results: int = 10) -> Dict[str, Any]:
        qe = self.embedding_model.encode([query_text])[0]
        return self.collection.query(query_embeddings=[qe.tolist()], n_results=n_results)

    def get_count(self) -> int:
        return self.collection.count()


def _get_meta(metadata: dict, *keys, default: str = 'Unknown') -> str:
    for key in keys:
        val = metadata.get(key)
        if val and val not in ('Unknown', ''):
            return val
    return default


class HybridRetriever:
    def __init__(self, rag_db: RAGDatabase):
        self.rag_db = rag_db
        self.bm25 = None
        self.documents: List[str] = []
        self.all_metadatas: List[dict] = []
        self._init_bm25()

    def _init_bm25(self):
        all_data = self.rag_db.collection.get()
        self.documents = all_data['documents'] or []
        self.all_metadatas = all_data['metadatas'] or []
        if self.documents:
            self.bm25 = BM25Okapi([d.lower().split() for d in self.documents])

    def retrieve(self, query: str, top_k: int = 20, bm25_weight: float = 0.4) -> List[Dict]:
        semantic_weight = 1 - bm25_weight
        k = 60

        raw = self.rag_db.query(query, n_results=top_k * 2)
        semantic_results = []
        if raw['documents'] and raw['documents'][0]:
            for i in range(len(raw['documents'][0])):
                semantic_results.append({
                    'text': raw['documents'][0][i],
                    'metadata': raw['metadatas'][0][i],
                    'semantic_score': 1 - raw['distances'][0][i],
                    'bm25_score': 0.0,
                })

        bm25_results = []
        if self.bm25 and self.documents:
            scores = self.bm25.get_scores(query.lower().split())
            for idx in np.argsort(scores)[::-1][:top_k * 2]:
                if scores[idx] > 0 and idx < len(self.documents):
                    bm25_results.append({
                        'text': self.documents[idx],
                        'metadata': self.all_metadatas[idx] if idx < len(self.all_metadatas) else {},
                        'semantic_score': 0.0,
                        'bm25_score': float(scores[idx]),
                    })

        rrf: Dict[str, Dict] = {}
        for rank, r in enumerate(semantic_results, 1):
            key = r['text'][:100]
            rrf[key] = {'score': semantic_weight / (k + rank), 'result': r}
        for rank, r in enumerate(bm25_results, 1):
            key = r['text'][:100]
            if key in rrf:
                rrf[key]['score'] += (1 - semantic_weight) / (k + rank)
                rrf[key]['result']['bm25_score'] = r['bm25_score']
            else:
                rrf[key] = {'score': (1 - semantic_weight) / (k + rank), 'result': r}

        sorted_items = sorted(rrf.values(), key=lambda x: x['score'], reverse=True)
        final = []
        for item in sorted_items[:top_k]:
            r = item['result']
            r['combined_score'] = item['score']
            final.append(r)

        if final:
            mx = final[0]['combined_score']
            if mx > 0:
                for r in final:
                    r['combined_score'] = r['combined_score'] / mx
        return final


# ==================== RAG Answer (no Streamlit calls inside) ====================

def rag_answer(
    query: str,
    retriever: HybridRetriever,
    api_key: Optional[str],
    provider: str,
    n_results: int = 25,
    temperature: float = 0.3,
) -> tuple:
    """Returns (answer_str_or_None, sources_list, error_str_or_None)"""
    results = retriever.retrieve(query, top_k=n_results, bm25_weight=0.4)
    if not results:
        return None, [], "No relevant content found for this query."

    sources = results[:10]

    if not api_key:
        return None, sources, None

    context_parts = []
    for i, chunk in enumerate(sources, 1):
        meta = chunk.get('metadata', {})
        date = _get_meta(meta, 'date', 'meeting_date')
        section = _get_meta(meta, 'section', 'section_type', default='General')
        fname = _get_meta(meta, 'file', 'filename')
        context_parts.append(
            f"[Source {i} | {fname} | {section} | {date}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert analyst reviewing DC DOEE's Green Building Advisory Council (GBAC) meeting notes.

Answer the question clearly and concisely, citing specific meeting dates and sections.

Context from GBAC meeting notes:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- Cite meeting dates (e.g., "In the October 2024 meeting...") to support your statements
- If the context is incomplete, state what is and isn't covered
- Be clear and direct

Answer:"""

    try:
        if provider == 'anthropic':
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text, sources, None

        elif provider == 'openai':
            client = OpenAI(api_key=api_key)
            for model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert analyst of government meeting notes."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        temperature=temperature
                    )
                    return resp.choices[0].message.content, sources, None
                except Exception as me:
                    if "does not exist" in str(me) or "model" in str(me).lower():
                        continue
                    raise me
            return None, sources, "No accessible OpenAI models found."

    except Exception as e:
        return None, sources, str(e)


def summarize_meeting(date: str, rag_db: RAGDatabase, api_key: str, provider: str, temperature: float = 0.3) -> tuple:
    all_data = rag_db.collection.get(include=['documents', 'metadatas'])
    chunks = []
    for doc, meta in zip(all_data['documents'] or [], all_data['metadatas'] or []):
        if _get_meta(meta, 'date', 'meeting_date', default='') == date:
            section = _get_meta(meta, 'section', 'section_type', default='General')
            chunks.append(f"[{section}]\n{doc}")
    if not chunks:
        return None, f"No data found for {date}"
    context = "\n\n".join(chunks)
    prompt = f"""Summarize this GBAC meeting from {date}.

Meeting content:
{context}

Provide a structured summary with:
1. Main topics discussed
2. Key decisions or recommendations
3. Action items (if any)
4. Notable concerns or questions raised

Summary:"""
    try:
        if provider == 'anthropic':
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1500,
                temperature=temperature, messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text, None
        elif provider == 'openai':
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert analyst of government meeting notes."},
                          {"role": "user", "content": prompt}],
                max_tokens=1500, temperature=temperature
            )
            return resp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


def compare_meetings(dates: List[str], topic: str, retriever: HybridRetriever, api_key: str, provider: str, temperature: float = 0.3) -> tuple:
    meeting_contexts = []
    for date in dates:
        results = retriever.retrieve(topic, top_k=10)
        dr = [r for r in results if _get_meta(r['metadata'], 'date', 'meeting_date', default='') == date]
        if dr:
            meeting_contexts.append(f"Meeting {date}:\n" + "\n".join(r['text'] for r in dr[:3]))
    if not meeting_contexts:
        return None, "No relevant content found for the selected meetings."
    combined = "\n\n---\n\n".join(meeting_contexts)
    prompt = f"""Analyze how the topic of "{topic}" evolved across these GBAC meetings.

{combined}

Provide: 1) How discussion changed over time, 2) Key developments, 3) Recurring themes, 4) Overall trajectory.

Analysis:"""
    try:
        if provider == 'anthropic':
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1500,
                temperature=temperature, messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text, None
        elif provider == 'openai':
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert analyst of government meeting notes."},
                          {"role": "user", "content": prompt}],
                max_tokens=1500, temperature=temperature
            )
            return resp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


def get_all_dates(rag_db: RAGDatabase) -> List[str]:
    all_data = rag_db.collection.get(include=['metadatas'])
    dates = set()
    for meta in (all_data['metadatas'] or []):
        d = _get_meta(meta, 'date', 'meeting_date', default='')
        if d and d != 'Unknown':
            dates.add(d)
    return sorted(dates)


def get_all_sections(rag_db: RAGDatabase) -> List[str]:
    all_data = rag_db.collection.get(include=['metadatas'])
    sections = set()
    for meta in (all_data['metadatas'] or []):
        s = _get_meta(meta, 'section', 'section_type', default='')
        if s and s not in ('General', 'Unknown', ''):
            sections.add(s)
    return sorted(sections)


# ==================== System Loading ====================

@st.cache_resource(show_spinner=False)
def load_system():
    """Auto-load database from ./chroma_db. Returns (rag_db, retriever) or (None, None)."""
    try:
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            return None, None
        rag_db = RAGDatabase(persist_directory=db_path)
        if rag_db.get_count() == 0:
            return None, None
        retriever = HybridRetriever(rag_db)
        return rag_db, retriever
    except Exception:
        return None, None


def get_api_config():
    """Read API key from environment variables. Returns (key, provider) or (None, None)."""
    ak = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    ok = os.environ.get('OPENAI_API_KEY', '').strip()
    if ak and ANTHROPIC_AVAILABLE:
        return ak, 'anthropic'
    if ok and OPENAI_AVAILABLE:
        return ok, 'openai'
    return None, None


# ==================== UI Rendering Helpers ====================

def render_sources_html(sources: List[Dict]) -> str:
    if not sources:
        return ""
    html = '<div class="sources-section"><div class="sources-label">📚 Sources</div>'
    for i, s in enumerate(sources, 1):
        meta = s.get('metadata', {})
        date = _get_meta(meta, 'date', 'meeting_date')
        section = _get_meta(meta, 'section', 'section_type', default='General')
        score = s.get('combined_score', 0)
        preview = s.get('text', '')[:200].replace('<', '&lt;').replace('>', '&gt;').replace('\n', ' ')
        html += f"""
        <div class="source-item">
          <div class="source-num">{i}</div>
          <div class="source-content">
            <div class="source-meta">
              <span class="source-date">📅 {date}</span>
              <span class="source-section-tag">{section}</span>
              <span class="source-score">{score:.2f}</span>
            </div>
            <div class="source-preview">{preview}…</div>
          </div>
        </div>"""
    html += '</div>'
    return html


# ==================== CSS ====================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

/* ── Hide Streamlit chrome ── */
#MainMenu, footer[data-testid="stFooter"],
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"], .stDeployButton,
section[data-testid="stSidebar"],
header[data-testid="stHeader"] { display: none !important; }

/* ── Full-width, zero-padding main container ── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
}
.stApp, body {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
    background: #fff;
    color: #1c2030;
}

/* ── Remove bottom gap on stMarkdown elements ── */
div[data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
.element-container { margin-bottom: 0 !important; }

/* ── Georgetown variables ── */
:root {
    --gu-navy:      #041e42;
    --gu-navy-deep: #021431;
    --gu-blue:      #1a4a8a;
    --gu-blue-acc:  #3d7bd9;
    --gu-blue-pale: #e8eff8;
    --gu-blue-tint: #f3f6fb;
    --gu-gray-100:  #f7f8fa;
    --gu-gray-200:  #ebedf2;
    --gu-gray-400:  #9aa3b1;
    --gu-gray-600:  #5a6475;
    --gu-gray-800:  #2d333f;
    --gu-gold:      #b4975a;
    --gu-gold-lt:   #d4bc82;
    --font-disp:    "Playfair Display", Georgia, serif;
    --font-body:    "Source Sans 3", "Segoe UI", sans-serif;
}

/* ════════════════════════════════
   HEADER SECTIONS
════════════════════════════════ */
.gu-utility-bar {
    background: var(--gu-navy-deep);
    padding: 7px 32px;
    display: flex;
    justify-content: flex-end;
    gap: 0;
}
.gu-utility-bar a {
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 500;
    color: rgba(255,255,255,0.68);
    text-decoration: none;
    letter-spacing: 0.3px;
    transition: color 0.15s;
}
.gu-utility-bar a:hover { color: #fff; }

.gu-logo-bar {
    background: #fff;
    padding: 13px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 8px rgba(4,30,66,0.08);
}
.gu-logo-mark {
    display: flex;
    align-items: center;
    gap: 14px;
    text-decoration: none;
}
.gu-logo-text .school-name {
    font-family: var(--font-disp);
    font-size: 17px;
    font-weight: 700;
    color: var(--gu-navy);
    display: block;
    letter-spacing: -0.2px;
}
.gu-logo-text .prog-name {
    font-size: 11px;
    font-weight: 600;
    color: var(--gu-gray-600);
    letter-spacing: 1.4px;
    text-transform: uppercase;
    display: block;
    margin-top: 2px;
}

.gu-nav-bar { background: var(--gu-navy); }
.gu-nav-list {
    list-style: none;
    display: flex;
    align-items: stretch;
    padding: 0 32px;
    margin: 0;
    gap: 0;
}
.gu-nav-list > li > a {
    display: block;
    padding: 12px 18px;
    font-size: 13.5px;
    font-weight: 500;
    color: rgba(255,255,255,0.82);
    text-decoration: none;
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: color 0.15s, background 0.15s, border-color 0.15s;
}
.gu-nav-list > li > a:hover {
    color: #fff;
    background: rgba(255,255,255,0.07);
    border-bottom-color: var(--gu-gold);
}

/* ════════════════════════════════
   HERO BAND
════════════════════════════════ */
.gu-hero {
    background: linear-gradient(170deg, var(--gu-navy) 0%, var(--gu-navy-deep) 100%);
    padding: 52px 80px 44px;
    position: relative;
    overflow: hidden;
}
.gu-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 80% 20%, rgba(180,151,90,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 20% 80%, rgba(61,123,217,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-crumb {
    position: relative;
    font-size: 12.5px;
    color: rgba(255,255,255,0.5);
    margin-bottom: 18px;
}
.hero-crumb a { color: rgba(255,255,255,0.5); text-decoration: none; }
.hero-crumb a:hover { color: #fff; }
.hero-crumb .sep { margin: 0 7px; opacity: 0.38; }
.hero-title {
    position: relative;
    font-family: var(--font-disp);
    font-size: 38px;
    font-weight: 700;
    color: #fff !important;
    line-height: 1.15;
    margin-bottom: 12px;
    letter-spacing: -0.3px;
}
.hero-subtitle {
    position: relative;
    font-size: 16px;
    font-weight: 300;
    color: rgba(255,255,255,0.65);
    margin-bottom: 22px;
    max-width: 600px;
    line-height: 1.6;
}
.hero-meta-row {
    position: relative;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    background: rgba(180,151,90,0.14);
    border: 1px solid rgba(180,151,90,0.35);
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--gu-gold-lt);
    letter-spacing: 0.4px;
}
.db-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    background: rgba(61,123,217,0.15);
    border: 1px solid rgba(61,123,217,0.3);
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 600;
    color: rgba(180,210,255,0.9);
}

/* ════════════════════════════════
   HERO TABS  (pill inside the navy hero band)
════════════════════════════════ */
.hero-tabs {
    display: inline-flex;
    gap: 2px;
    background: rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 5px;
    border-radius: 50px;
    margin-top: 28px;
}
.hero-tab, .hero-tab:link, .hero-tab:visited {
    padding: 9px 28px;
    border-radius: 50px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: rgba(255,255,255,0.5) !important;
    transition: all 0.2s;
    user-select: none;
    text-decoration: none !important;
    display: inline-block;
    white-space: nowrap;
}
.hero-tab-active, .hero-tab-active:link, .hero-tab-active:visited {
    background: rgba(255,255,255,0.13);
    color: #ffffff !important;
    font-weight: 600;
}
.hero-tab:hover:not(.hero-tab-active) {
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.8) !important;
    text-decoration: none !important;
}

/* ════════════════════════════════
   BODY SECTION
════════════════════════════════ */
.body-wrap {
    background: #fff;
    padding: 24px 80px 80px;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.body-heading {
    width: 100%;
    max-width: 720px;
    font-family: var(--font-disp);
    font-size: 22px;
    font-weight: 600;
    color: #1c2030;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.ai-badge {
    display: inline-block;
    background: var(--gu-blue-pale);
    border: 1px solid #c4d4eb;
    border-radius: 5px;
    font-family: var(--font-body);
    font-size: 10px;
    font-weight: 700;
    color: var(--gu-navy);
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 9px;
}

/* ── Search form ── */
div[data-testid="stForm"] {
    border: 1.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    padding: 10px 16px !important;
    background: var(--gu-navy) !important;
    box-shadow: 0 1px 4px rgba(4,30,66,0.18) !important;
    transition: border-color 0.25s, box-shadow 0.25s, background 0.25s !important;
}
div[data-testid="stForm"]:focus-within {
    border-color: var(--gu-gold-lt) !important;
    box-shadow: 0 2px 18px rgba(4,30,66,0.35) !important;
    background: var(--gu-navy) !important;
}
/* Hide text input label */
div[data-testid="stTextInput"] > label { display: none !important; }
/* Remove input border/bg */
div[data-testid="stTextInput"] > div {
    background: transparent !important;
}
div[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 15px !important;
    font-family: var(--font-body) !important;
    color: #ffffff !important;
    padding: 6px 0 !important;
    caret-color: #ffffff !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: rgba(255,255,255,0.4) !important;
}
/* + add button */
.add-btn-wrap {
    width: 34px; height: 34px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 300;
    color: rgba(255,255,255,0.85);
    cursor: pointer;
    margin-top: 3px;
    flex-shrink: 0;
    transition: all 0.2s;
    user-select: none;
}
.add-btn-wrap:hover { background: rgba(255,255,255,0.18); transform: scale(1.05); }
/* Model selector pill */
.model-sel-wrap {
    display: flex; align-items: center; gap: 5px;
    padding: 5px 11px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 7px;
    font-size: 13px; font-weight: 600;
    color: rgba(255,255,255,0.85);
    white-space: nowrap;
    height: 34px;
    margin-top: 3px;
    cursor: pointer;
    transition: all 0.2s;
}
.model-sel-wrap:hover { background: rgba(255,255,255,0.18); }
.model-sel-wrap .caret { font-size: 10px; margin-left: 2px; }
/* Form submit (send) button — square like .send-btn */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--gu-navy), var(--gu-blue)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0 !important;
    font-size: 18px !important;
    font-weight: 400 !important;
    font-family: var(--font-body) !important;
    width: 34px !important;
    height: 34px !important;
    min-height: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 3px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: scale(1.06) !important;
    box-shadow: 0 4px 16px rgba(4,30,66,0.3) !important;
}

/* ── Quick action chips ── */
.chips-row {
    display: flex;
    gap: 10px;
    flex-wrap: nowrap;
    justify-content: center;
    margin: 6px 0 44px;
    width: 100%;
    max-width: 900px;
    overflow-x: auto;
}
.action-chip, .action-chip:link, .action-chip:visited {
    padding: 14px 22px;
    background: #ffffff;
    border: 1.5px solid #e5e7eb;
    border-radius: 14px;
    cursor: pointer;
    font-size: 14.5px;
    font-family: var(--font-body);
    font-weight: 500;
    color: #1c2030 !important;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    text-decoration: none !important;
    transition: all 0.2s;
    white-space: nowrap;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.action-chip:hover {
    background: #f5f7fa;
    border-color: #c4cdd8;
    color: #041e42 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(4,30,66,0.1);
    text-decoration: none !important;
}

/* ── Section label ── */
.section-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--gu-gray-400);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    text-align: center;
    margin: 4px 0 22px;
    width: 100%;
}

/* ── Resource cards ── */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 20px;
    width: 100%;
    max-width: 900px;
    margin: 0 auto;
}
.gu-card {
    background: #fff;
    border: 1.5px solid var(--gu-gray-200);
    border-radius: 12px;
    padding: 24px;
    cursor: pointer;
    transition: all 0.25s;
    box-shadow: 0 1px 4px rgba(4,30,66,0.04);
}
.gu-card:hover {
    border-color: var(--gu-blue-acc);
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(4,30,66,0.12);
}
.card-icon {
    width: 46px; height: 46px;
    background: var(--gu-blue-pale);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 21px; margin-bottom: 14px;
}
.card-title {
    font-family: var(--font-disp);
    font-size: 15.5px; font-weight: 600;
    color: var(--gu-navy); margin-bottom: 7px;
}
.card-desc { font-size: 13px; color: var(--gu-gray-600); line-height: 1.55; }

/* ── Answer display ── */
.answer-wrap {
    width: 100%;
    animation: fadeUp 0.4s ease-out both;
}
.answer-card {
    background: var(--gu-blue-tint);
    border: 1.5px solid #c4d4eb;
    border-left: 4px solid var(--gu-navy);
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 20px;
    font-size: 15px;
    line-height: 1.7;
    color: #1c2030;
}
.answer-label {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--gu-navy);
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* ── Source citations ── */
.sources-section { margin-top: 4px; }
.sources-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--gu-gray-400);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.source-item {
    display: flex;
    gap: 13px;
    padding: 13px 16px;
    background: var(--gu-gray-100);
    border: 1.5px solid var(--gu-gray-200);
    border-radius: 9px;
    margin-bottom: 9px;
    transition: border-color 0.15s;
}
.source-item:hover { border-color: #c4d4eb; }
.source-num {
    min-width: 26px; height: 26px;
    background: var(--gu-blue-pale);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11.5px; font-weight: 700;
    color: var(--gu-navy);
    flex-shrink: 0; margin-top: 1px;
}
.source-content { flex: 1; min-width: 0; }
.source-meta {
    display: flex; flex-wrap: wrap; gap: 7px;
    margin-bottom: 5px; align-items: center;
}
.source-date { font-size: 12px; font-weight: 600; color: var(--gu-navy); }
.source-section-tag {
    font-size: 11.5px; color: var(--gu-gray-600);
    background: var(--gu-gray-200); border-radius: 4px; padding: 2px 8px;
}
.source-score { font-size: 11px; color: var(--gu-gray-400); margin-left: auto; }
.source-preview {
    font-size: 13px; color: var(--gu-gray-600); line-height: 1.5;
    overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}

/* ── Query bubble ── */
.query-bubble {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
}
.query-bubble-inner {
    background: var(--gu-navy);
    color: #fff;
    padding: 11px 18px;
    border-radius: 14px 14px 4px 14px;
    font-size: 15px;
    max-width: 85%;
    line-height: 1.5;
}
.edit-bubble-btn {
    flex-shrink: 0;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 6px;
    background: transparent;
    color: rgba(4,30,66,0.35);
    font-size: 13px;
    text-decoration: none !important;
    transition: all 0.2s;
    opacity: 0;
}
.query-bubble:hover .edit-bubble-btn {
    opacity: 1;
    background: var(--gu-gray-100);
    color: var(--gu-navy);
}
.edit-indicator {
    font-size: 12.5px;
    color: var(--gu-blue-acc);
    font-family: var(--font-body);
    margin-bottom: 6px;
    padding: 6px 10px;
    background: var(--gu-blue-pale);
    border-left: 3px solid var(--gu-blue-acc);
    border-radius: 0 6px 6px 0;
}

/* ── Analysis/Documents tool cards ── */
.tool-card {
    background: var(--gu-gray-100);
    border: 1.5px solid var(--gu-gray-200);
    border-radius: 11px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.tool-card h4 {
    font-family: var(--font-disp);
    font-size: 15px; font-weight: 600;
    color: var(--gu-navy); margin-bottom: 12px;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stTextInput"] label {
    font-size: 13px !important;
    color: var(--gu-gray-600) !important;
    font-weight: 500 !important;
    font-family: var(--font-body) !important;
}
/* Primary action buttons in body */
div[data-testid="stButton"] > button {
    background: #ffffff !important;
    color: var(--gu-navy) !important;
    border: 1.5px solid #c4cdd8 !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 22px !important;
    height: auto !important;
    min-height: unset !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 4px rgba(4,30,66,0.06) !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--gu-blue-pale) !important;
    border-color: var(--gu-blue-acc) !important;
    color: var(--gu-navy) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(4,30,66,0.12) !important;
}

/* ── No-results / error ── */
.empty-state {
    text-align: center; padding: 40px 20px;
    color: var(--gu-gray-600); font-size: 15px;
}
.empty-state .icon { font-size: 36px; margin-bottom: 12px; }

/* ── Footer ── */
.gu-footer { background: var(--gu-navy-deep); color: #fff; }
.footer-main {
    max-width: 1280px; margin: 0 auto;
    padding: 48px 32px 36px;
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr;
    gap: 48px;
}
.footer-brand-link {
    display: flex; align-items: center; gap: 12px;
    text-decoration: none; margin-bottom: 16px;
}
.footer-school { font-family: var(--font-disp); font-size: 15.5px; font-weight: 700; color: #fff; display: block; }
.footer-prog { font-size: 11px; font-weight: 600; color: var(--gu-gold-lt); letter-spacing: 1.2px; text-transform: uppercase; display: block; margin-top: 2px; }
.footer-desc { font-size: 13.5px; color: rgba(255,255,255,0.58); line-height: 1.7; margin-bottom: 18px; }
.footer-col h4 { font-family: var(--font-disp); font-size: 14px; font-weight: 700; color: var(--gu-gold-lt); margin-bottom: 14px; }
.footer-col a { display: block; font-size: 13.5px; color: rgba(255,255,255,0.6); text-decoration: none; line-height: 2.1; }
.footer-col a:hover { color: #fff; }
.footer-col p { font-size: 13.5px; color: rgba(255,255,255,0.6); line-height: 1.8; }
.footer-divider { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 0; }
.footer-bottom {
    max-width: 1280px; margin: 0 auto;
    padding: 17px 32px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}
.footer-legal { display: flex; flex-wrap: wrap; align-items: center; }
.footer-legal a { font-size: 12.5px; color: rgba(255,255,255,0.45); text-decoration: none; padding: 0 13px; border-right: 1px solid rgba(255,255,255,0.13); transition: color 0.15s; }
.footer-legal a:first-child { padding-left: 0; }
.footer-legal a:last-child { border-right: none; }
.footer-legal a:hover { color: rgba(255,255,255,0.8); }
.footer-copy { font-size: 12.5px; color: rgba(255,255,255,0.36); }

/* ── Animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.answer-wrap, .source-item, .gu-card { animation: fadeUp 0.45s ease-out both; }

/* ── Responsive ── */
@media (max-width: 1024px) {
    .gu-hero { padding: 44px 40px 40px; }
    .body-wrap { padding: 20px 40px 80px; }
}
@media (max-width: 768px) {
    .gu-hero { padding: 36px 20px 40px; }
    .hero-title { font-size: 26px; }
    .body-wrap { padding: 18px 20px 60px; }
    .cards-grid { grid-template-columns: 1fr; }
    .footer-main { grid-template-columns: 1fr; gap: 28px; padding: 32px 20px 24px; }
    .gu-nav-list { display: none !important; }
    .footer-bottom { flex-direction: column; align-items: flex-start; padding: 14px 20px; }
    .tab-strip-wrap { padding: 14px 20px 16px; }
}
</style>
"""

# ==================== Static HTML Sections ====================

HEADER_HTML = """
<div class="gu-utility-bar">
  <a href="https://mccourt.georgetown.edu/news/">News</a>
  <a href="https://mccourt.georgetown.edu/about-us/">About Us</a>
  <a href="https://gradapply.georgetown.edu/apply/">Apply Now</a>
</div>
<div class="gu-logo-bar">
  <a class="gu-logo-mark" href="https://mccourt.georgetown.edu/tech-and-public-policy/">
    <svg width="42" height="42" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="2" y="2" width="44" height="44" rx="6" fill="#041E42"/>
      <path d="M24 8L14 14V28C14 34 18 39 24 41C30 39 34 34 34 28V14L24 8Z" fill="none" stroke="#b4975a" stroke-width="1.8"/>
      <path d="M24 13L17 17.5V28C17 32.5 20 36 24 37.5C28 36 31 32.5 31 28V17.5L24 13Z" fill="rgba(180,151,90,0.15)"/>
      <text x="24" y="29" text-anchor="middle" font-family="Georgia,serif" font-size="14" font-weight="bold" fill="#b4975a">G</text>
    </svg>
    <span class="gu-logo-text">
      <span class="school-name">McCourt School of Public Policy</span>
      <span class="prog-name">Tech &amp; Public Policy</span>
    </span>
  </a>
</div>
<nav class="gu-nav-bar">
  <ul class="gu-nav-list">
    <li><a href="https://mccourt.georgetown.edu/tech-and-public-policy/">TPP Home</a></li>
    <li><a href="https://mccourt.georgetown.edu/tech-and-public-policy/about-us-2/">About</a></li>
    <li><a href="https://mccourt.georgetown.edu/tech-and-public-policy/our-work/">Our Work</a></li>
    <li><a href="https://mccourt.georgetown.edu/programs/">Programs</a></li>
    <li><a href="https://mccourt.georgetown.edu/news/tag/tech-public-policy/">News</a></li>
    <li><a href="https://events.georgetown.edu/tech-public-policy">Events</a></li>
  </ul>
</nav>
"""

FOOTER_HTML = """
<footer class="gu-footer">
  <div class="footer-main">
    <div>
      <a class="footer-brand-link" href="https://mccourt.georgetown.edu">
        <svg width="36" height="36" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="2" y="2" width="44" height="44" rx="6" fill="rgba(255,255,255,0.08)"/>
          <path d="M24 8L14 14V28C14 34 18 39 24 41C30 39 34 34 34 28V14L24 8Z" fill="none" stroke="rgba(180,151,90,0.6)" stroke-width="1.8"/>
          <text x="24" y="29" text-anchor="middle" font-family="Georgia,serif" font-size="14" font-weight="bold" fill="rgba(180,151,90,0.7)">G</text>
        </svg>
        <span>
          <span class="footer-school">McCourt School of Public Policy</span>
          <span class="footer-prog">Tech &amp; Public Policy</span>
        </span>
      </a>
      <p class="footer-desc">The TPP program at McCourt works to shape technology's promise for a better world through cross-disciplinary research, emerging leaders, and policy innovation.</p>
    </div>
    <div class="footer-col">
      <h4>Quick Links</h4>
      <a href="https://mccourt.georgetown.edu/tech-and-public-policy/about-us-2/">About TPP</a>
      <a href="https://mccourt.georgetown.edu/tech-and-public-policy/our-work/">Our Work</a>
      <a href="https://mccourt.georgetown.edu/programs/">Programs</a>
      <a href="https://mccourt.georgetown.edu/research/">Research</a>
      <a href="https://mccourt.georgetown.edu/news/tag/tech-public-policy/">TPP News</a>
    </div>
    <div class="footer-col">
      <h4>Contact</h4>
      <p>McCourt School of Public Policy</p>
      <p>Georgetown University</p>
      <p>125 E St. NW, Washington, DC 20001</p>
      <p style="margin-top:10px"><a href="tel:202-687-5932" style="display:inline">P. (202) 687-5932</a></p>
    </div>
  </div>
  <hr class="footer-divider"/>
  <div class="footer-bottom">
    <div class="footer-legal">
      <a href="https://www.georgetown.edu/privacy-policy/">Privacy Policy</a>
      <a href="https://www.library.georgetown.edu/copyright">Copyright</a>
      <a href="https://accessibility.georgetown.edu/">Accessibility</a>
      <a href="https://ideaa.georgetown.edu/notice-of-non-discrimination/">Non-Discrimination</a>
    </div>
    <span class="footer-copy">© 2026 McCourt School of Public Policy, Georgetown University</span>
  </div>
</footer>
"""


# ==================== Tab Content Renderers ====================

def render_chat_tab(rag_db, retriever, api_key, provider):
    _, col, _ = st.columns([1, 4, 1])
    with col:
        st.markdown(
            '<div class="body-heading">Consult GBAC through TPP Labs '
            '<span class="ai-badge">AI-Powered</span></div>',
            unsafe_allow_html=True
        )

        has_messages = bool(st.session_state.messages)
        editing_idx = st.session_state.get('editing_idx', None)
        auto_q = st.session_state.pop('auto_query', None)

        # ── Empty state: form at top + chips + resource cards ──
        if not has_messages:
            with st.form("search_form", border=False, clear_on_submit=True):
                c_plus, c_input, c_model, c_send = st.columns([0.5, 7, 2, 0.55])
                with c_plus:
                    st.markdown('<div class="add-btn-wrap">+</div>', unsafe_allow_html=True)
                with c_input:
                    query_input = st.text_input(
                        "", value='', placeholder="Ask about GBAC",
                        label_visibility="collapsed", key="main_query_input"
                    )
                with c_model:
                    st.markdown(
                        '<div class="model-sel-wrap">TPP 4.5<span class="caret">▼</span></div>',
                        unsafe_allow_html=True
                    )
                with c_send:
                    submitted = st.form_submit_button("↑", use_container_width=True)

            st.markdown("""
            <div class="chips-row">
              <a class="action-chip" href="?chip=0" target="_self">📋 Analyze Documents</a>
              <a class="action-chip" href="?chip=1" target="_self">📊 Policy Research</a>
              <a class="action-chip" href="?chip=2" target="_self">💡 Best Practices</a>
              <a class="action-chip" href="?chip=3" target="_self">🎯 TPP Insights</a>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label">Get started with TPP Labs resources</div>',
                unsafe_allow_html=True
            )

            cards = [
                ("📚", "Policy Analysis",
                 "Review and analyze technology policy research, regulatory frameworks, and legislative developments"),
                ("🔍", "AI Governance Research",
                 "Search through AI safety standards, algorithmic accountability guidelines, and governance frameworks"),
                ("📈", "Data Projects",
                 "Build predictive models and analysis pipelines for public interest technology research"),
            ]
            card_html = '<div class="cards-grid">'
            for icon, title, desc in cards:
                card_html += f"""
                <div class="gu-card">
                  <div class="card-icon">{icon}</div>
                  <div class="card-title">{title}</div>
                  <div class="card-desc">{desc}</div>
                </div>"""
            card_html += '</div>'
            st.markdown(card_html, unsafe_allow_html=True)

        # ── Active conversation: history → form at bottom ──
        else:
            # Render conversation history
            for i, msg in enumerate(st.session_state.messages):
                if msg['role'] == 'user':
                    st.markdown(
                        f'<div class="query-bubble">'
                        f'<a class="edit-bubble-btn" href="?edit={i}" target="_self" title="Edit message">✏</a>'
                        f'<div class="query-bubble-inner">{msg["content"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    answer_text = msg.get('content')
                    sources = msg.get('sources', [])
                    error = msg.get('error')

                    if error and not answer_text:
                        st.markdown(
                            f'<div class="answer-card">'
                            f'<div class="answer-label">⚠ Note</div>'
                            f'{error}</div>',
                            unsafe_allow_html=True
                        )
                    elif answer_text:
                        st.markdown(
                            f'<div class="answer-wrap">'
                            f'<div class="answer-card">'
                            f'<div class="answer-label">✦ GBAC Intelligence</div>'
                            f'{answer_text}'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )
                    elif sources:
                        st.markdown(
                            '<div class="answer-wrap"><div class="answer-card">'
                            '<div class="answer-label">📄 Relevant Passages</div>'
                            'No AI key configured — showing the most relevant excerpts from meeting notes.'
                            '</div></div>',
                            unsafe_allow_html=True
                        )
                    if sources:
                        st.markdown(render_sources_html(sources), unsafe_allow_html=True)

            # New conversation button
            st.markdown('<br>', unsafe_allow_html=True)
            if st.button("↩ New conversation", key="clear_chat"):
                st.session_state.messages = []
                st.session_state.prefill = ''
                st.session_state.editing_idx = None
                st.rerun()

            st.markdown('<br>', unsafe_allow_html=True)

            # Edit indicator
            if editing_idx is not None:
                st.markdown(
                    '<div class="edit-indicator">✏ Editing — submit to replace from this message onward</div>',
                    unsafe_allow_html=True
                )

            # Form at the bottom
            with st.form("search_form", border=False, clear_on_submit=True):
                c_plus, c_input, c_model, c_send = st.columns([0.5, 7, 2, 0.55])
                with c_plus:
                    st.markdown('<div class="add-btn-wrap">+</div>', unsafe_allow_html=True)
                with c_input:
                    query_input = st.text_input(
                        "", value=st.session_state.get('prefill', ''),
                        placeholder="Ask about GBAC",
                        label_visibility="collapsed", key="main_query_input"
                    )
                with c_model:
                    st.markdown(
                        '<div class="model-sel-wrap">TPP 4.5<span class="caret">▼</span></div>',
                        unsafe_allow_html=True
                    )
                with c_send:
                    submitted = st.form_submit_button("↑", use_container_width=True)

        # ── Handle submission ──
        run_query = None
        if submitted and query_input.strip():
            run_query = query_input.strip()
            st.session_state.prefill = ''
        elif auto_q:
            run_query = auto_q

        if run_query:
            if not retriever:
                st.error("Database not loaded. Please check that ./chroma_db exists and contains data.")
            else:
                # If editing, truncate conversation from that message onward
                if editing_idx is not None:
                    st.session_state.messages = st.session_state.messages[:editing_idx]
                    st.session_state.editing_idx = None

                with st.spinner("Searching GBAC meeting notes…"):
                    answer, sources, error = rag_answer(
                        run_query, retriever, api_key, provider, n_results=25
                    )

                st.session_state.messages.append({'role': 'user', 'content': run_query})
                st.session_state.messages.append(
                    {'role': 'assistant', 'content': answer, 'sources': sources, 'error': error}
                )
                st.rerun()


def render_analysis_tab(rag_db, retriever, api_key, provider):
    _, col, _ = st.columns([1, 4, 1])
    with col:
        st.markdown(
            '<div class="body-heading">Meeting Analysis '
            '<span class="ai-badge">AI-Powered</span></div>',
            unsafe_allow_html=True
        )

        if not api_key:
            st.info("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment to enable AI analysis.")
            return
        if not rag_db:
            st.error("Database not loaded.")
            return

        available_dates = get_all_dates(rag_db)

        # Summarize a meeting
        st.markdown('<div class="tool-card"><h4>📋 Summarize a Meeting</h4>', unsafe_allow_html=True)
        summary_date = st.selectbox("Select meeting date:", available_dates, key="summary_date_sel") if available_dates else st.text_input("Meeting date (e.g. October 2, 2024):", key="summary_date_input")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Generate Summary", key="sum_btn"):
            if summary_date:
                with st.spinner(f"Summarizing meeting from {summary_date}…"):
                    text, err = summarize_meeting(summary_date, rag_db, api_key, provider)
                if text:
                    st.markdown(
                        f'<div class="answer-card"><div class="answer-label">📋 Summary — {summary_date}</div>{text}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error(err or "No data found for this date.")
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Compare meetings
        st.markdown('<div class="tool-card"><h4>📊 Compare Across Meetings</h4>', unsafe_allow_html=True)
        compare_topic = st.text_input("Topic to track:", placeholder="e.g. electrification requirements", key="cmp_topic", label_visibility="visible")
        compare_dates = st.multiselect("Select 2–5 meetings to compare:", available_dates, key="cmp_dates")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Compare Meetings", key="cmp_btn"):
            if not compare_topic:
                st.warning("Please enter a topic.")
            elif len(compare_dates) < 2:
                st.warning("Select at least 2 meeting dates.")
            else:
                with st.spinner(f"Comparing '{compare_topic}' across {len(compare_dates)} meetings…"):
                    text, err = compare_meetings(compare_dates, compare_topic, retriever, api_key, provider)
                if text:
                    st.markdown(
                        f'<div class="answer-card"><div class="answer-label">📊 Topic Evolution — {compare_topic}</div>{text}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.error(err or "No relevant content found.")
        st.markdown('</div></div>', unsafe_allow_html=True)


def render_documents_tab(rag_db, retriever):
    _, col, _ = st.columns([1, 4, 1])
    with col:
        st.markdown('<div class="body-heading">Browse Documents</div>', unsafe_allow_html=True)

        if not rag_db:
            st.error("Database not loaded.")
            return

        # List dates and sections
        st.markdown('<div class="tool-card"><h4>📅 Available Meetings &amp; Sections</h4>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("List All Meeting Dates", key="list_dates_btn", use_container_width=True):
                dates = get_all_dates(rag_db)
                st.markdown(
                    '<div class="sources-section"><div class="sources-label">Meeting Dates</div>' +
                    ''.join(f'<div class="source-item"><div class="source-num">📅</div><div class="source-content"><div class="source-date">{d}</div></div></div>' for d in dates) +
                    '</div>',
                    unsafe_allow_html=True
                )
        with c2:
            if st.button("List All Section Types", key="list_secs_btn", use_container_width=True):
                sections = get_all_sections(rag_db)
                st.markdown(
                    '<div class="sources-section"><div class="sources-label">Section Types</div>' +
                    ''.join(f'<div class="source-item"><div class="source-num">📑</div><div class="source-content"><div class="source-date">{s}</div></div></div>' for s in sections) +
                    '</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # Find by section
        st.markdown('<div class="tool-card"><h4>📑 Find by Section Type</h4>', unsafe_allow_html=True)
        sec_q = st.text_input("Section name (partial match):", placeholder="e.g. Discussion, Action Items, Attendees", key="sec_query", label_visibility="visible")
        sec_n = st.slider("Max results:", 5, 30, 10, key="sec_n")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Search by Section", key="sec_btn"):
            if sec_q:
                all_data = rag_db.collection.get(include=['documents', 'metadatas'])
                matches = []
                for doc, meta in zip(all_data['documents'] or [], all_data['metadatas'] or []):
                    sec = _get_meta(meta, 'section', 'section_type', default='')
                    if sec_q.lower() in sec.lower():
                        matches.append({'text': doc, 'metadata': meta, 'combined_score': 1.0})
                    if len(matches) >= sec_n:
                        break
                if matches:
                    st.success(f"Found {len(matches)} chunks in sections matching '{sec_q}'")
                    st.markdown(render_sources_html(matches), unsafe_allow_html=True)
                else:
                    st.warning(f"No sections found matching '{sec_q}'")
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Date range search
        st.markdown('<div class="tool-card"><h4>📅 Search Within Date Range</h4>', unsafe_allow_html=True)
        dr_query = st.text_input("Search query:", key="dr_query", label_visibility="visible")
        c1, c2 = st.columns(2)
        with c1:
            dr_start = st.text_input("Start date:", placeholder="February 1, 2024", key="dr_start", label_visibility="visible")
        with c2:
            dr_end = st.text_input("End date:", placeholder="December 31, 2024", key="dr_end", label_visibility="visible")
        dr_n = st.slider("Max results:", 5, 20, 8, key="dr_n")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Search", key="dr_btn"):
            if not dr_query:
                st.warning("Please enter a search query.")
            elif not dr_start or not dr_end:
                st.warning("Please enter both start and end dates.")
            else:
                try:
                    start_dt = datetime.strptime(dr_start, "%B %d, %Y")
                    end_dt = datetime.strptime(dr_end, "%B %d, %Y")
                    candidates = retriever.retrieve(dr_query, top_k=dr_n * 3)
                    filtered = []
                    for r in candidates:
                        ds = _get_meta(r['metadata'], 'date', 'meeting_date', default='')
                        try:
                            if start_dt <= datetime.strptime(ds, "%B %d, %Y") <= end_dt:
                                filtered.append(r)
                        except ValueError:
                            pass
                    filtered = filtered[:dr_n]
                    if filtered:
                        st.success(f"Found {len(filtered)} results between {dr_start} and {dr_end}")
                        st.markdown(render_sources_html(filtered), unsafe_allow_html=True)
                    else:
                        st.warning("No results found in that date range. Check the date format: 'Month DD, YYYY'")
                except ValueError:
                    st.error("Invalid date format. Use 'Month DD, YYYY' e.g. 'February 5, 2025'")
        st.markdown('</div></div>', unsafe_allow_html=True)


# ==================== Main ====================

def main():
    st.set_page_config(
        page_title="TPP Labs — Experiments in Policy Intelligence | McCourt",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Session state defaults
    for key, val in [
        ('active_tab', 'Chat'),
        ('messages', []),
        ('prefill', ''),
        ('editing_idx', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    # Silent auto-load
    rag_db, retriever = load_system()
    api_key, provider = get_api_config()

    # Tab / chip from URL query params set by anchor links
    _CHIP_QUERIES = [
        "What were the main green building policies discussed across GBAC meetings?",
        "What key decisions were made in recent GBAC meetings?",
        "What were the most frequently discussed topics in GBAC meetings?",
        "What action items were assigned in GBAC meetings and who is responsible?",
    ]
    try:
        _tp = st.query_params.get('tab', None)
        _cp = st.query_params.get('chip', None)
        _ep = st.query_params.get('edit', None)
        if _tp in ('Chat', 'Analysis', 'Documents'):
            st.session_state.active_tab = _tp
        if _cp is not None:
            try:
                _ci = int(_cp)
                if 0 <= _ci < len(_CHIP_QUERIES):
                    st.session_state.auto_query = _CHIP_QUERIES[_ci]
                    st.session_state.active_tab = 'Chat'
            except (ValueError, TypeError):
                pass
        if _ep is not None:
            try:
                _ei = int(_ep)
                msgs = st.session_state.messages
                if 0 <= _ei < len(msgs) and msgs[_ei]['role'] == 'user':
                    st.session_state.editing_idx = _ei
                    st.session_state.prefill = msgs[_ei]['content']
                    st.query_params.pop('edit', None)
            except (ValueError, TypeError, IndexError):
                pass
    except AttributeError:
        pass  # Streamlit < 1.30

    # ── Header ──
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    # ── Hero with tabs as plain anchor links ──
    active = st.session_state.active_tab

    def _tab_cls(key):
        return 'hero-tab hero-tab-active' if active == key else 'hero-tab'

    st.markdown(f"""
    <div class="gu-hero">
      <div class="hero-crumb">
        <a href="https://mccourt.georgetown.edu">McCourt School</a>
        <span class="sep">›</span>
        <a href="https://mccourt.georgetown.edu/tech-and-public-policy/">Tech &amp; Public Policy</a>
        <span class="sep">›</span>
        TPP Labs
      </div>
      <h1 class="hero-title">TPP Labs — Experiments in Policy Intelligence</h1>
      <p class="hero-subtitle">
        Powered by the McCourt School's Tech &amp; Public Policy Lab for experimental use only<br/>
        AI-assisted research and policy analysis for the public good
      </p>
      <div class="hero-meta-row">
        <span class="hero-badge">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
          </svg>
          TPP Labs | McCourt School of Public Policy
        </span>
      </div>
      <div class="hero-tabs">
        <a class="{_tab_cls('Chat')}"      href="?tab=Chat"      target="_self">💬 Chat</a>
        <a class="{_tab_cls('Analysis')}"  href="?tab=Analysis"  target="_self">📊 Analysis</a>
        <a class="{_tab_cls('Documents')}" href="?tab=Documents" target="_self">📂 Documents</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Body ──
    st.markdown('<div class="body-wrap">', unsafe_allow_html=True)
    if active == 'Chat':
        render_chat_tab(rag_db, retriever, api_key, provider)
    elif active == 'Analysis':
        render_analysis_tab(rag_db, retriever, api_key, provider)
    elif active == 'Documents':
        render_documents_tab(rag_db, retriever)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
