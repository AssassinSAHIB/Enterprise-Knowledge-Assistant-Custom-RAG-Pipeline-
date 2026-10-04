"""
Week 4 - Retro Glassmorphism Web Interface & End-to-End Integration
Module: app.py
Enterprise Knowledge Assistant (Custom RAG Pipeline)

Assembles ingestion.py + retrieval.py + llm_generator.py into a single
interactive Streamlit application with Digital Retro Dark/Bright themes
and Liquid Glass (Glassmorphism) UI elements.
"""

import time
import tempfile
import os
import sys

import streamlit as st
import numpy as np

# Ensure the project directory is in path for local module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import extract_text_from_pdf, custom_text_splitter
from retrieval import generate_embeddings, search_top_k
from llm_generator import build_grounded_prompt, query_llm_api

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Retro Dark"
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = []
if "latency" not in st.session_state:
    st.session_state.latency = None


# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(theme: str) -> None:
    is_dark = theme == "Retro Dark"

    # ── Colour tokens ──────────────────────────────────────────────────────
    if is_dark:
        bg_primary        = "#0a0e1a"
        bg_secondary      = "#0d1224"
        text_primary      = "#e0f7ff"
        text_secondary    = "#8ab8c8"
        accent_cyan       = "#00f3ff"
        accent_magenta    = "#ff007f"
        grid_color        = "rgba(0, 243, 255, 0.03)"
        glass_bg          = "rgba(18, 24, 38, 0.55)"
        glass_border      = "rgba(0, 243, 255, 0.2)"
        glass_shadow      = "0 8px 32px 0 rgba(0, 0, 0, 0.37)"
        input_bg          = "rgba(10, 20, 40, 0.7)"
        btn_text          = "#00f3ff"
        btn_bg            = "rgba(0, 243, 255, 0.08)"
        btn_border        = "rgba(0, 243, 255, 0.35)"
        btn_hover_shadow  = "0 0 15px rgba(0, 243, 255, 0.6), 0 0 30px rgba(0, 243, 255, 0.2)"
        header_shadow     = "0 0 20px rgba(0, 243, 255, 0.7), 0 0 60px rgba(0, 243, 255, 0.3)"
        tag_bg            = "rgba(0, 243, 255, 0.12)"
        tag_color         = "#00f3ff"
        divider_color     = "rgba(0, 243, 255, 0.15)"
        expander_bg       = "rgba(10, 20, 40, 0.65)"
        metric_accent     = "#00f3ff"
        scrollbar_thumb   = "rgba(0, 243, 255, 0.4)"
        scrollbar_track   = "rgba(0, 0, 0, 0.2)"
    else:
        bg_primary        = "#f4ebd0"
        bg_secondary      = "#f0e6d2"
        text_primary      = "#1a1208"
        text_secondary    = "#5a4a30"
        accent_cyan       = "#c0392b"
        accent_magenta    = "#8e44ad"
        grid_color        = "rgba(180, 140, 80, 0.06)"
        glass_bg          = "rgba(255, 255, 255, 0.45)"
        glass_border      = "rgba(255, 255, 255, 0.6)"
        glass_shadow      = "0 8px 32px 0 rgba(180, 160, 120, 0.25)"
        input_bg          = "rgba(255, 248, 235, 0.75)"
        btn_text          = "#6b2d0c"
        btn_bg            = "rgba(200, 150, 80, 0.15)"
        btn_border        = "rgba(180, 120, 50, 0.45)"
        btn_hover_shadow  = "0 0 15px rgba(192, 80, 30, 0.45), 0 0 30px rgba(192, 80, 30, 0.15)"
        header_shadow     = "0 0 20px rgba(180, 80, 30, 0.5), 0 0 60px rgba(180, 80, 30, 0.2)"
        tag_bg            = "rgba(180, 110, 40, 0.15)"
        tag_color         = "#7b3b0c"
        divider_color     = "rgba(180, 120, 50, 0.25)"
        expander_bg       = "rgba(255, 248, 230, 0.65)"
        metric_accent     = "#c0392b"
        scrollbar_thumb   = "rgba(180, 110, 40, 0.5)"
        scrollbar_track   = "rgba(200, 170, 120, 0.15)"

    # ── Import Google Font ─────────────────────────────────────────────────
    font_import = "@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');"

    css = f"""
<style>
{font_import}

/* ── GLOBAL RESET & TYPOGRAPHY ─────────────────────────────────────── */
*, *::before, *::after {{
    box-sizing: border-box;
}}

html, body, [class*="css"], .stApp {{
    font-family: 'Space Mono', 'Courier New', monospace !important;
    background-color: {bg_primary} !important;
    color: {text_primary} !important;
}}

/* Digital grid background */
.stApp {{
    background-color: {bg_primary} !important;
    background-image:
        linear-gradient({grid_color} 1px, transparent 1px),
        linear-gradient(90deg, {grid_color} 1px, transparent 1px);
    background-size: 32px 32px;
    min-height: 100vh;
}}

/* ── MAIN CONTENT AREA ─────────────────────────────────────────────── */
.main .block-container {{
    padding: 2rem 3rem 4rem !important;
    max-width: 1200px;
}}

/* ── HEADER ────────────────────────────────────────────────────────── */
.eka-header {{
    font-size: clamp(1.1rem, 2.5vw, 1.6rem);
    font-weight: 700;
    letter-spacing: 0.12em;
    color: {accent_cyan} !important;
    text-shadow: {header_shadow};
    padding: 1.4rem 2rem;
    border: 1px solid {glass_border};
    border-radius: 12px;
    background: {glass_bg};
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    box-shadow: {glass_shadow};
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.eka-header::before {{
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
    animation: shimmer 4s infinite;
}}
@keyframes shimmer {{
    0%   {{ left: -100%; }}
    100% {{ left: 200%; }}
}}

/* ── SIDEBAR GLASS ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {glass_bg} !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border-right: 1px solid {glass_border} !important;
    box-shadow: {glass_shadow} !important;
}}
[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {{
    font-family: 'Space Mono', 'Courier New', monospace !important;
}}

/* Keep Streamlit's Material icon font so ligatures (e.g. "visibility",
   "upload") render as icons instead of raw text. */
[data-testid="stIconMaterial"],
[class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded' !important;
    letter-spacing: normal !important;
    text-transform: none !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stTextInput label {{
    color: {accent_cyan} !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}}
.sidebar-brand {{
    color: {accent_cyan} !important;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-shadow: 0 0 10px {accent_cyan};
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid {divider_color};
    margin-bottom: 1rem;
    text-align: center;
}}

/* ── TEXT INPUTS ────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    font-family: 'Space Mono', 'Courier New', monospace !important;
    background: {input_bg} !important;
    color: {text_primary} !important;
    border: 1px solid {glass_border} !important;
    border-radius: 8px !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    padding: 0.75rem 1rem !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {accent_cyan} !important;
    box-shadow: 0 0 0 2px {accent_cyan}33, inset 0 0 0 1px {accent_cyan}22 !important;
    outline: none !important;
}}
.stTextInput > label, .stTextArea > label {{
    color: {accent_cyan} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}}

/* Input wrapper (holds the field + the password eye toggle). The border
   lives on the wrapper so the toggle button sits inside the same box. */
[data-testid="stTextInputRootElement"],
.stTextInput [data-baseweb="input"] {{
    background: {input_bg} !important;
    border: 1px solid {glass_border} !important;
    border-radius: 8px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}}
[data-testid="stTextInputRootElement"]:focus-within,
.stTextInput [data-baseweb="input"]:focus-within {{
    border-color: {accent_cyan} !important;
    box-shadow: 0 0 0 2px {accent_cyan}33 !important;
}}
[data-testid="stTextInputRootElement"] input,
.stTextInput [data-baseweb="input"] input,
.stTextInput [data-baseweb="base-input"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stTextInputRootElement"] > div,
[data-testid="stTextInputRootElement"] button,
.stTextInput [data-baseweb="input"] button {{
    background: transparent !important;
    border: none !important;
}}
[data-testid="stTextInputRootElement"] button span,
.stTextInput [data-baseweb="input"] button span {{
    color: {text_secondary} !important;
}}
[data-testid="stTextInputRootElement"] button:hover span,
.stTextInput [data-baseweb="input"] button:hover span {{
    color: {accent_cyan} !important;
}}

/* Password / masked input */
.stTextInput [type="password"] {{
    letter-spacing: 0.15em;
}}

/* ── BUTTONS ─────────────────────────────────────────────────────────── */
.stButton > button {{
    font-family: 'Space Mono', 'Courier New', monospace !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {btn_text} !important;
    background: {btn_bg} !important;
    border: 1px solid {btn_border} !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.4rem !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    box-shadow: {glass_shadow} !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    box-shadow: {btn_hover_shadow} !important;
    transform: translateY(-2px) !important;
    border-color: {accent_cyan} !important;
}}
.stButton > button:active {{
    transform: translateY(0px) !important;
}}

/* ── SLIDERS ─────────────────────────────────────────────────────────── */
[data-testid="stSlider"] .stSlider > div > div > div > div {{
    background: {accent_cyan} !important;
}}
[data-testid="stSlider"] label {{
    color: {accent_cyan} !important;
}}

/* ── SELECT BOX ──────────────────────────────────────────────────────── */
.stSelectbox > div > div {{
    background: {input_bg} !important;
    border: 1px solid {glass_border} !important;
    border-radius: 8px !important;
    color: {text_primary} !important;
}}

/* ── FILE UPLOADER ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    border: 1px dashed {glass_border} !important;
    border-radius: 10px !important;
    background: {glass_bg} !important;
    backdrop-filter: blur(10px) !important;
    padding: 0.5rem;
    transition: border-color 0.3s ease;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {input_bg} !important;
    border-radius: 8px !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background: {btn_bg} !important;
    border: 1px solid {btn_border} !important;
    color: {btn_text} !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover {{
    border-color: {accent_cyan} !important;
}}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: {text_secondary} !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {accent_cyan} !important;
}}

/* ── STATUS / METRIC CARDS ────────────────────────────────────────────── */
.status-card {{
    background: {glass_bg};
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid {glass_border};
    border-radius: 12px;
    box-shadow: {glass_shadow};
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
}}
.status-card-title {{
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {accent_cyan};
    margin-bottom: 0.8rem;
    font-weight: 700;
}}
.metric-row {{
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}}
.metric-item {{
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}}
.metric-label {{
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {text_secondary};
}}
.metric-value {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {metric_accent};
    text-shadow: 0 0 8px {metric_accent}55;
}}

/* ── RESPONSE CARD ────────────────────────────────────────────────────── */
.response-card {{
    background: {glass_bg};
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid {glass_border};
    border-radius: 12px;
    box-shadow: {glass_shadow};
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    position: relative;
}}
.response-card-header {{
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {accent_cyan};
    margin-bottom: 1rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.response-card-header::before {{
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {accent_cyan};
    box-shadow: 0 0 6px {accent_cyan};
    animation: pulse-dot 1.8s ease-in-out infinite;
}}
@keyframes pulse-dot {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%       {{ opacity: 0.4; transform: scale(0.7); }}
}}
.response-body {{
    font-size: 0.88rem;
    line-height: 1.75;
    color: {text_primary};
    white-space: pre-wrap;
    word-break: break-word;
}}
.latency-tag {{
    display: inline-block;
    background: {tag_bg};
    color: {tag_color};
    border: 1px solid {glass_border};
    border-radius: 20px;
    font-size: 0.65rem;
    padding: 0.2rem 0.7rem;
    letter-spacing: 0.1em;
    margin-top: 1rem;
    font-weight: 700;
}}

/* ── EXPANDERS (source chunks) ─────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {expander_bg} !important;
    border: 1px solid {divider_color} !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 0.6rem !important;
    overflow: hidden;
}}
[data-testid="stExpander"] summary {{
    font-family: 'Space Mono', 'Courier New', monospace !important;
    color: {text_secondary} !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.7rem 1rem !important;
}}
[data-testid="stExpander"] summary:hover {{
    color: {accent_cyan} !important;
}}
.chunk-text {{
    font-size: 0.78rem;
    line-height: 1.65;
    color: {text_secondary};
    font-family: 'Space Mono', 'Courier New', monospace;
    white-space: pre-wrap;
    padding: 0.4rem 0;
}}
.score-badge {{
    display: inline-block;
    background: {tag_bg};
    color: {tag_color};
    border: 1px solid {glass_border};
    border-radius: 4px;
    font-size: 0.65rem;
    padding: 0.15rem 0.55rem;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 0.6rem;
}}

/* ── DIVIDERS ───────────────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {divider_color} !important;
    margin: 1.5rem 0 !important;
}}

/* ── HIDE STREAMLIT DEFAULT DECORATION ─────────────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ── SCROLLBAR ──────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {scrollbar_track}; }}
::-webkit-scrollbar-thumb {{
    background: {scrollbar_thumb};
    border-radius: 10px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {accent_cyan}; }}

/* ── SECTION LABEL ──────────────────────────────────────────────────── */
.section-label {{
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {accent_cyan};
    font-weight: 700;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid {divider_color};
}}

/* ── ALERT OVERRIDES ────────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    background: {glass_bg} !important;
    border-left-color: {accent_cyan} !important;
    backdrop-filter: blur(10px) !important;
}}

/* ── SPINNER ────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > span {{
    color: {accent_cyan} !important;
}}

/* ── RADIO BUTTONS (theme toggle) ───────────────────────────────────── */
.stRadio > label {{
    color: {accent_cyan} !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}}

</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">⚡ EKA // CONFIG</div>', unsafe_allow_html=True)

        # API Key
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            key="api_key_input",
            help="Get your free key at console.groq.com/keys",
        )

        st.markdown("---")

        # PDF Upload
        uploaded_file = st.file_uploader(
            "Upload Knowledge Document",
            type=["pdf"],
            help="Drag and drop a PDF file to populate the knowledge base.",
            key="pdf_uploader",
        )

        st.markdown("---")

        # Chunking controls
        chunk_size = st.slider("Chunk Size (chars)", 100, 2000, 500, 50, key="chunk_size")
        overlap    = st.slider("Overlap (chars)",    0,   500,  50, 10, key="overlap")
        top_k      = st.slider("Top-K Results",      1,   10,    3,  1, key="top_k")

        st.markdown("---")

        # Theme toggle
        theme_choice = st.radio(
            "Theme Mode",
            ["Retro Dark", "Retro Bright"],
            index=0 if st.session_state.theme == "Retro Dark" else 1,
            key="theme_radio",
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.markdown("---")

        # Process button
        process_clicked = st.button("⚙ Process Document", key="process_btn")

    return api_key, uploaded_file, chunk_size, overlap, top_k, process_clicked


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_document(uploaded_file, chunk_size: int, overlap: int):
    """Extract text, chunk, and embed. Updates session state."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        raw_text = extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not raw_text.strip():
        st.error("Could not extract text from the PDF. Please try a different file.")
        return

    chunks = custom_text_splitter(raw_text, chunk_size=chunk_size, chunk_overlap=overlap)

    t0 = time.perf_counter()
    embeddings = generate_embeddings(chunks)
    embed_time = time.perf_counter() - t0

    st.session_state.chunks    = chunks
    st.session_state.embeddings = embeddings
    st.session_state.doc_stats = {
        "filename"   : uploaded_file.name,
        "char_count" : len(raw_text),
        "num_chunks" : len(chunks),
        "embed_time" : embed_time,
    }
    # Reset previous query results
    st.session_state.llm_response      = None
    st.session_state.retrieval_results = []
    st.session_state.latency           = None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    api_key, uploaded_file, chunk_size, overlap, top_k, process_clicked = render_sidebar()

    # CSS (re-injected on every render to respect theme state)
    inject_css(st.session_state.theme)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="eka-header">⚡ ENTERPRISE KNOWLEDGE ASSISTANT</div>',
        unsafe_allow_html=True,
    )

    # ── Document Processing ──────────────────────────────────────────────────
    if process_clicked:
        if uploaded_file is None:
            st.warning("Please upload a PDF document in the sidebar first.")
        else:
            with st.spinner("Processing document — chunking & vectorising..."):
                process_document(uploaded_file, chunk_size, overlap)
            st.success(f"Document processed: {len(st.session_state.chunks)} chunks ready.")

    # ── Status Card ──────────────────────────────────────────────────────────
    if st.session_state.doc_stats:
        s = st.session_state.doc_stats
        st.markdown(f"""
<div class="status-card">
  <div class="status-card-title">📄 Document Processing Status</div>
  <div class="metric-row">
    <div class="metric-item">
      <span class="metric-label">File</span>
      <span class="metric-value" style="font-size:0.9rem;">{s['filename']}</span>
    </div>
    <div class="metric-item">
      <span class="metric-label">Characters</span>
      <span class="metric-value">{s['char_count']:,}</span>
    </div>
    <div class="metric-item">
      <span class="metric-label">Chunks</span>
      <span class="metric-value">{s['num_chunks']}</span>
    </div>
    <div class="metric-item">
      <span class="metric-label">Embed Time</span>
      <span class="metric-value">{s['embed_time']:.2f}s</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Query Area ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Natural Language Query</div>', unsafe_allow_html=True)

    col_query, col_btn = st.columns([5, 1])
    with col_query:
        user_query = st.text_input(
            "Query",
            placeholder="Ask a question about your document...",
            label_visibility="collapsed",
            key="user_query",
        )
    with col_btn:
        query_clicked = st.button("⚡ Ask", key="query_btn")

    # ── Run RAG Pipeline ─────────────────────────────────────────────────────
    if query_clicked:
        if not user_query.strip():
            st.warning("Please enter a question before querying.")
        elif not st.session_state.chunks:
            st.warning("Please process a document first using the sidebar controls.")
        elif not api_key.strip():
            st.warning("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Searching knowledge base and generating grounded response..."):
                # 1. Vectorise query
                query_vec = generate_embeddings([user_query])[0]

                # 2. Cosine similarity search
                top_results = search_top_k(
                    query_vec,
                    st.session_state.embeddings,
                    top_k=top_k,
                )

                # 3. Assemble ranked chunks with scores
                retrieved = [
                    (st.session_state.chunks[idx], score)
                    for idx, score in top_results
                ]
                st.session_state.retrieval_results = retrieved

                # 4. Build grounded prompt
                chunk_texts = [text for text, _ in retrieved]
                prompt = build_grounded_prompt(user_query, chunk_texts)

                # 5. Query LLM API
                try:
                    response_text, latency = query_llm_api(
                        prompt, api_key, model_name="llama-3.3-70b-versatile"
                    )
                    st.session_state.llm_response = response_text
                    st.session_state.latency      = latency
                except Exception as err:
                    st.error(f"API Error: {str(err)}")
                    st.session_state.llm_response = None

    # ── Response Display Card ─────────────────────────────────────────────────
    if st.session_state.llm_response:
        latency_str = f"{st.session_state.latency:.2f}s" if st.session_state.latency else "—"
        st.markdown(f"""
<div class="response-card">
  <div class="response-card-header">LLM Response // Grounded Output</div>
  <div class="response-body">{st.session_state.llm_response}</div>
  <span class="latency-tag">⏱ Latency: {latency_str} &nbsp;|&nbsp; Model: llama-3.3-70b-versatile</span>
</div>
""", unsafe_allow_html=True)

    # ── Source Chunk Accordions ──────────────────────────────────────────────
    if st.session_state.retrieval_results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Retrieved Source Chunks</div>', unsafe_allow_html=True)

        for rank, (chunk_text, score) in enumerate(st.session_state.retrieval_results, start=1):
            preview = chunk_text[:80].replace("\n", " ")
            with st.expander(f"Rank {rank} — {preview}..."):
                st.markdown(
                    f'<div class="score-badge">Cosine Similarity: {score:.4f}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="chunk-text">{chunk_text}</div>',
                    unsafe_allow_html=True,
                )

    # ── Empty State ──────────────────────────────────────────────────────────
    if not st.session_state.doc_stats and not st.session_state.llm_response:
        st.markdown("""
<div style="text-align:center; padding: 4rem 0; opacity: 0.4;">
  <div style="font-size: 3rem;">📂</div>
  <div style="font-size: 0.8rem; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 1rem;">
    Upload a PDF in the sidebar to begin
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()