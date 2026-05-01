#!/usr/bin/env bash
# =============================================================
#  AI Analytics Agent — Project Setup Script (Mac / Linux)
#  Run: bash setup_project.sh
# =============================================================

set -e   # exit on any error

PROJECT="ai-analytics-agent"

echo ""
echo "==========================================="
echo "  AI Analytics Agent — Project Setup"
echo "==========================================="
echo ""

# ── 1. Create folder structure ─────────────────────────────────────
echo "📁 Creating folder structure..."

mkdir -p "$PROJECT"/{.streamlit,assets}
mkdir -p "$PROJECT"/src/{llm,data,metadata,kpi,insights,dashboard}

# Touch all __init__.py files
touch "$PROJECT"/src/__init__.py
touch "$PROJECT"/src/llm/__init__.py
touch "$PROJECT"/src/data/__init__.py
touch "$PROJECT"/src/metadata/__init__.py
touch "$PROJECT"/src/kpi/__init__.py
touch "$PROJECT"/src/insights/__init__.py
touch "$PROJECT"/src/dashboard/__init__.py

echo "✅ Folder structure created"

# ── 2. Confirm structure ───────────────────────────────────────────
echo ""
echo "📂 Project structure:"
find "$PROJECT" -type f -o -type d | sort | sed "s|$PROJECT/||" | sed 's|[^/]*/|  |g'

# ── 3. Instructions ────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "  Next steps:"
echo "==========================================="
echo ""
echo "1. Copy your source files into the folders:"
echo "   app.py                 → $PROJECT/"
echo "   requirements.txt       → $PROJECT/"
echo "   .gitignore             → $PROJECT/"
echo "   config.toml            → $PROJECT/.streamlit/"
echo "   sample_metadata.json   → $PROJECT/assets/"
echo "   groq_client.py         → $PROJECT/src/llm/"
echo "   ingestion.py           → $PROJECT/src/data/"
echo "   preprocessing.py       → $PROJECT/src/data/"
echo "   handler.py             → $PROJECT/src/metadata/"
echo "   detector.py            → $PROJECT/src/kpi/"
echo "   engine.py              → $PROJECT/src/insights/"
echo "   charts.py              → $PROJECT/src/dashboard/"
echo ""
echo "2. Create virtual environment:"
echo "   cd $PROJECT"
echo "   python -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Run the app:"
echo "   streamlit run app.py"
echo ""
echo "==========================================="
