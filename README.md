# 🤖 AI Analytics Agent

> Transform any CSV or Excel file into an automated, insight-driven business dashboard — powered by **Groq (Llama 3.3 70B)**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange.svg)](https://groq.com)

---

## ✨ Features

| Feature | Details |
|---|---|
| 📂 **Multi-format upload** | CSV, XLSX, XLS — including multi-sheet Excel |
| 🗂️ **Flexible metadata** | Upload JSON file **or** fill in a quick form |
| 🤖 **Auto KPI detection** | No predefined rules — adapts to any business domain |
| 💡 **AI insights** | Trends, anomalies, recommendations, health score |
| 📊 **Interactive charts** | Plotly — bar, line, pie, scatter, histogram, heatmap |
| 🔍 **Data explorer** | Filter, inspect, and download cleaned data |
| 🆓 **Free to run** | Groq free tier + Streamlit Cloud free tier |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│  Sidebar                    Main Dashboard               │
│  ├─ API Key input           ├─ Tab: Overview             │
│  ├─ Data file upload        ├─ Tab: KPIs                 │
│  ├─ Metadata (file/form)    ├─ Tab: Insights             │
│  └─ Run Analysis button     ├─ Tab: Data Explorer        │
│                             └─ Tab: Raw Details          │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │         Analysis Pipeline       │
         │  1. Data Ingestion              │
         │  2. Preprocessing & Cleaning    │
         │  3. Metadata Inference (LLM)    │
         │  4. KPI Detection (LLM + Stats) │
         │  5. Insight Generation (LLM)    │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │    Groq API (Llama 3.3 70B)    │
         │    llama-3.3-70b-versatile      │
         │    llama-3.1-8b-instant (fast)  │
         └────────────────────────────────┘
```

### Folder Structure

```
ai-analytics-agent/
├── app.py                      # Main Streamlit application
├── requirements.txt
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml             # Theme + server config
├── assets/
│   └── sample_metadata.json    # Downloadable metadata template
└── src/
    ├── llm/
    │   └── groq_client.py      # Groq API wrapper
    ├── data/
    │   ├── ingestion.py        # CSV / Excel loader
    │   └── preprocessing.py   # Cleaning & normalisation
    ├── metadata/
    │   └── handler.py          # Parse / form / LLM-infer metadata
    ├── kpi/
    │   └── detector.py         # Auto KPI detection
    ├── insights/
    │   └── engine.py           # Insight & recommendation generation
    └── dashboard/
        └── charts.py           # Plotly chart builders
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-analytics-agent.git
cd ai-analytics-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get a free Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up (free) → API Keys → Create Key
- Copy the key (starts with `gsk_…`)

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy to Streamlit Cloud (Free)

Follow these steps exactly — estimated time: **10 minutes**.

### Step 1 — Push to GitHub

```bash
# Inside your project folder
git init
git add .
git commit -m "Initial commit"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/ai-analytics-agent.git
git branch -M main
git push -u origin main
```

> ⚠️ Make sure `.streamlit/secrets.toml` and `.env` are in `.gitignore` — **never push API keys**.

### Step 2 — Create a Streamlit Cloud account
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with your GitHub account (free tier available)

### Step 3 — Deploy the app
1. Click **"New app"**
2. **Repository:** `your-username/ai-analytics-agent`
3. **Branch:** `main`
4. **Main file path:** `app.py`
5. Click **"Deploy!"**

### Step 4 — Add Groq API key as a secret (optional but recommended)

If you want a shared deployment where users don't need to enter their own key:

1. In your Streamlit Cloud app settings → **Secrets**
2. Add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. In `app.py`, update the API key input to use:
   ```python
   import os
   api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
   ```

### Step 5 — Share your app
Your app will be live at:
```
https://your-username-ai-analytics-agent-app-XXXX.streamlit.app
```

---

## 🗂️ Metadata Format

Download the template from the app sidebar, fill it in, and upload it:

```json
{
  "dataset_description": "Required — what is this dataset?",
  "business_context": "Business domain and use case",
  "domain": "sales | finance | marketing | operations | hr | other",
  "time_period": "Jan 2024 – Dec 2024",
  "kpi_hints": ["Revenue", "Conversion Rate"],
  "columns": {
    "revenue": {
      "description": "Total order revenue",
      "unit": "USD",
      "type": "metric",
      "is_kpi": true
    }
  }
}
```

**Only `dataset_description` is mandatory.** Everything else is optional — the AI will infer the rest.

---

## 🔬 How KPI Detection Works

1. **Data profiling** — numeric stats, column types, cardinality, and sample values are extracted.
2. **Metadata enrichment** — user-provided or LLM-inferred context is added.
3. **LLM reasoning** — Llama 3.3 70B proposes 5–8 KPIs with aggregation method and chart type.
4. **Validation** — each proposed KPI is checked against real column names.
5. **Computation** — aggregate values are computed (sum, mean, ratio, growth rate…).

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit 1.32+ |
| **LLM** | Groq API — Llama 3.3 70B Versatile |
| **Data** | Pandas, NumPy |
| **Charts** | Plotly Express + Graph Objects |
| **Stats** | SciPy, Statsmodels |
| **Excel** | openpyxl, xlrd |

---

## 🛡️ Limitations & Notes

- **Data privacy:** Your data is sent to Groq's servers for LLM calls. Do not upload sensitive PII or confidential data in a shared deployment.
- **Token limits:** Very wide datasets (100+ columns) may be truncated in the LLM context. The agent automatically caps column stats to avoid this.
- **LLM accuracy:** KPI and insight quality depends on dataset clarity. Good metadata always improves results.
- **Groq free tier:** 14,400 requests/day and 6,000 tokens/minute. Sufficient for typical usage.

---

## 📄 License

MIT — free to use, modify, and deploy.
