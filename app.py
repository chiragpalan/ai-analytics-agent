"""
AI Analytics Agent — Main Streamlit Application
================================================
Turns raw CSV / Excel data into an automated, insight-driven dashboard
powered by Groq (Llama 3.3 70B).

Entry point: `streamlit run app.py`
"""

import json
import os
import time

import pandas as pd
import streamlit as st

# ── Page config MUST be first Streamlit call ────────────────────────
st.set_page_config(
    page_title="AI Analytics Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/ai-analytics-agent",
        "Report a bug": "https://github.com/your-repo/ai-analytics-agent/issues",
        "About": "AI-powered analytics agent using Groq + Llama 3.3 70B",
    },
)

# ── Module imports ────────────────────────────────────────────────────
from src.llm.groq_client import GroqClient
from src.data.ingestion import load_file, get_data_summary
from src.data.preprocessing import preprocess_all, get_statistical_summary
from src.metadata.handler import (
    DOMAIN_OPTIONS,
    build_metadata_from_form,
    get_metadata_template_json,
    infer_metadata_with_llm,
    load_metadata_file,
    validate_metadata,
)
from src.kpi.detector import detect_kpis
from src.insights.engine import generate_insights
from src.dashboard.charts import (
    render_correlation_heatmap,
    render_distribution_charts,
    render_kpi_chart,
    render_kpi_priority_chart,
    render_missing_values_chart,
    render_trend_chart,
)


# ================================================================== #
# CSS                                                                  #
# ================================================================== #
def _inject_css() -> None:
    st.markdown(
        """
<style>
/* ── Global ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: #6366f1; color: white !important;
    border: none; border-radius: 8px; width: 100%;
}

/* ── KPI cards ────────────────────────────────────────── */
.kpi-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 16px;
    margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    transition: box-shadow .2s;
}
.kpi-card:hover { box-shadow: 0 4px 12px rgba(99,102,241,.15); }
.kpi-value { font-size: 28px; font-weight: 700; color: #6366f1; }
.kpi-name  { font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 4px; }
.kpi-desc  { font-size: 12px; color: #94a3b8; margin-top: 4px; }
.badge-high   { background:#fef2f2; color:#dc2626; border-radius:999px; padding:2px 8px; font-size:11px; font-weight:600; }
.badge-medium { background:#fffbeb; color:#d97706; border-radius:999px; padding:2px 8px; font-size:11px; font-weight:600; }
.badge-low    { background:#f0fdf4; color:#16a34a; border-radius:999px; padding:2px 8px; font-size:11px; font-weight:600; }

/* ── Insight cards ────────────────────────────────────── */
.insight-positive { border-left:4px solid #10b981; background:#f0fdf4; border-radius:0 8px 8px 0; padding:12px 16px; margin:6px 0; }
.insight-concern  { border-left:4px solid #ef4444; background:#fef2f2; border-radius:0 8px 8px 0; padding:12px 16px; margin:6px 0; }
.insight-rec      { border-left:4px solid #6366f1; background:#eef2ff; border-radius:0 8px 8px 0; padding:12px 16px; margin:6px 0; }
.insight-title    { font-weight:600; font-size:14px; margin-bottom:4px; }
.insight-detail   { font-size:13px; color:#475569; }

/* ── Health score ring ────────────────────────────────── */
.health-ring {
    display:flex; flex-direction:column; align-items:center;
    background:white; border-radius:16px; padding:24px;
    border:1px solid #e2e8f0; box-shadow:0 1px 4px rgba(0,0,0,.06);
}
.health-score { font-size:56px; font-weight:800; }
.health-label { font-size:16px; font-weight:600; margin-top:4px; }

/* ── Section headers ──────────────────────────────────── */
.section-header {
    font-size:20px; font-weight:700; color:#1e293b;
    border-bottom:2px solid #6366f1;
    padding-bottom:6px; margin:24px 0 16px;
}

/* ── Upload area ──────────────────────────────────────── */
.upload-hint { font-size:12px; color:#94a3b8; margin-top:4px; }
</style>
""",
        unsafe_allow_html=True,
    )


# ================================================================== #
# Session-state initialisation                                         #
# ================================================================== #
def _init_state() -> None:
    defaults = {
        "sheets":       None,
        "file_type":    None,
        "clean_sheets": None,
        "prep_reports": None,
        "metadata":     None,
        "kpis":         None,
        "insights":     None,
        "llm":          None,
        "processed":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Auto-initialise LLM from Streamlit Cloud secrets ────────────
    # If GROQ_API_KEY is configured server-side, users don't need to
    # enter a key manually — the app initialises itself on first load.
    if st.session_state.llm is None:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
        if secret_key:
            try:
                st.session_state.llm = GroqClient(secret_key)
            except Exception:
                pass  # Will surface as "Still needed: API key" in sidebar


# ================================================================== #
# Sidebar                                                              #
# ================================================================== #
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🤖 AI Analytics Agent")
        st.markdown("---")

        # ── API Key ──────────────────────────────────────────────────
        st.markdown("### 🔑 Groq API Key")

        # If the key is already loaded from st.secrets, show a success
        # badge and skip the manual input field entirely.
        if st.session_state.llm is not None and not st.session_state.get("_manual_key"):
            st.success("✅ API key pre-configured")
            st.caption("This app is running with a server-side API key — no key needed.")
        else:
            api_key = st.text_input(
                "Enter your Groq API key",
                type="password",
                placeholder="gsk_...",
                help="Get a free key at console.groq.com",
            )
            if api_key:
                try:
                    st.session_state.llm = GroqClient(api_key)
                    st.session_state["_manual_key"] = True
                    st.success("✅ API key accepted")
                except Exception as e:
                    st.error(f"Invalid key: {e}")
            st.markdown("[Get free Groq API key →](https://console.groq.com)", unsafe_allow_html=False)
        st.markdown("---")

        # ── File upload ───────────────────────────────────────────────
        st.markdown("### 📂 Upload Dataset")
        data_file = st.file_uploader(
            "CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Max 200 MB. Excel multi-sheet supported.",
        )
        if data_file:
            if data_file.name != st.session_state.get("_uploaded_filename"):
                st.session_state["_uploaded_filename"] = data_file.name
                _handle_data_upload(data_file)

        st.markdown("---")

        # ── Metadata ──────────────────────────────────────────────────
        st.markdown("### 🗂️ Metadata")
        meta_mode = st.radio(
            "How would you like to provide metadata?",
            ["📝 Fill in form", "📁 Upload JSON file"],
            horizontal=True,
        )
        if st.session_state.sheets is not None:
            if meta_mode == "📁 Upload JSON file":
                _metadata_file_upload()
            else:
                _metadata_form()
        else:
            st.info("Upload a dataset first.")

        # ── Template download ─────────────────────────────────────────
        st.download_button(
            "⬇ Download metadata template",
            data=get_metadata_template_json(),
            file_name="metadata_template.json",
            mime="application/json",
        )
        st.markdown("---")

        # ── Run button ────────────────────────────────────────────────
        ready = (
            st.session_state.sheets is not None
            and st.session_state.llm is not None
            and st.session_state.metadata is not None
        )
        if st.button("🚀 Run Analysis", disabled=not ready, use_container_width=True):
            _run_analysis()

        if not ready:
            missing = []
            if st.session_state.sheets is None:   missing.append("dataset")
            if st.session_state.llm is None:       missing.append("API key")
            if st.session_state.metadata is None:  missing.append("metadata / description")
            st.caption(f"Still needed: {', '.join(missing)}")


# ── Data upload handler ───────────────────────────────────────────────
def _handle_data_upload(data_file) -> None:
    with st.spinner("Loading file…"):
        try:
            sheets, ftype = load_file(data_file)
            st.session_state.sheets    = sheets
            st.session_state.file_type = ftype
            st.session_state.processed = False   # reset on new upload
            total_rows = sum(len(df) for df in sheets.values())
            st.success(
                f"✅ Loaded {'sheet: ' if ftype == 'csv' else str(len(sheets)) + ' sheet(s): '}"
                f"**{', '.join(sheets.keys())}**  "
                f"({total_rows:,} rows)"
            )
        except Exception as exc:
            st.error(f"Could not load file: {exc}")


# ── Metadata: file upload ─────────────────────────────────────────────
def _metadata_file_upload() -> None:
    meta_file = st.file_uploader(
        "Upload metadata JSON", type=["json"], key="meta_file_uploader"
    )
    if meta_file:
        try:
            metadata = load_metadata_file(meta_file)
            valid, err = validate_metadata(metadata)
            if valid:
                st.session_state.metadata = metadata
                st.success("✅ Metadata loaded")
            else:
                st.error(err)
        except Exception as exc:
            st.error(f"Metadata parse error: {exc}")


# ── Metadata: manual form ─────────────────────────────────────────────
def _metadata_form() -> None:
    with st.form("metadata_form"):
        desc = st.text_area(
            "Dataset Description ⭐ (required)",
            placeholder="Describe what this dataset represents, its source, and purpose.",
            height=80,
        )
        context = st.text_area(
            "Business Context",
            placeholder="e.g. Monthly APAC e-commerce sales for FY 2024.",
            height=60,
        )
        domain  = st.selectbox("Domain", DOMAIN_OPTIONS)
        period  = st.text_input("Time Period", placeholder="e.g. Jan 2023 – Dec 2023")
        hints   = st.text_input(
            "KPI Hints (comma-separated, optional)",
            placeholder="Revenue, Conversion Rate, Churn",
        )
        submitted = st.form_submit_button("Save Metadata")

    if submitted:
        if not desc.strip():
            st.error("Dataset description is required.")
        else:
            meta = build_metadata_from_form(
                {
                    "dataset_description": desc,
                    "business_context":    context,
                    "domain":              domain,
                    "time_period":         period,
                    "kpi_hints":           hints,
                }
            )
            st.session_state.metadata = meta
            st.success("✅ Metadata saved")


# ================================================================== #
# Analysis pipeline                                                    #
# ================================================================== #
def _run_analysis() -> None:
    llm      = st.session_state.llm
    sheets   = st.session_state.sheets
    metadata = st.session_state.metadata

    progress = st.sidebar.progress(0, "Starting…")

    try:
        # 1 ── Preprocessing
        progress.progress(10, "🔧 Preprocessing data…")
        clean, reports = preprocess_all(sheets)
        st.session_state.clean_sheets = clean
        st.session_state.prep_reports = reports

        # 2 ── Metadata inference (if columns not described)
        progress.progress(25, "🧠 Inferring metadata…")
        if not metadata.get("columns"):
            inferred = infer_metadata_with_llm(
                clean, llm, metadata["dataset_description"]
            )
            metadata.update({k: v for k, v in inferred.items() if v})
            st.session_state.metadata = metadata

        # 3 ── KPI detection
        progress.progress(50, "📊 Detecting KPIs…")
        kpis = detect_kpis(clean, metadata, llm, reports)
        st.session_state.kpis = kpis

        # 4 ── Insight generation
        progress.progress(75, "💡 Generating insights…")
        insights = generate_insights(clean, kpis, metadata, llm)
        st.session_state.insights = insights

        progress.progress(100, "✅ Done!")
        time.sleep(0.4)
        progress.empty()
        st.session_state.processed = True
        st.rerun()

    except Exception as exc:
        progress.empty()
        st.sidebar.error(f"Analysis failed: {exc}")


# ================================================================== #
# Dashboard tabs                                                       #
# ================================================================== #
def _render_dashboard() -> None:
    sheets   = st.session_state.clean_sheets
    metadata = st.session_state.metadata
    kpis     = st.session_state.kpis     or []
    insights = st.session_state.insights or {}
    reports  = st.session_state.prep_reports or {}

    # Header
    domain = metadata.get("domain", metadata.get("inferred_domain", "Analytics"))
    st.markdown(
        f"<h1 style='color:#1e293b;margin-bottom:0'>🤖 {domain} Dashboard</h1>"
        f"<p style='color:#64748b'>{metadata.get('dataset_description','')}</p>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_kpis, tab_insights, tab_data, tab_raw = st.tabs([
        "📊 Overview",
        "🎯 KPIs",
        "💡 Insights",
        "🔍 Data Explorer",
        "⚙️ Raw Details",
    ])

    with tab_overview:
        _tab_overview(sheets, insights, kpis)

    with tab_kpis:
        _tab_kpis(sheets, kpis, metadata)

    with tab_insights:
        _tab_insights(insights)

    with tab_data:
        _tab_data(sheets)

    with tab_raw:
        _tab_raw(metadata, reports, kpis, insights)


# ── TAB: Overview ─────────────────────────────────────────────────────
def _tab_overview(sheets, insights, kpis) -> None:
    # Health score + executive summary
    score = insights.get("health_score", 0)
    label = insights.get("health_label", "Unknown")
    colour = (
        "#10b981" if score >= 75 else
        "#f59e0b" if score >= 50 else
        "#ef4444"
    )

    col_score, col_summary = st.columns([1, 3])
    with col_score:
        st.markdown(
            f"""
<div class="health-ring">
  <div class="health-score" style="color:{colour}">{score}</div>
  <div class="health-label" style="color:{colour}">{label}</div>
  <div style="font-size:12px;color:#94a3b8;margin-top:4px">Health Score</div>
</div>""",
            unsafe_allow_html=True,
        )

    with col_summary:
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:15px;color:#334155;line-height:1.7'>"
            f"{insights.get('executive_summary','—')}</p>",
            unsafe_allow_html=True,
        )

        # Quick KPI metrics row
        if kpis:
            cols = st.columns(min(len(kpis), 4))
            for i, kpi in enumerate(kpis[:4]):
                with cols[i]:
                    st.metric(
                        label=kpi["name"],
                        value=kpi.get("display_value", "N/A"),
                    )

    st.divider()

    # KPI priority chart
    if kpis:
        fig = render_kpi_priority_chart(kpis)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Data distributions
    st.markdown('<div class="section-header">Column Distributions</div>', unsafe_allow_html=True)
    for sheet_name, df in sheets.items():
        if len(sheets) > 1:
            st.caption(f"Sheet: {sheet_name}")
        figs = render_distribution_charts(df, max_cols=6)
        if figs:
            cols = st.columns(min(len(figs), 3))
            for i, fig in enumerate(figs):
                with cols[i % 3]:
                    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    for sheet_name, df in sheets.items():
        fig = render_correlation_heatmap(df)
        if fig:
            st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
            if len(sheets) > 1:
                st.caption(f"Sheet: {sheet_name}")
            st.plotly_chart(fig, use_container_width=True)

    # Missing values
    for sheet_name, df in sheets.items():
        fig = render_missing_values_chart(df)
        if fig:
            st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
            if len(sheets) > 1:
                st.caption(f"Sheet: {sheet_name}")
            st.plotly_chart(fig, use_container_width=True)


# ── TAB: KPIs ────────────────────────────────────────────────────────
def _tab_kpis(sheets, kpis, metadata) -> None:
    st.markdown('<div class="section-header">🎯 Detected KPIs</div>', unsafe_allow_html=True)

    if not kpis:
        st.info("No KPIs detected. Run the analysis first.")
        return

    # Optional user filter
    priorities = st.multiselect(
        "Filter by priority", ["high", "medium", "low"],
        default=["high", "medium", "low"]
    )
    filtered_kpis = [k for k in kpis if k.get("priority", "low") in priorities]

    badge_cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}

    for kpi in filtered_kpis:
        with st.expander(f"**{kpi['name']}** — {kpi.get('display_value','N/A')}", expanded=True):
            col_info, col_chart = st.columns([1, 2])

            with col_info:
                p = kpi.get("priority", "low")
                st.markdown(
                    f"<span class='{badge_cls.get(p,'')}'>Priority: {p.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Category:** {kpi.get('category','—')}")
                st.markdown(f"**Aggregation:** `{kpi.get('aggregation','—')}`")
                st.markdown(f"**Columns:** `{', '.join(kpi.get('columns',[]))}`")
                if kpi.get("group_by"):
                    st.markdown(f"**Grouped by:** `{kpi['group_by']}`")
                if kpi.get("time_column"):
                    st.markdown(f"**Time column:** `{kpi['time_column']}`")
                st.markdown(f"**Sheet:** `{kpi.get('sheet','—')}`")
                st.caption(kpi.get("description", ""))

            with col_chart:
                df = sheets.get(kpi.get("sheet", ""), list(sheets.values())[0])
                fig = render_kpi_chart(kpi, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Chart could not be rendered for this KPI.")

                # Trend chart (if available)
                td = kpi.get("trend_data")
                if td:
                    trend_fig = render_trend_chart(td, kpi["name"])
                    if trend_fig:
                        st.plotly_chart(trend_fig, use_container_width=True)


# ── TAB: Insights ─────────────────────────────────────────────────────
def _tab_insights(insights) -> None:
    if not insights:
        st.info("Run analysis to generate insights.")
        return

    # ── Positives ────────────────────────────────────────────────────
    positives = insights.get("positives", [])
    if positives:
        st.markdown('<div class="section-header">✅ What\'s Going Well</div>', unsafe_allow_html=True)
        for p in positives:
            metric_str = f" &nbsp;·&nbsp; <code>{p.get('metric','')}</code>" if p.get("metric") else ""
            st.markdown(
                f"""<div class="insight-positive">
  <div class="insight-title">🟢 {p['title']}{metric_str}</div>
  <div class="insight-detail">{p['detail']}</div>
</div>""",
                unsafe_allow_html=True,
            )

    # ── Concerns ─────────────────────────────────────────────────────
    concerns = insights.get("concerns", [])
    if concerns:
        st.markdown('<div class="section-header">⚠️ What Needs Attention</div>', unsafe_allow_html=True)
        urgency_emoji = {"high": "🔴", "medium": "🟡", "low": "🟠"}
        for c in concerns:
            urg = c.get("urgency", "low")
            st.markdown(
                f"""<div class="insight-concern">
  <div class="insight-title">{urgency_emoji.get(urg,'🟡')} {c['title']} <span class="badge-{urg}">{urg.upper()}</span></div>
  <div class="insight-detail">{c['detail']}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    col_trends, col_anomalies = st.columns(2)

    # ── Trends ───────────────────────────────────────────────────────
    with col_trends:
        trends = insights.get("trends", [])
        if trends:
            st.markdown("#### 📈 Key Trends")
            dir_emoji = {"up": "📈", "down": "📉", "stable": "➡️"}
            for t in trends:
                st.markdown(
                    f"**{dir_emoji.get(t.get('direction','stable'), '')} {t['title']}**  \n"
                    f"<span style='color:#64748b;font-size:13px'>{t['detail']}</span>",
                    unsafe_allow_html=True,
                )

    # ── Anomalies ─────────────────────────────────────────────────────
    with col_anomalies:
        anomalies = insights.get("anomalies", [])
        if anomalies:
            st.markdown("#### 🔍 Anomalies Detected")
            for a in anomalies:
                st.markdown(
                    f"**`{a.get('column','—')}`**  \n"
                    f"<span style='color:#64748b;font-size:13px'>{a['detail']}</span>",
                    unsafe_allow_html=True,
                )

    # ── Recommendations ───────────────────────────────────────────────
    recs = insights.get("recommendations", [])
    if recs:
        st.divider()
        st.markdown('<div class="section-header">🚀 Recommendations</div>', unsafe_allow_html=True)
        for i, r in enumerate(recs, 1):
            prio = r.get("priority", "medium")
            with st.container():
                st.markdown(
                    f"""<div class="insight-rec">
  <div class="insight-title">{i}. {r['action']} <span class="badge-{prio}">{prio.upper()}</span></div>
  <div class="insight-detail"><b>Why:</b> {r['rationale']}</div>
  <div class="insight-detail"><b>Expected impact:</b> {r['expected_impact']}</div>
</div>""",
                    unsafe_allow_html=True,
                )


# ── TAB: Data Explorer ────────────────────────────────────────────────
def _tab_data(sheets) -> None:
    st.markdown('<div class="section-header">🔍 Data Explorer</div>', unsafe_allow_html=True)

    sheet_names = list(sheets.keys())
    selected = st.selectbox("Select sheet", sheet_names) if len(sheet_names) > 1 else sheet_names[0]
    df = sheets[selected]

    col_rows, col_cols, col_missing = st.columns(3)
    col_rows.metric("Rows",    f"{len(df):,}")
    col_cols.metric("Columns", len(df.columns))
    col_missing.metric("Missing cells", f"{df.isnull().sum().sum():,}")

    # Column filter
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect("Filter columns", all_cols, default=all_cols[:10])
    if selected_cols:
        df = df[selected_cols]

    # Row filter
    n_rows = st.slider("Rows to display", 10, min(500, len(df)), 50)
    st.dataframe(df.head(n_rows), use_container_width=True, height=400)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download cleaned data (CSV)",
        data=csv,
        file_name=f"cleaned_{selected}.csv",
        mime="text/csv",
    )

    # Stats
    with st.expander("📐 Statistical Summary"):
        st.dataframe(df.describe(include="all").round(3), use_container_width=True)


# ── TAB: Raw Details ──────────────────────────────────────────────────
def _tab_raw(metadata, reports, kpis, insights) -> None:
    st.markdown('<div class="section-header">⚙️ Raw Details</div>', unsafe_allow_html=True)
    with st.expander("Metadata used"):
        st.json(metadata)
    with st.expander("Preprocessing reports"):
        st.json(reports)
    with st.expander("Detected KPIs (raw JSON)"):
        st.json(kpis)
    with st.expander("Insights (raw JSON)"):
        st.json(insights)


# ================================================================== #
# Landing page (no data uploaded yet)                                  #
# ================================================================== #
def _render_landing() -> None:
    st.markdown(
        """
<div style="text-align:center;padding:60px 20px;">
  <div style="font-size:64px">🤖</div>
  <h1 style="color:#1e293b;margin:12px 0 8px">AI Analytics Agent</h1>
  <p style="color:#64748b;font-size:18px;max-width:600px;margin:0 auto 32px">
    Upload any CSV or Excel file and get an automated, insight-driven dashboard
    powered by <b>Groq + Llama 3.3 70B</b> — no predefined KPIs, no coding needed.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """### 📊 Auto KPI Detection
Automatically surfaces the most relevant KPIs for your domain — sales, finance, HR, and more."""
        )
    with col2:
        st.markdown(
            """### 💡 AI-Powered Insights
Identifies trends, anomalies, correlations, and generates actionable recommendations."""
        )
    with col3:
        st.markdown(
            """### 🗂️ Flexible Metadata
Upload a JSON metadata file or fill in a quick form. Dataset description is the only requirement."""
        )

    st.info(
        "👈 **Get started:** Enter your Groq API key and upload a dataset in the sidebar."
    )


# ================================================================== #
# Entry point                                                          #
# ================================================================== #
def main() -> None:
    _inject_css()
    _init_state()
    _render_sidebar()

    if st.session_state.processed:
        _render_dashboard()
    else:
        _render_landing()


if __name__ == "__main__":
    main()