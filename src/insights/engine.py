"""
Insight Generation Engine — with full debug logging (self-contained)
"""
from __future__ import annotations
import json, logging, os, traceback
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# ── Logging setup (inline — no external dependency) ──────────────────
def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "analytics_agent.log"), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger

log = _make_logger("insights.engine")
# ─────────────────────────────────────────────────────────────────────

SYSTEM_INSIGHTS = (
    "You are a senior business intelligence analyst. "
    "Generate specific, data-driven insights — avoid vague generalisations. "
    "Every positive or concern must reference a concrete metric or pattern. "
    "Respond ONLY with valid JSON."
)

def generate_insights(sheets, kpis, metadata, llm_client) -> Dict:
    log.info("=== INSIGHT GENERATION STARTED ===")
    log.info(f"KPIs passed in: {len(kpis)}")

    try:
        stat_insights = _compute_statistical_insights(sheets)
        log.info("Statistical insights computed OK")
    except Exception as exc:
        log.error(f"_compute_statistical_insights FAILED:\n{traceback.format_exc()}")
        raise

    kpi_summary = _summarise_kpis(kpis)
    meta_ctx    = _metadata_context(metadata)
    log.debug(f"KPI summary for prompt:\n{json.dumps(kpi_summary, indent=2, default=str)[:800]}")

    prompt = f"""
Analyse this business dataset and return structured insights.

{meta_ctx}

Detected KPIs:
{json.dumps(kpi_summary, indent=2, default=str)}

Statistical analysis:
{json.dumps(stat_insights, indent=2, default=str)}

Return ONLY this JSON (no extra text):
{{
  "executive_summary": "2-3 sentence overview of the dataset business state",
  "health_score": <integer 0-100>,
  "health_label": "Excellent | Good | Moderate | At Risk | Critical",
  "positives": [
    {{"title": "short title", "detail": "specific finding with numbers", "metric": "value or null"}}
  ],
  "concerns": [
    {{"title": "short title", "detail": "specific issue with numbers", "urgency": "high | medium | low"}}
  ],
  "trends": [
    {{"title": "trend name", "direction": "up | down | stable", "detail": "explanation"}}
  ],
  "anomalies": [
    {{"column": "col_name", "detail": "what is anomalous and why it matters"}}
  ],
  "recommendations": [
    {{
      "action":          "Concrete action to take",
      "rationale":       "Why this action is needed (data-backed)",
      "expected_impact": "What improvement is expected",
      "priority":        "high | medium | low"
    }}
  ]
}}
"""
    log.info("Calling Groq LLM for insights (model=heavy)...")
    try:
        insights = llm_client.generate_json(prompt, system=SYSTEM_INSIGHTS, model="heavy")
        log.info(f"LLM call SUCCESS. Keys: {list(insights.keys())}")
        log.debug(f"executive_summary: {insights.get('executive_summary','')[:200]}")
    except Exception as exc:
        log.error(f"LLM call FAILED: {exc}\n{traceback.format_exc()}")
        raise RuntimeError(f"Insight generation failed — Groq API error: {exc}") from exc

    insights["statistical_details"] = stat_insights
    log.info("=== INSIGHT GENERATION DONE ===")
    return insights


def _compute_statistical_insights(sheets):
    result = {}
    for name, df in sheets.items():
        num_df = df.select_dtypes(include=np.number)
        if num_df.empty:
            log.warning(f"Sheet '{name}' has no numeric columns — skipping")
            continue
        sheet_out = {}
        sheet_out["summary_stats"] = num_df[num_df.columns[:15]].describe().round(3).to_dict()
        outliers = {}
        for col in num_df.columns:
            q1, q3 = num_df[col].quantile([0.25, 0.75]); iqr = q3 - q1
            n = int(((num_df[col] < q1-1.5*iqr) | (num_df[col] > q3+1.5*iqr)).sum())
            if n: outliers[col] = n
        sheet_out["outlier_counts"] = outliers
        if len(num_df.columns) > 1:
            corr = num_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            top = upper.stack().sort_values(ascending=False).head(5)
            sheet_out["top_correlations"] = [
                {"pair": f"{i[0]} <-> {i[1]}", "r": round(float(v),3)} for i,v in top.items()
            ]
        cat_dist = {}
        for col in df.select_dtypes(include="object").columns[:5]:
            cat_dist[col] = df[col].value_counts().head(3).to_dict()
        sheet_out["categorical_distributions"] = cat_dist
        result[name] = sheet_out
        log.debug(f"  Sheet '{name}' stats done. Outliers: {outliers}")
    return result


def _summarise_kpis(kpis):
    return [{"name": k.get("name"), "value": k.get("value"),
             "display_value": k.get("display_value"), "priority": k.get("priority"),
             "category": k.get("category"), "description": k.get("description"),
             "unit": k.get("unit")} for k in kpis]


def _metadata_context(metadata):
    return (
        f"Dataset: {metadata.get('dataset_description', 'N/A')}\n"
        f"Context: {metadata.get('business_context', metadata.get('inferred_business_context', 'N/A'))}\n"
        f"Domain : {metadata.get('domain', metadata.get('inferred_domain', 'N/A'))}\n"
    )