"""
KPI Detector — with full debug logging (self-contained)
"""
from __future__ import annotations
import json, logging, os, traceback
from typing import Any, Dict, List, Optional
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

log = _make_logger("kpi.detector")
# ─────────────────────────────────────────────────────────────────────

SYSTEM_KPI = (
    "You are a senior business analyst specialising in KPI design. "
    "Identify the most actionable, measurable KPIs from the data provided. "
    "Respond ONLY with valid JSON — no prose, no markdown."
)

def detect_kpis(sheets, metadata, llm_client, preprocessing_reports) -> List[Dict]:
    log.info("=== KPI DETECTION STARTED ===")
    log.info(f"Sheets: {list(sheets.keys())}")
    for name, df in sheets.items():
        log.info(f"  Sheet '{name}': {df.shape} | cols={list(df.columns)}")

    try:
        data_ctx = _build_data_context(sheets)
        log.debug(f"Data context built ({len(data_ctx)} chars)")
    except Exception as exc:
        log.error(f"_build_data_context FAILED:\n{traceback.format_exc()}")
        raise

    try:
        meta_ctx = _build_metadata_context(metadata)
        log.debug(f"Metadata context:\n{meta_ctx}")
    except Exception as exc:
        log.error(f"_build_metadata_context FAILED:\n{traceback.format_exc()}")
        raise

    prompt = f"""
Based on the dataset information below, identify the 5-8 most important KPIs
for business monitoring and decision-making.

{meta_ctx}

Dataset statistics:
{data_ctx}

For each KPI return a JSON object inside the "kpis" array:
{{
  "kpis": [
    {{
      "name":            "Human-readable KPI name",
      "description":     "Why this KPI matters",
      "columns":         ["primary_col"],
      "sheet":           "sheet_name",
      "aggregation":     "sum | mean | count | max | min | ratio | growth_rate | nunique",
      "numerator_col":   "col (only if aggregation=ratio)",
      "denominator_col": "col (only if aggregation=ratio)",
      "chart_type":      "bar | line | pie | scatter | gauge | table | histogram",
      "time_column":     "date_col or null",
      "group_by":        "category_col or null",
      "priority":        "high | medium | low",
      "category":        "revenue | cost | efficiency | quality | growth | satisfaction | other",
      "unit":            "USD | % | count | days | etc."
    }}
  ]
}}
"""
    log.info("Calling Groq LLM for KPI detection (model=heavy)...")
    try:
        result = llm_client.generate_json(prompt, system=SYSTEM_KPI, model="heavy")
        log.info(f"LLM call SUCCESS. Result keys: {list(result.keys())}")
        log.debug(f"Full LLM result (first 1000 chars): {json.dumps(result, default=str)[:1000]}")
    except Exception as exc:
        log.error(f"LLM call FAILED: {exc}\n{traceback.format_exc()}")
        raise RuntimeError(f"KPI detection failed — Groq API error: {exc}") from exc

    raw_kpis: List[Dict] = result.get("kpis", [])
    log.info(f"Raw KPIs from LLM: {len(raw_kpis)}")
    for i, k in enumerate(raw_kpis):
        log.debug(f"  [{i}] name={k.get('name')} | cols={k.get('columns')} | sheet={k.get('sheet')}")

    validated = []
    for i, kpi in enumerate(raw_kpis):
        log.debug(f"Validating KPI [{i}]: {kpi.get('name')}")
        kpi = _validate_and_fix(kpi, sheets)
        if kpi is None:
            log.warning(f"  KPI [{i}] DROPPED — column mismatch")
            continue
        try:
            kpi["value"]         = _compute_value(kpi, sheets)
            kpi["trend_data"]    = _compute_trend(kpi, sheets)
            kpi["display_value"] = _format_value(kpi)
            log.debug(f"  KPI [{i}] OK: value={kpi['value']} | display={kpi['display_value']}")
            validated.append(kpi)
        except Exception as exc:
            log.error(f"  KPI [{i}] compute FAILED: {exc}\n{traceback.format_exc()}")

    log.info(f"=== KPI DETECTION DONE — {len(validated)}/{len(raw_kpis)} valid ===")
    return validated


def _build_data_context(sheets):
    parts = []
    for name, df in sheets.items():
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        dt_cols  = df.select_dtypes(include="datetime").columns.tolist()
        stats = {}
        for col in num_cols[:12]:
            s = df[col].dropna()
            stats[col] = {"sum": round(float(s.sum()),2), "mean": round(float(s.mean()),2),
                          "min": round(float(s.min()),2), "max": round(float(s.max()),2),
                          "nunique": int(s.nunique())}
        parts.append(
            f"Sheet='{name}' | rows={len(df)} cols={len(df.columns)}\n"
            f"  Numeric : {num_cols}\n  Categorical: {cat_cols}\n  Datetime: {dt_cols}\n"
            f"  Stats   : {json.dumps(stats, default=str)}\n"
            f"  Sample  : {df.head(2).to_dict(orient='records')}\n"
        )
    return "\n".join(parts)


def _build_metadata_context(metadata):
    return (
        f"Dataset description : {metadata.get('dataset_description', 'N/A')}\n"
        f"Business context    : {metadata.get('business_context', metadata.get('inferred_business_context', 'N/A'))}\n"
        f"Domain              : {metadata.get('domain', metadata.get('inferred_domain', 'N/A'))}\n"
        f"KPI hints from user : {metadata.get('kpi_hints', [])}\n"
        f"Time period         : {metadata.get('time_period', 'N/A')}\n"
    )


def _validate_and_fix(kpi, sheets):
    sheet = kpi.get("sheet", "")
    if sheet not in sheets:
        sheet = list(sheets.keys())[0]
        kpi["sheet"] = sheet
        log.debug(f"  Sheet not found — using '{sheet}'")
    df = sheets[sheet]
    cols = [c for c in kpi.get("columns", []) if c in df.columns]
    if not cols:
        log.warning(f"  Columns {kpi.get('columns')} not found in {list(df.columns)}")
        return None
    kpi["columns"] = cols
    for field in ("time_column", "group_by", "numerator_col", "denominator_col"):
        val = kpi.get(field)
        if val and val not in df.columns:
            log.debug(f"  '{field}'='{val}' not in df — clearing")
            kpi[field] = None
    return kpi


def _compute_value(kpi, sheets):
    try:
        df = sheets[kpi["sheet"]]; col = kpi["columns"][0]; agg = kpi.get("aggregation","sum")
        if not pd.api.types.is_numeric_dtype(df[col]):
            return float(df[col].nunique()) if agg in ("count","nunique") else None
        s = df[col].dropna()
        mapping = {"sum": s.sum(), "mean": s.mean(), "count": s.count(), "max": s.max(),
                   "min": s.min(), "nunique": s.nunique(),
                   "growth_rate": _growth_rate(s), "ratio": _ratio(kpi, df)}
        val = mapping.get(agg, s.sum())
        return round(float(val), 4) if val is not None else None
    except Exception as exc:
        log.error(f"_compute_value: {exc}"); return None

def _growth_rate(series):
    if len(series) < 2 or series.iloc[0] == 0: return None
    return (series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0]) * 100

def _ratio(kpi, df):
    num = kpi.get("numerator_col", kpi["columns"][0]); den = kpi.get("denominator_col")
    if not den or den not in df.columns: return None
    d = df[den].sum(); return df[num].sum() / d if d != 0 else None

def _compute_trend(kpi, sheets):
    time_col = kpi.get("time_column")
    if not time_col: return None
    try:
        df = sheets[kpi["sheet"]]; col = kpi["columns"][0]
        if time_col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]): return None
        agg_fn = "sum" if kpi.get("aggregation","sum") in ("sum","ratio","growth_rate") else kpi.get("aggregation","sum")
        grouped = (df.groupby(df[time_col].dt.to_period("M") if hasattr(df[time_col],"dt") else time_col)[col]
                   .agg(agg_fn).reset_index())
        return [{"label": str(row[time_col]), "value": round(float(row[col]),2)} for _,row in grouped.iterrows()]
    except Exception as exc:
        log.warning(f"_compute_trend: {exc}"); return None

def _format_value(kpi):
    val = kpi.get("value"); unit = kpi.get("unit",""); agg = kpi.get("aggregation","sum")
    if val is None: return "N/A"
    if agg == "growth_rate" or "%" in unit: return f"{val:+.2f} %"
    if unit and unit.upper() in ("USD","$","EUR","INR","GBP"):
        if abs(val) >= 1_000_000: return f"{unit} {val/1_000_000:.2f} M"
        if abs(val) >= 1_000: return f"{unit} {val/1_000:.1f} K"
        return f"{unit} {val:,.2f}"
    if agg == "ratio": return f"{val:.2%}"
    if abs(val) >= 1_000_000: return f"{val/1_000_000:.2f} M"
    if abs(val) >= 1_000: return f"{val/1_000:.1f} K"
    return f"{val:,.2f}"