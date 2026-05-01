"""
Metadata Handler
Manages three metadata sources:
  1. Uploaded JSON file
  2. Manually entered Streamlit form
  3. LLM-inferred (fallback when no metadata is supplied)
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import pandas as pd


# ------------------------------------------------------------------ #
# Template shown to the user for download                              #
# ------------------------------------------------------------------ #

METADATA_TEMPLATE: Dict = {
    "dataset_description": "REQUIRED — Describe what this dataset represents and its purpose.",
    "business_context": "e.g., Monthly e-commerce sales across APAC region for FY 2024.",
    "domain": "sales | finance | marketing | operations | hr | healthcare | logistics | other",
    "time_period": "e.g., Jan 2023 – Dec 2023",
    "granularity": "e.g., daily, monthly, per-transaction, per-customer",
    "kpi_hints": ["Revenue", "Conversion Rate", "Customer Count"],
    "columns": {
        "column_name_example": {
            "description": "What this column represents",
            "unit": "e.g., USD, %, count, days",
            "type": "metric | dimension | date | identifier",
            "is_kpi": True
        }
    },
    "sheets": {
        "Sheet1": "Description of what this particular sheet contains"
    }
}

DOMAIN_OPTIONS = [
    "Auto-detect", "Sales", "Finance", "Marketing",
    "Operations", "HR", "Healthcare", "Logistics", "Other"
]


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def get_metadata_template_json() -> str:
    """Return pretty-printed JSON template string for download."""
    return json.dumps(METADATA_TEMPLATE, indent=2)


def load_metadata_file(uploaded_file) -> Dict:
    """Parse an uploaded JSON metadata file."""
    try:
        content = uploaded_file.read()
        return json.loads(content)
    except Exception as exc:
        raise ValueError(f"Could not parse metadata file: {exc}") from exc


def build_metadata_from_form(form_data: Dict) -> Dict:
    """Convert Streamlit form inputs into the standard metadata dict."""
    hints_raw = form_data.get("kpi_hints", "")
    kpi_hints = [k.strip() for k in hints_raw.split(",") if k.strip()]

    return {
        "dataset_description": form_data.get("dataset_description", "").strip(),
        "business_context":    form_data.get("business_context", "").strip(),
        "domain":              form_data.get("domain", "Auto-detect"),
        "time_period":         form_data.get("time_period", "").strip(),
        "granularity":         form_data.get("granularity", "").strip(),
        "kpi_hints":           kpi_hints,
        "columns":             {},
        "sheets":              {},
    }


def infer_metadata_with_llm(
    sheets: Dict[str, pd.DataFrame],
    llm_client,
    dataset_description: str,
) -> Dict:
    """
    Use the LLM to infer column semantics, business domain, and KPI
    candidates when no metadata file is provided.
    """
    compact: Dict = {}
    for sheet_name, df in sheets.items():
        compact[sheet_name] = {
            "columns": df.columns.tolist(),
            "dtypes":  df.dtypes.astype(str).to_dict(),
            "sample_rows": df.head(3).to_dict(orient="records"),
            "shape": list(df.shape),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        }

    prompt = f"""
You are a senior data analyst. Examine the dataset structure below and infer:
1. Column descriptions and semantic type (metric, dimension, date, identifier)
2. Likely business domain and context
3. KPI candidates (column names most useful for business monitoring)

User-supplied description: "{dataset_description}"

Dataset structure (JSON):
{json.dumps(compact, indent=2, default=str)}

Return ONLY this JSON (no extra text):
{{
  "inferred_domain": "string",
  "inferred_business_context": "string",
  "columns": {{
    "<col_name>": {{
      "description": "string",
      "type": "metric | dimension | date | identifier",
      "unit": "string or null",
      "is_kpi_candidate": true | false
    }}
  }},
  "kpi_hints": ["col or metric name", ...],
  "data_quality_notes": ["string", ...]
}}
"""
    return llm_client.generate_json(prompt, model="fast")


def validate_metadata(metadata: Dict) -> tuple[bool, str]:
    """
    Return (is_valid, error_message).
    dataset_description is the only mandatory field.
    """
    desc = metadata.get("dataset_description", "").strip()
    if not desc or desc == METADATA_TEMPLATE["dataset_description"]:
        return False, "Dataset description is required and must not be the template placeholder."
    return True, ""
