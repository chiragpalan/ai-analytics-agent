"""
Data Preprocessing
Cleans, normalises, and enriches every DataFrame before analysis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def preprocess_all(
    sheets: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """Preprocess every sheet in the uploaded dataset."""
    cleaned: Dict[str, pd.DataFrame] = {}
    reports: Dict = {}
    for name, df in sheets.items():
        cleaned[name], reports[name] = preprocess_dataframe(df)
    return cleaned, reports


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline for a single DataFrame.

    Steps
    -----
    1. Strip whitespace from string cells
    2. Auto-detect and convert date columns
    3. Drop columns with >80 % missing values
    4. Impute remaining missing values (median for numeric, mode for categorical)
    5. Remove exact duplicate rows
    6. Classify column types

    Returns
    -------
    cleaned_df : pd.DataFrame
    report     : dict  (what was done)
    """
    report: Dict = {}
    df = df.copy()

    # 1 ── Whitespace cleanup
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    # 2 ── Date detection
    date_cols: List[str] = []
    for col in df.select_dtypes(include="object").columns:
        # Primary attempt: let pandas infer dates (no deprecated args)
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().mean() >= 0.70:          # ≥70 % parseable → it's a date
            df[col] = converted
            date_cols.append(col)
            continue

        # Fallback: try a handful of common explicit formats to improve parsing
        common_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]
        for fmt in common_formats:
            converted = pd.to_datetime(df[col], format=fmt, errors="coerce")
            if converted.notna().mean() >= 0.70:
                df[col] = converted
                date_cols.append(col)
                break
    report["date_columns_detected"] = date_cols

    # 3 ── Drop high-missing columns
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.80].index.tolist()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    report["dropped_high_missing_columns"] = drop_cols

    # 4 ── Impute
    missing_before = int(df.isnull().sum().sum())
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col].fillna(method="ffill", inplace=True)
        else:
            mode = df[col].mode()
            df[col].fillna(mode[0] if not mode.empty else "Unknown", inplace=True)
    report["missing_values_imputed"] = missing_before - int(df.isnull().sum().sum())

    # 5 ── Deduplication
    dupe_count = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    report["duplicate_rows_removed"] = dupe_count

    # 6 ── Column classification
    report["numeric_columns"]  = df.select_dtypes(include=np.number).columns.tolist()
    report["categorical_columns"] = df.select_dtypes(include="object").columns.tolist()
    report["datetime_columns"] = df.select_dtypes(include="datetime").columns.tolist()
    report["final_shape"] = list(df.shape)

    return df, report


def get_statistical_summary(df: pd.DataFrame) -> Dict:
    """Rich statistical summary including correlations and outlier flags."""
    numeric_df = df.select_dtypes(include=np.number)
    result: Dict = {}

    if not numeric_df.empty:
        result["descriptive_stats"] = numeric_df.describe().round(3).to_dict()

        if len(numeric_df.columns) > 1:
            result["correlation_matrix"] = (
                numeric_df.corr().round(3).to_dict()
            )

        # Outlier counts via IQR
        outliers: Dict = {}
        for col in numeric_df.columns:
            q1, q3 = numeric_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = int(
                ((numeric_df[col] < q1 - 1.5 * iqr) |
                 (numeric_df[col] > q3 + 1.5 * iqr)).sum()
            )
            if n_out:
                outliers[col] = n_out
        result["outlier_counts"] = outliers

    return result
