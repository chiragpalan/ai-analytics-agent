"""
Data Ingestion
Handles loading CSV and multi-sheet Excel files into a standardised
dict[sheet_name -> pd.DataFrame] structure.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Tuple


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def load_file(uploaded_file) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Load a user-uploaded CSV or Excel file.

    Returns
    -------
    sheets : dict  {sheet_name: DataFrame}
    file_type : str  "csv" | "excel"
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = _read_csv(uploaded_file)
        return {"main": df}, "csv"

    if name.endswith((".xlsx", ".xls")):
        sheets = _read_excel(uploaded_file)
        return sheets, "excel"

    raise ValueError(
        f"Unsupported file type: '{uploaded_file.name}'. "
        "Please upload a .csv, .xlsx, or .xls file."
    )


def get_data_summary(sheets: Dict[str, pd.DataFrame]) -> Dict:
    """Return a lightweight descriptive summary of every sheet."""
    summary: Dict = {}
    for name, df in sheets.items():
        summary[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
    return summary


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def _read_csv(file) -> pd.DataFrame:
    """Try common encodings to read a CSV robustly."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8", errors="replace")


def _read_excel(file) -> Dict[str, pd.DataFrame]:
    """Read all sheets from an Excel file, skipping completely empty ones."""
    xl = pd.ExcelFile(file)
    sheets: Dict[str, pd.DataFrame] = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        if df.empty:
            continue
        sheets[sheet] = df
    if not sheets:
        raise ValueError("The Excel file contains no non-empty sheets.")
    return sheets
