"""
utils.py
--------
Data loading + preprocessing for the DAV mini-project.

The cleaning steps follow this order (see also data_cleaning_colab.ipynb):
  trim whitespace -> standardise missing markers -> remove duplicates ->
  fix data types -> standardise text case -> handle missing values ->
  outlier check (IQR)
Two small, deliberate differences (explained in the app and in the README):
  * Order_ID is NOT title-cased (it is an identifier, not a category).
  * IQR outliers are REPORTED but NOT capped, because capping would break the
    exact rule Sales = Units_Sold x Unit_Price.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "PROJECT_4__3SKILLS_.xlsx"
SALES_SHEET = "Sales_Data"        # the 12,000-row transaction table
SUMMARY_SHEET = "Sheet1"          # totals used in the Power BI dashboard

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

TEXT_COLS = ["Order_ID", "Product", "Category", "Region"]
CATEGORY_COLS = ["Product", "Category", "Region"]          # columns that get case-standardised
NUMERIC_COLS = ["Units_Sold", "Unit_Price", "Sales"]
MISSING_TOKENS = ["", "nan", "NaN", "NAN", "null", "NULL", "None", "none",
                  "N/A", "n/a", "NA", "na", "-", "--", "?"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Read the Sales_Data sheet exactly as it is in the Excel file."""
    return pd.read_excel(DATA_PATH, sheet_name=SALES_SHEET)


def load_dashboard_totals() -> pd.DataFrame:
    """Read the small totals table (Sheet1) that fed the Power BI KPI cards."""
    return pd.read_excel(DATA_PATH, sheet_name=SUMMARY_SHEET)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame):
    """
    Apply the notebook's cleaning steps to the raw data.
    Returns (clean_df, log_df, checks) where
      log_df  = one row per step with what was found / done
      checks  = dict of validation facts used by the app
    """
    df = raw.copy()
    log = []

    # 1. Trim whitespace in text columns --------------------------------------
    changed = 0
    for c in TEXT_COLS:
        before = df[c].astype(str)
        after = before.str.strip().str.replace(r"\s+", " ", regex=True)
        changed += int((before != after).sum())
        df[c] = after
    log.append(("1. Trim extra spaces", "Text columns: Order_ID, Product, Category, Region",
                f"{changed} cells changed"))

    # 2. Standardise missing-value markers ("N/A", "null", "-", ...) ------------
    n_tokens = 0
    for c in TEXT_COLS:
        mask = df[c].isin(MISSING_TOKENS)
        n_tokens += int(mask.sum())
        df.loc[mask, c] = np.nan
    log.append(("2. Standardise missing markers", f"Looked for {len(MISSING_TOKENS)} markers such as N/A, null, -, ?",
                f"{n_tokens} found"))

    # 3. Remove duplicates -----------------------------------------------------
    n_dup_rows = int(df.duplicated().sum())
    n_dup_ids = int(df["Order_ID"].duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(("3. Remove duplicates", "Fully duplicated rows and repeated Order_IDs",
                f"{n_dup_rows} duplicate rows removed, {n_dup_ids} repeated Order_IDs"))

    # 4. Fix data types --------------------------------------------------------
    d_before = int(df["Date"].isna().sum())
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    d_bad = int(df["Date"].isna().sum()) - d_before
    n_bad = 0
    for c in NUMERIC_COLS:
        nb = int(df[c].isna().sum())
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n_bad += int(df[c].isna().sum()) - nb
    log.append(("4. Fix data types", "Date -> datetime; Units_Sold, Unit_Price, Sales -> numeric",
                f"{d_bad + n_bad} values could not be converted"))

    # 5. Standardise text case (not for Order_ID) --------------------------------
    changed = 0
    for c in CATEGORY_COLS:
        before = df[c]
        after = before.str.title()
        changed += int((before.fillna("") != after.fillna("")).sum())
        df[c] = after
    log.append(("5. Standardise text case", "Title Case for Product, Category, Region (Order_ID left unchanged)",
                f"{changed} values changed"))

    # 6. Handle missing values (same strategy as notebook: median / mode) ---------
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        df = df.dropna(subset=["Date", "Order_ID"])
        for c in NUMERIC_COLS:
            df[c] = df[c].fillna(df[c].median())
        for c in CATEGORY_COLS:
            df[c] = df[c].fillna(df[c].mode()[0])
        action = "Filled numeric with median, text with most frequent value"
    else:
        action = "Nothing to fill (strategy in notebook: median for numbers, mode for text)"
    log.append(("6. Handle missing values", action, f"{n_missing} missing values found"))

    # 7. Business-rule validation -------------------------------------------------
    sales_rule_ok = int((df["Sales"] == df["Units_Sold"] * df["Unit_Price"]).sum())
    price_per_product = df.groupby("Product")["Unit_Price"].nunique().max()
    cat_per_product = df.groupby("Product")["Category"].nunique().max()
    log.append(("7. Validate consistency",
                "Sales = Units_Sold x Unit_Price; one price per product; one category per product",
                f"Sales rule holds in {sales_rule_ok:,} of {len(df):,} rows; "
                f"max prices/product = {price_per_product}; max categories/product = {cat_per_product}"))

    # 8. Feature engineering from the date ------------------------------------------
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Day_Name"] = df["Date"].dt.day_name()
    log.append(("8. Feature engineering", "Created Year, Month, Month_Name, Quarter, DayOfWeek, Day_Name from Date",
                "6 new columns"))

    # 9. Outliers: only reported (see outlier_table) ----------------------------------
    log.append(("9. Outlier check (IQR)", "Counted values outside Q1-1.5*IQR / Q3+1.5*IQR (see table below)",
                "Reported, not capped"))

    log_df = pd.DataFrame(log, columns=["Step", "What was checked / done", "Result on this dataset"])

    checks = dict(
        rows_raw=len(raw),
        rows_clean=len(df),
        sales_rule_ok=sales_rule_ok,
        price_per_product=int(price_per_product),
        cat_per_product=int(cat_per_product),
        order_id_unique=bool(df["Order_ID"].is_unique),
        missing=n_missing,
        duplicates=n_dup_rows,
        date_min=df["Date"].min(),
        date_max=df["Date"].max(),
    )
    return df.reset_index(drop=True), log_df, checks


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    """IQR outlier count for the numeric columns (same rule as the notebook, step 11)."""
    rows = []
    for c in NUMERIC_COLS:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < low) | (df[c] > high)).sum())
        rows.append({"Column": c, "Q1": q1, "Q3": q3, "Lower fence": low, "Upper fence": high,
                     "Outliers": n, "% of rows": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows)


def product_price_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product: its category and its (single) unit price."""
    return (df.groupby(["Product", "Category"], as_index=False)["Unit_Price"].first()
              .sort_values("Unit_Price", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Insights: every sentence is computed from the data (nothing typed in by hand)
# --------------------------------------------------------------------------- #
def fmt_money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f}bn"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:,.0f}"


def compute_insights(df: pd.DataFrame) -> list:
    total = df["Sales"].sum()
    out = []

    by_prod = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    top_p = by_prod.index[0]
    low2 = by_prod.tail(2)
    out.append(f"**{top_p}** is the top product with {fmt_money(by_prod.iloc[0])} sales "
               f"({100 * by_prod.iloc[0] / total:.1f}% of total). The two smallest products "
               f"({', '.join(low2.index)}) together make only {100 * low2.sum() / total:.2f}%.")

    by_cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    out.append(f"**{by_cat.index[0]}** contributes {100 * by_cat.iloc[0] / total:.1f}% of all sales, "
               f"followed by {by_cat.index[1]} ({100 * by_cat.iloc[1] / total:.1f}%).")

    m_sales = df.groupby("Month_Name")["Sales"].sum().reindex(MONTH_ORDER)
    top2 = m_sales.sort_values(ascending=False).head(2)
    out.append(f"**{top2.index[0]}** ({fmt_money(top2.iloc[0])}) and **{top2.index[1]}** "
               f"({fmt_money(top2.iloc[1])}) are the peak months; together they give "
               f"{100 * top2.sum() / total:.1f}% of yearly sales.")

    peak_months = list(top2.index)
    is_peak = df["Month_Name"].isin(peak_months)
    u_peak, u_rest = df.loc[is_peak, "Units_Sold"].mean(), df.loc[~is_peak, "Units_Sold"].mean()
    o_peak = df[is_peak].groupby("Month_Name").size().mean()
    o_rest = df[~is_peak].groupby("Month_Name").size().mean()
    out.append(f"The peak is driven by **bigger quantities per order**, not more orders: average units per "
               f"order is {u_peak:.1f} in {' & '.join(sorted(peak_months, key=MONTH_ORDER.index))} vs "
               f"{u_rest:.1f} in other months, while orders per month are about the same "
               f"({o_peak:.0f} vs {o_rest:.0f}).")

    y = df.groupby("Year")["Sales"].sum()
    chg = 100 * (y.iloc[-1] - y.iloc[0]) / y.iloc[0]
    out.append(f"Yearly sales: " + ", ".join(f"{k}: {fmt_money(v)}" for k, v in y.items()) +
               f" ({chg:+.1f}% from {y.index[0]} to {y.index[-1]}).")

    r = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    out.append(f"Regions are fairly balanced: **{r.index[0]}** is highest ({fmt_money(r.iloc[0])}) and "
               f"**{r.index[-1]}** lowest ({fmt_money(r.iloc[-1])}), a gap of "
               f"{100 * (r.iloc[0] - r.iloc[-1]) / r.iloc[-1]:.1f}%.")
    return out
