"""
Sales Forecasting & Demand Prediction - DAV Mini Project (Streamlit app)
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ml_pipeline as mlp
from utils import (DAY_ORDER, MONTH_ORDER, NUMERIC_COLS, DATA_PATH, clean_data, compute_insights,
                   fmt_money, load_dashboard_totals, load_raw, outlier_table, product_price_table, BASE_DIR)

st.set_page_config(page_title="Sales Forecasting & Demand Prediction", page_icon="📊", layout="wide")

BLUE, GOLD, PURPLE, GREY = "#2F7FD0", "#E0B92F", "#8E73D0", "#B8C0CC"
PRODUCT_ORDER = ["Laptop", "Phone", "Printer", "Desk", "Chair", "Headphones", "Notebook", "Pen"]

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.banner {background: linear-gradient(90deg,#17324f,#2F7FD0); color: white; padding: 18px 26px;
         border-radius: 12px; margin-bottom: 18px;}
.banner h1 {color: white; margin: 0; font-size: 1.8rem;}
.banner p {margin: 4px 0 0 0; opacity: .9; font-size: .95rem;}
div[data-testid="stMetric"] {background: #f3f7fc; border: 1px solid #d9e4f2; border-radius: 10px; padding: 12px 16px;}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {color: #17324f !important;}
.note {background:#fff8e1; border-left:5px solid #E0B92F; padding:10px 14px; border-radius:6px; color:#4a3b00;}
.good {background:#e8f5e9; border-left:5px solid #43a047; padding:10px 14px; border-radius:6px; color:#1b4d1f;}
.result {background:#17324f; color:white; padding:20px 26px; border-radius:12px; text-align:center;}
.result h1 {color:white; font-size:2.6rem; margin:0;}
.result p {margin:0; opacity:.85;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached data / models
# --------------------------------------------------------------------------- #
@st.cache_data
def get_data():
    raw = load_raw()
    clean, log, checks = clean_data(raw)
    return raw, clean, log, checks


@st.cache_resource(show_spinner=False)
def get_models(_clean: pd.DataFrame):
    """Use saved models if present (python train_models.py), otherwise train now (about a minute)."""
    saved = mlp.load_models()
    if saved is not None:
        return saved
    with st.spinner("Training models for the first time (about a minute)..."):
        res = mlp.train_all(_clean)
        try:
            mlp.save_models(res)
        except Exception:
            pass
    return res


raw, df, clean_log, checks = get_data()
results = get_models(df)


def banner(title, sub):
    st.markdown(f'<div class="banner"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def plot(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      title_font_size=16)
    st.plotly_chart(fig)


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 Sales Forecasting and Demand Prediction")
PAGES = ["1. Overview", "2. Data & Preprocessing", "3. EDA & Visualizations",
         "4. ML Models & Comparison", "5. Prediction"]
page = st.sidebar.radio("Go to section", PAGES)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records:** {len(df):,}  \n**Columns (raw):** {raw.shape[1]}  \n"
                    f"**Period:** {checks['date_min']:%b %Y} to {checks['date_max']:%b %Y}")


# =========================================================================== #
# 1. OVERVIEW
# =========================================================================== #
def page_overview():
    banner("Sales Forecasting & Demand Prediction",
           "DAV Data Analytics and Visualization Project")

    st.subheader("Project description")
    st.write(
        "This project analyses a sales dataset using **Python** (data cleaning and preprocessing, EDA, "
        "**two machine-learning models with comparison**) and a **Power BI dashboard**. "
        "The developed application presents the whole analysis and the prediction results in one interactive "
        "web application.")

    totals = load_dashboard_totals().iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders (records)", f"{len(df):,}")
    c2.metric("Total sales", f"{df['Sales'].sum():,.0f}")
    c3.metric("Units sold", f"{df['Units_Sold'].sum():,}")
    c4.metric("Average sales / order", f"{df['Sales'].mean():,.2f}")

    st.subheader("Dataset at a glance")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(pd.DataFrame({
            "Column": ["Date", "Order_ID", "Product", "Category", "Region", "Units_Sold", "Unit_Price", "Sales"],
            "Meaning": ["Order date", "Unique order number", "Product name (8 products)",
                        "Electronics / Furniture / Stationery", "North / South / East / West",
                        "Quantity sold in the order", "Price of one unit (fixed per product)",
                        "Order value = Units_Sold x Unit_Price"],
        }), hide_index=True)
    with right:
        st.write(f"- **{len(raw):,} records** (requirement: at least 1,000)")
        st.write(f"- **{df['Product'].nunique()} products**, {df['Category'].nunique()} categories, "
                 f"{df['Region'].nunique()} regions")
        st.write(f"- Dates from **{checks['date_min']:%d %b %Y}** to **{checks['date_max']:%d %b %Y}**")
        st.write(f"- Source file: `{DATA_PATH.name}` (sheet `Sales_Data`)")

    st.subheader("How this project meets the DAV requirements")
    st.dataframe(pd.DataFrame({
        "Requirement": ["At least 1,000 records", "Data preprocessing & wrangling", "EDA", "Meaningful visualizations",
                        "At least 2 ML models + comparison", "User-friendly web application"],
        "Where": [f"This page ({len(df):,} records)", "Section 2", "Section 3", "Sections 3 and 4",
                  "Section 4 (Linear Regression vs Random Forest)", "This Streamlit app, Section 5 = Prediction"],
    }), hide_index=True)

    st.subheader("Power BI dashboard")
    img = BASE_DIR / "assets" / "dashboard-1.png"
    if img.exists():
        st.image(str(img), caption="Sales Forecasting and Demand Prediction Dashboard (Power BI)")

    st.subheader("Dashboard insights - verified against the data")
    st.write("Every number below is recalculated from the Excel data, so it can be compared with the Power BI visuals.")
    insights = compute_insights(df)
    for t in insights:
        st.markdown(f"- {t}")

    st.markdown(
        '<div class="note"><b>Two points noted while verifying the dashboard:</b><br>'
        f'1) The KPI card shows "Sum of Sales = 1bn" because of rounding; the exact total is '
        f'<b>{df["Sales"].sum():,.0f}</b> (about 1.45bn), which also matches the totals sheet '
        f'({totals["TOTAL SALES"]:,}).<br>'
        '2) The donut chart titled "Forecasted sales trend" actually shows each month\'s share of '
        '<i>historical</i> sales, it is not a forecast. The real prediction work is done in Sections 4 and 5.</div>',
        unsafe_allow_html=True)


# =========================================================================== #
# 2. DATA & PREPROCESSING
# =========================================================================== #
def page_preprocessing():
    banner("Data & Preprocessing", "Raw data, cleaning steps, checks and feature engineering")

    st.subheader("1) Raw data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{raw.shape[0]:,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Missing values", int(raw.isna().sum().sum()))
    c4.metric("Duplicate rows", int(raw.duplicated().sum()))
    st.dataframe(raw.head(15), hide_index=True)

    with st.expander("Column types and summary statistics"):
        a, b = st.columns(2)
        a.write("**Data types**")
        a.dataframe(raw.dtypes.astype(str).rename("dtype").to_frame())
        b.write("**Describe (numeric columns)**")
        b.dataframe(raw[NUMERIC_COLS].describe().T)

    st.subheader("2) Cleaning steps")
    st.dataframe(clean_log, hide_index=True)
    st.markdown(
        '<div class="good"><b>Result:</b> this dataset was already clean - no missing values, no duplicates, no '
        'wrong data types - so the cleaned data has the same '
        f'<b>{checks["rows_clean"]:,}</b> rows as the raw data. The steps are still run (and shown) so the '
        'process is reproducible and would catch problems if the file changes.</div>', unsafe_allow_html=True)

    st.subheader("3) Consistency checks (data wrangling)")
    k1, k2, k3 = st.columns(3)
    k1.metric("Rows where Sales = Units x Price", f"{checks['sales_rule_ok']:,} / {checks['rows_clean']:,}")
    k2.metric("Order_ID unique?", "Yes" if checks["order_id_unique"] else "No")
    k3.metric("Different prices per product", checks["price_per_product"])
    st.write("Each product has exactly one price and belongs to exactly one category:")
    st.dataframe(product_price_table(df), hide_index=True)

    st.subheader("4) Outlier check (IQR method)")
    ot = outlier_table(df)
    st.dataframe(ot.style.format({"Q1": "{:,.1f}", "Q3": "{:,.1f}", "Lower fence": "{:,.1f}",
                                  "Upper fence": "{:,.1f}", "% of rows": "{:.2f}"}), hide_index=True)
    n_sales_out = int(ot.loc[ot["Column"] == "Sales", "Outliers"].iloc[0])
    st.markdown(
        f'<div class="note"><b>Why the outliers are not capped:</b> '
        f'{n_sales_out:,} Sales values are flagged, but they are real orders of expensive products '
        '(laptops, phones), not errors. Capping them would break the rule Sales = Units x Price and hide the '
        'products that earn the most money. Units_Sold has no outliers.</div>', unsafe_allow_html=True)

    st.subheader("5) Feature engineering")
    st.write("From the `Date` column I created: **Year, Month, Month_Name, Quarter, DayOfWeek, Day_Name** "
             "so that seasonality can be analysed and used by the models.")
    st.dataframe(df[["Date", "Year", "Month", "Month_Name", "Quarter", "DayOfWeek", "Day_Name"]].head(8),
                 hide_index=True)

    st.subheader("6) Preparation for machine learning")
    st.markdown(
        "- **One-hot encoding** for text inputs (Product, Region, Month, Day) - models need numbers.\n"
        "- **Standard scaling** for numeric inputs (Units_Sold, Unit_Price) in Task A.\n"
        f"- **Train/test split:** {int((1 - mlp.TEST_SIZE) * 100)}% / {int(mlp.TEST_SIZE * 100)}% "
        f"(`random_state={mlp.RANDOM_STATE}`), so every score is measured on data the model has never seen.\n"
        "- **Year is not used as a feature:** the data covers only 2023-2025, so a model could not "
        "reliably extrapolate to future years.\n"
        "- **Leakage control:** when predicting demand (Units_Sold) the price and sales columns are excluded, "
        "because Sales is calculated from Units_Sold.")

    st.download_button("Download cleaned dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_sales_data.csv", mime="text/csv")


# =========================================================================== #
# 3. EDA
# =========================================================================== #
def page_eda():
    banner("EDA & Visualizations", "Exploring sales and demand patterns (use the filters to drill down)")

    f1, f2, f3 = st.columns(3)
    years = f1.multiselect("Year", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
    regions = f2.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    cats = f3.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    d = df[df["Year"].isin(years) & df["Region"].isin(regions) & df["Category"].isin(cats)]
    if d.empty:
        st.warning("No data for this filter combination - please select at least one value in each filter.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders", f"{len(d):,}")
    k2.metric("Total sales", fmt_money(d["Sales"].sum()))
    k3.metric("Units sold", f"{d['Units_Sold'].sum():,}")
    k4.metric("Average sales / order", f"{d['Sales'].mean():,.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["💰 Sales analysis", "📦 Demand & seasonality", "📈 Distributions", "🔗 Relationships"])

    with tab1:
        a, b = st.columns(2)
        p = d.groupby("Product", as_index=False)["Sales"].sum().sort_values("Sales")
        a.plotly_chart(px.bar(p, x="Sales", y="Product", orientation="h", title="Sales by product",
                              color_discrete_sequence=[GOLD]).update_layout(height=370, template="plotly_white",
                                                                              margin=dict(l=10, r=10, t=50, b=10)))
        c = d.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        b.plotly_chart(px.bar(c, x="Category", y="Sales", title="Sales by category", text_auto=".3s",
                              color_discrete_sequence=[BLUE]).update_layout(height=370, template="plotly_white",
                                                                             margin=dict(l=10, r=10, t=50, b=10)))
        m = d.groupby("Month_Name", as_index=False)["Sales"].sum()
        m["Month_Name"] = pd.Categorical(m["Month_Name"], MONTH_ORDER, ordered=True)
        m = m.sort_values("Month_Name")
        fig = px.line(m, x="Month_Name", y="Sales", markers=True, title="Total sales by month (all selected years)")
        fig.update_traces(line_color=BLUE)
        plot(fig, 360)

        a, b = st.columns(2)
        ym = d.groupby(["Year", "Month_Name"], as_index=False)["Sales"].sum()
        a.plotly_chart(px.line(ym, x="Month_Name", y="Sales", color=ym["Year"].astype(str), markers=True,
                               category_orders={"Month_Name": MONTH_ORDER},
                               title="Monthly sales for each year (does the pattern repeat?)",
                               labels={"color": "Year"}).update_layout(height=370, template="plotly_white",
                                                                        margin=dict(l=10, r=10, t=50, b=10)))
        y = d.groupby("Year", as_index=False)["Sales"].sum()
        b.plotly_chart(px.bar(y, x="Year", y="Sales", text_auto=".3s", title="Sales by year",
                              color_discrete_sequence=[PURPLE]).update_layout(height=370, template="plotly_white",
                                                                               margin=dict(l=10, r=10, t=50, b=10),
                                                                               xaxis=dict(type="category")))
        a, b = st.columns(2)
        r = d.groupby("Region", as_index=False)["Sales"].sum()
        a.plotly_chart(px.bar(r, x="Region", y="Sales", text_auto=".3s", title="Sales by region",
                              color_discrete_sequence=[BLUE]).update_layout(height=370, template="plotly_white",
                                                                             margin=dict(l=10, r=10, t=50, b=10)))
        pr = d.pivot_table(index="Product", columns="Region", values="Sales", aggfunc="sum")
        b.plotly_chart(px.imshow(pr, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                                 title="Sales heatmap: product x region").update_layout(
            height=370, margin=dict(l=10, r=10, t=50, b=10)))

    with tab2:
        a, b = st.columns(2)
        mu = d.groupby("Month_Name", as_index=False)["Units_Sold"].mean()
        mu["Month_Name"] = pd.Categorical(mu["Month_Name"], MONTH_ORDER, ordered=True)
        mu = mu.sort_values("Month_Name")
        top2 = mu.nlargest(2, "Units_Sold")["Month_Name"].tolist()
        mu["Highlight"] = np.where(mu["Month_Name"].isin(top2), "Peak months", "Other months")
        a.plotly_chart(px.bar(mu, x="Month_Name", y="Units_Sold", color="Highlight", text_auto=".1f",
                              color_discrete_map={"Peak months": "#d9534f", "Other months": GREY},
                              title="Average units per order by month").update_layout(
            height=380, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), xaxis_title=None))
        oc = d.groupby("Month_Name", as_index=False).size()
        oc["Month_Name"] = pd.Categorical(oc["Month_Name"], MONTH_ORDER, ordered=True)
        oc = oc.sort_values("Month_Name")
        b.plotly_chart(px.bar(oc, x="Month_Name", y="size", title="Number of orders by month",
                              color_discrete_sequence=[PURPLE]).update_layout(
            height=380, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
            yaxis_title="Orders", xaxis_title=None))
        st.markdown('<div class="note"><b>Key finding:</b> the November-December peak is caused by larger quantities '
                    'per order, while the number of orders stays almost the same every month. '
                    'This is the main demand pattern in the data and the reason the month is used as a model input.</div>',
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        yu = d.groupby(["Year", "Month_Name"], as_index=False)["Units_Sold"].mean()
        a.plotly_chart(px.line(yu, x="Month_Name", y="Units_Sold", color=yu["Year"].astype(str), markers=True,
                               category_orders={"Month_Name": MONTH_ORDER},
                               title="Average units per order by month, per year",
                               labels={"color": "Year"}).update_layout(height=370, template="plotly_white",
                                                                        margin=dict(l=10, r=10, t=50, b=10)))
        du = d.groupby("Day_Name", as_index=False)["Units_Sold"].mean()
        b.plotly_chart(px.bar(du, x="Day_Name", y="Units_Sold", text_auto=".2f", title="Average units by day of week",
                              category_orders={"Day_Name": DAY_ORDER}, color_discrete_sequence=[BLUE]).update_layout(
            height=370, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), xaxis_title=None))

    with tab3:
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(d, x="Units_Sold", nbins=25, title="Distribution of Units_Sold",
                                    color_discrete_sequence=[BLUE]).update_layout(
            height=370, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), bargap=0.05))
        b.plotly_chart(px.box(d, x="Product", y="Sales", log_y=True, category_orders={"Product": PRODUCT_ORDER},
                              title="Sales per order by product (log scale)",
                              color_discrete_sequence=[GOLD]).update_layout(
            height=370, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10)))
        a, b = st.columns(2)
        pc = d.groupby("Product", as_index=False).size().sort_values("size", ascending=False)
        a.plotly_chart(px.bar(pc, x="Product", y="size", title="Number of orders by product",
                              color_discrete_sequence=[PURPLE]).update_layout(
            height=370, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Orders"))
        sh = d.groupby("Category", as_index=False)["Sales"].sum()
        b.plotly_chart(px.pie(sh, names="Category", values="Sales", hole=0.5, title="Share of sales by category")
                       .update_layout(height=370, margin=dict(l=10, r=10, t=50, b=10)))
        st.caption("All products are ordered about equally often, but their prices differ by a huge factor "
                   "(a pen costs 20, a laptop 55,000), which is why sales are dominated by a few products.")

    with tab4:
        num = d[["Units_Sold", "Unit_Price", "Sales", "Month", "Year", "DayOfWeek"]]
        corr = num.corr().round(2)
        a, b = st.columns(2)
        a.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                 title="Correlation matrix").update_layout(height=420,
                                                                           margin=dict(l=10, r=10, t=50, b=10)))
        s = d.sample(min(len(d), 2500), random_state=1)
        b.plotly_chart(px.scatter(s, x="Units_Sold", y="Sales", color="Product", opacity=0.6,
                                  category_orders={"Product": PRODUCT_ORDER},
                                  title="Units sold vs Sales (2,500-order sample)").update_layout(
            height=420, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10)))
        st.write(f"**How to read this:** Sales is strongly linked to Unit_Price (r = {corr.loc['Sales', 'Unit_Price']:.2f}) "
                 f"and moderately to Units_Sold (r = {corr.loc['Sales', 'Units_Sold']:.2f}), because "
                 "Sales = Units_Sold x Unit_Price and prices differ hugely between products. "
                 f"Units_Sold is linked to Month (r = {corr.loc['Units_Sold', 'Month']:.2f}, the Nov-Dec peak; "
                 "this is a curved/seasonal link, so a single r understates it) and not at all to day of week "
                 f"(r = {corr.loc['Units_Sold', 'DayOfWeek']:.2f}).")

    st.subheader("Key insights from the data")
    for t in compute_insights(d):
        st.markdown(f"- {t}")


# =========================================================================== #
# 4. ML MODELS & COMPARISON
# =========================================================================== #
def page_models():
    banner("ML Models & Comparison", "Linear Regression vs Random Forest, evaluated on unseen test data")

    st.write("Both models are **regression** models (the targets are numbers). Two questions are modelled:")
    st.markdown(
        "- **Task A - Sales (order value):** predict `Sales` from Product, Region, Month, Units_Sold and Unit_Price.\n"
        "- **Task B - Demand (units per order):** predict `Units_Sold` from Product, Region, Month and Day of week only.")
    task_label = st.radio("Choose task", [v["label"] for v in mlp.TASKS.values()], horizontal=True)
    key = next(k for k, v in mlp.TASKS.items() if v["label"] == task_label)
    r, cfg = results[key], mlp.TASKS[key]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target", cfg["target"])
    c2.metric("Input features", len(cfg["cat"]) + len(cfg["num"]))
    c3.metric("Training rows", f"{r['n_train']:,}")
    c4.metric("Test rows", f"{r['n_test']:,}")
    st.caption("Inputs: " + ", ".join(cfg["cat"] + cfg["num"]))

    with st.expander("About the two models (for viva)"):
        st.markdown(
            "**Linear Regression** fits one straight-line formula: prediction = intercept + sum(weight x input). "
            "It is fast and easy to explain, but it can only add effects together - it cannot represent "
            "'units **times** price'.\n\n"
            "**Random Forest** builds many decision trees (100 here) on random parts of the data and averages "
            "them. It can learn non-linear patterns and interactions between inputs. "
            f"Its `min_samples_leaf` setting (minimum rows per leaf, {r['rf_leaf']} here) was chosen by "
            f"{mlp.CV_FOLDS}-fold cross-validation on the **training** data only; larger values stop the "
            "trees from memorising noise.\n\n"
            "**Metrics:** **R²** = share of variation explained (1 = perfect, 0 = no better than predicting the "
            "average). **MAE** = average absolute error (same unit as the target). **RMSE** = like MAE but "
            "punishes big errors more. **CV R²** = average R² from cross-validation on the training data "
            "(checks stability). The *baseline* row (always predict the average) shows what 'no model' scores.")

    st.subheader("Performance comparison (test set)")
    m = r["metrics"].copy()
    best = r["best_name"]
    show = m.rename(columns={"R2": "R²", "CV_R2": f"CV R² ({mlp.CV_FOLDS}-fold)"})
    st.dataframe(show.style.format({"R²": "{:.4f}", "MAE": "{:,.2f}", "RMSE": "{:,.2f}",
                                    f"CV R² ({mlp.CV_FOLDS}-fold)": "{:.4f}"}, na_rep="-")
                 .apply(lambda row: ["background-color:#e8f5e9; font-weight:bold" if row["Model"] == best else ""
                                     for _ in row], axis=1), hide_index=True)
    st.success(f"Best model on this task (highest test R²): **{best}**")

    real = m[m["Model"].isin([mlp.LR_NAME, mlp.RF_NAME])]
    a, b = st.columns(2)
    a.plotly_chart(px.bar(real, x="Model", y="R2", text_auto=".4f", title="R² (higher is better)",
                          color="Model", color_discrete_sequence=[GOLD, BLUE]).update_layout(
        height=340, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="R²"))
    mm = real.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error")
    b.plotly_chart(px.bar(mm, x="Metric", y="Error", color="Model", barmode="group", text_auto=".3s",
                          title="Error (lower is better)", color_discrete_sequence=[GOLD, BLUE]).update_layout(
        height=340, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10)))

    st.subheader("Actual vs predicted (test set)")
    cols = st.columns(2)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(r["y_test"]), size=min(700, len(r["y_test"])), replace=False)
    lo, hi = float(r["y_test"].min()), float(r["y_test"].max())
    for col, name, colr in zip(cols, [mlp.LR_NAME, mlp.RF_NAME], [GOLD, BLUE]):
        f = go.Figure()
        f.add_trace(go.Scatter(x=r["y_test"][idx], y=r["preds"][name][idx], mode="markers",
                               marker=dict(color=colr, opacity=0.55, size=6), name="Predictions"))
        f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="black", dash="dash"),
                               name="Perfect prediction"))
        f.update_layout(title=name, xaxis_title="Actual", yaxis_title="Predicted", height=380,
                        template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"))
        col.plotly_chart(f)
    st.caption("Points on the dashed line = perfect predictions. Showing a random sample of 700 test rows.")

    st.subheader("Which inputs matter? (Random Forest feature importance)")
    imp = r["importance"].sort_values().reset_index()
    imp.columns = ["Feature", "Importance"]
    plot(px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[BLUE],
                title="Share of the Random Forest's decisions explained by each input"), 320)

    st.subheader("Interpretation")
    lr = m[m["Model"] == mlp.LR_NAME].iloc[0]
    rf = m[m["Model"] == mlp.RF_NAME].iloc[0]
    if key == "sales":
        st.markdown(
            f"- Random Forest reaches R² = **{rf['R2']:.4f}** (MAE {rf['MAE']:,.0f}) while Linear Regression "
            f"reaches **{lr['R2']:.4f}** (MAE {lr['MAE']:,.0f}). The forest wins clearly because Sales is a "
            "**multiplication** (Units x Price) and a straight line cannot model that.")
        st.markdown(
            '<div class="note"><b>Honest note for the viva:</b> because Sales is exactly Units_Sold x Unit_Price in all '
            f'{checks["rows_clean"]:,} rows, a near-perfect R² is expected - the forest has learned that rule, not a '
            'hidden business pattern. It is a good demonstration of how model type matters, but the genuinely '
            'predictive task is <b>Task B (demand)</b>.</div>', unsafe_allow_html=True)
    else:
        imp = r["importance"]
        base_mae = m[m["Model"].str.startswith("Baseline")].iloc[0]["MAE"]
        best_r2 = max(lr["R2"], rf["R2"])
        st.markdown(
            f"- Both models score similarly: Linear Regression R² = **{lr['R2']:.4f}**, Random Forest "
            f"R² = **{rf['R2']:.4f}**; both beat the baseline (R² ≈ 0) and cut the average error from "
            f"{base_mae:.2f} to about {min(lr['MAE'], rf['MAE']):.2f} units.\n"
            f"- The most important input is **{imp.index[0]}** ({100 * imp.iloc[0]:.0f}% of the forest's importance) - "
            "the November/December jump seen in the EDA. The other inputs together add only "
            f"{100 * imp.iloc[1:].sum():.0f}%.\n"
            f"- Roughly {100 * (1 - best_r2):.0f}% of the variation in order size is not explained by these inputs "
            "(it looks random in this dataset), so the model gives a **realistic estimate, not an exact value**. "
            "The more complex model did not help, which is a useful result in itself.")
        st.markdown(
            '<div class="note"><b>Limitation:</b> the dataset has no information such as promotions, customer type or '
            'stock levels, so demand cannot be predicted more precisely from it.</div>', unsafe_allow_html=True)


# =========================================================================== #
# 5. PREDICTION
# =========================================================================== #
def page_prediction():
    banner("Prediction", "Enter order details, click Predict and get the model's estimate")

    task_label = st.radio("What do you want to predict?",
                          ["Sales (order value)  -  Task A", "Demand (units per order)  -  Task B"], horizontal=True)
    key = "sales" if task_label.startswith("Sales") else "demand"
    r, cfg = results[key], mlp.TASKS[key]

    model_choice = st.selectbox("Model", [f"Best model automatically ({r['best_name']})", mlp.LR_NAME, mlp.RF_NAME])
    model_name = r["best_name"] if model_choice.startswith("Best") else model_choice

    prices = df.groupby("Product")["Unit_Price"].first()
    prod_cat = df.groupby("Product")["Category"].first()

    st.markdown("#### Input values")
    c1, c2, c3 = st.columns(3)
    product = c1.selectbox("Product", [p for p in PRODUCT_ORDER if p in prices.index])
    region = c2.selectbox("Region", sorted(df["Region"].unique()))
    month = c3.selectbox("Month", MONTH_ORDER, index=10)

    row = dict(Product=product, Region=region, Month_Name=month)
    if key == "sales":
        c4, c5 = st.columns(2)
        units = c4.number_input("Units sold", min_value=int(df["Units_Sold"].min()),
                                max_value=int(df["Units_Sold"].max()), value=10, step=1,
                                help="Range seen in the dataset: 1 to 25")
        c5.metric("Unit price (fixed for this product)", f"{prices[product]:,}")
        row.update(Units_Sold=units, Unit_Price=int(prices[product]))
    else:
        day = st.selectbox("Day of week", DAY_ORDER)
        row.update(Day_Name=day)
    st.caption(f"{product} belongs to the category **{prod_cat[product]}**.")

    if st.button("🔮 Predict", type="primary"):
        pred = mlp.predict_one(r["models"][model_name], row, key)
        mrow = r["metrics"][r["metrics"]["Model"] == model_name].iloc[0]

        if key == "sales":
            pred = max(pred, 0.0)
            st.markdown(f'<div class="result"><p>Predicted sales for this order ({model_name})</p>'
                        f'<h1>{pred:,.0f}</h1></div>', unsafe_allow_html=True)
            exact = row["Units_Sold"] * row["Unit_Price"]
            st.write("")
            a, b, c = st.columns(3)
            a.metric("Exact Units x Price", f"{exact:,}")
            b.metric("Model error for this input", f"{pred - exact:+,.0f}")
            c.metric(f"{model_name} test R²", f"{mrow['R2']:.4f}")
            st.caption("The dataset defines Sales = Units x Price, so the exact value is shown for comparison. "
                       "Linear Regression is visibly less accurate than Random Forest here.")
        else:
            pred = max(pred, 0.0)
            st.markdown(f'<div class="result"><p>Predicted units for this order ({model_name})</p>'
                        f'<h1>{pred:.1f} units</h1></div>', unsafe_allow_html=True)
            st.write("")
            hist = df[(df["Product"] == product) & (df["Month_Name"] == month)]["Units_Sold"]
            a, b, c = st.columns(3)
            a.metric("Typical error (MAE)", f"± {mrow['MAE']:.1f} units")
            b.metric(f"Historical avg: {product}, {month}", f"{hist.mean():.1f} units", help=f"{len(hist)} past orders")
            c.metric(f"{model_name} test R²", f"{mrow['R2']:.3f}")
            st.caption(f"Derived estimate: {pred:.1f} units x {prices[product]:,} = "
                       f"{pred * prices[product]:,.0f} expected order value.")
            st.markdown('<div class="note">This is an <b>estimate with wide uncertainty</b>: the model explains only '
                        f'about {mrow["R2"] * 100:.0f}% of the variation in order size (mainly the Nov-Dec peak). '
                        'Quote the typical error together with the prediction.</div>', unsafe_allow_html=True)

        with st.expander("Compare with the other model"):
            rows = []
            for n in [mlp.LR_NAME, mlp.RF_NAME]:
                rows.append({"Model": n, "Prediction": round(max(mlp.predict_one(r["models"][n], row, key), 0), 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("Choose the inputs above and click **Predict**.")


{"1. Overview": page_overview,
 "2. Data & Preprocessing": page_preprocessing,
 "3. EDA & Visualizations": page_eda,
 "4. ML Models & Comparison": page_models,
 "5. Prediction": page_prediction}[page]()
