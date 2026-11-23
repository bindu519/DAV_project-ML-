# Sales Forecasting & Demand Prediction – DAV Mini Project

DAV Data Analytics and Visualization Project: Python analysis, machine learning and a Power BI dashboard,
presented in an interactive Streamlit web application.

## How to run

1. Install Python 3.9+ and open a terminal in this folder.
2. (Recommended) create a virtual environment
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install libraries: `pip install -r requirements.txt`
4. (Optional, ~1 min) train and save the models: `python train_models.py`
   – if you skip this, the app trains automatically the first time it opens.
5. Start the app: `streamlit run app.py`  → opens at http://localhost:8501

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app with 5 sections (Overview, Data & Preprocessing, EDA, ML Models, Prediction) |
| `utils.py` | Data loading, cleaning steps, feature engineering, insights |
| `ml_pipeline.py` | Model definitions, training, evaluation, saving/loading, prediction |
| `train_models.py` | Command-line script that trains both models and prints the results |
| `data/PROJECT_4__3SKILLS_.xlsx` | Project dataset (sheet `Sales_Data`, 12,000 rows) |
| `data_cleaning_colab.ipynb` | Cleaning notebook (Google Colab, kept as reference) |
| `assets/` | Power BI dashboard (PDF + image used in the Overview page) |
| `models/` | Saved trained models (created by `train_models.py`) |

## Key facts about the data (all verified in the app)

* 12,000 records, 8 columns, Jan 2023 – Dec 2025, 8 products, 3 categories, 4 regions.
* No missing values, no duplicates.
* `Sales = Units_Sold × Unit_Price` in **every** row; every product has exactly one price.
* The only real demand pattern: **Nov & Dec orders are bigger** (about 14 units vs about 8). The number of orders
  per month is almost constant.

## ML design (and why)

| | Task A – Sales | Task B – Demand |
|---|---|---|
| Target | `Sales` | `Units_Sold` |
| Inputs | Product, Region, Month, Units_Sold, Unit_Price | Product, Region, Month, Day of week |
| Models | Linear Regression, Random Forest | Linear Regression, Random Forest |
| Meaning | Easy task (a multiplication) – shows linear vs non-linear model | Real prediction task – realistic, modest scores |

Split: 80/20 (`random_state=42`). Metrics: R², MAE, RMSE, 3-fold CV R² (+ "predict the mean" baseline).
Scores are produced when you run the code; they are not typed into the app.

## Viva cheat-sheet

* **Why two tasks?** Because Sales is exactly Units × Price, predicting Sales from Units and Price is almost
  trivial. So Task A shows model differences (a forest can learn a product, a straight line cannot) and Task B is
  the honest demand-prediction problem.
* **Why is Random Forest R² ≈ 1.0 in Task A?** It learned the multiplication rule. This is expected, not a
  discovery – say so.
* **Why is Task B only ≈ 0.22?** The dataset has no information (promotions, customers, stock) that explains
  order size except the month. Model quality is limited by data, not by the algorithm.
* **Why did the complex model not win Task B?** There is only a simple seasonal signal; Linear Regression with
  month dummies captures it as well as a forest.
* **Why are outliers not capped?** The flagged Sales values are real orders for expensive products;
  capping would break Sales = Units × Price.
* **Why is `Year` not a feature?** Only 3 years of data; models cannot extrapolate to future years.
* **Data leakage:** price and sales are excluded when predicting units, because Sales is calculated from units.
* **Dashboard note:** the "Forecasted sales trend" donut shows historical monthly share, not a forecast; the
  real prediction is in this app.
