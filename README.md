# Sales Forecasting & Demand Prediction
### Data Analytics and Visualization | Machine Learning Project

A data analytics and machine learning project developed using Python to analyze historical sales data, identify demand patterns, build predictive models, and visualize business insights.

The project integrates **Data Preprocessing, Exploratory Data Analysis (EDA), Machine Learning, and Power BI** into an interactive web application developed using Streamlit.

---

## 1. Project Overview

This project focuses on analyzing historical sales transactions and developing machine learning models for two prediction tasks:

- **Sales Prediction:** Estimating sales based on product, region, month, units sold, and unit price.
- **Demand Prediction:** Predicting the number of units sold using product, region, month, and day of the week.

Two regression algorithms, Linear Regression and Random Forest Regressor, are implemented and evaluated for both tasks.

The project also includes an interactive Streamlit application for data exploration, model evaluation, and real-time user-input-based predictions.

## 2. Key Features

- Data loading and preprocessing
- Data cleaning and feature engineering
- Exploratory Data Analysis (EDA)
- Sales and demand pattern analysis
- Linear Regression implementation
- Random Forest Regressor implementation
- Model training and performance evaluation
- Interactive prediction interface
- Power BI dashboard integration
- Automated model training and saving

## 3. Dataset Description

The project uses a sales dataset containing 12,000 records and 8 columns.

| Dataset Attribute | Details |
|---|---|
| Total Records | 12,000 |
| Total Columns | 8 |
| Time Period | January 2023 – December 2025 |
| Products | 8 |
| Categories | 3 |
| Regions | 4 |
| Missing Values | None |
| Duplicate Records | None |

**Important Data Observations:**

- Sales is calculated as Units Sold × Unit Price.
- Every product has exactly one unit price.
- November and December have higher average order quantities.
- The number of orders per month is almost constant.

## 4. Machine Learning Implementation

Two prediction tasks are developed using the following algorithms:

**Algorithms Used:**

1. Linear Regression
2. Random Forest Regressor

### Task A – Sales Prediction

- **Target Variable:** Sales
- **Input Features:** Product, Region, Month, Units Sold, Unit Price
- **Objective:** Predict sales based on product and transaction information.

This task demonstrates how machine learning models handle the relationship between units sold and unit price.

### Task B – Demand Prediction

- **Target Variable:** Units Sold
- **Input Features:** Product, Region, Month, Day of Week
- **Objective:** Estimate product demand based on historical sales patterns.

This task focuses on understanding demand variations and seasonal patterns.

### Model Evaluation

The models are evaluated using:

| Metric | Purpose |
|---|---|
| R² Score | Measures model performance |
| MAE | Measures average absolute prediction error |
| RMSE | Measures prediction error with greater penalty for large errors |
| 3-Fold Cross Validation | Evaluates model consistency |

- Train-Test Split: 80:20
- Random State: 42
- Baseline: Predict-the-mean model

Model evaluation scores are generated during execution rather than manually entered into the application.

## 5. Technologies and Tools

| Technology | Application |
|---|---|
| Python | Programming and data analysis |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| Streamlit | Interactive web application |
| Power BI | Dashboard development |
| Google Colab | Data cleaning and notebook execution |
| Jupyter Notebook | Data analysis |

## 6. Project Structure

```text
DAV-ML-Project/
│
├── app.py
├── utils.py
├── ml_pipeline.py
├── train_models.py
├── requirements.txt
│
├── data/
│   └── PROJECT_4__3SKILLS_.xlsx
│
├── data_cleaning_colab.ipynb
│
├── assets/
│   ├── dashboard-1.png
│   └── project_4_dashboard.pdf
│
├── models/
│   └── trained_models.joblib
│
└── .streamlit/
    └── config.toml
```

### Main Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application with five sections |
| `utils.py` | Data loading, cleaning, feature engineering, and insights |
| `ml_pipeline.py` | Model definitions, training, evaluation, and prediction |
| `train_models.py` | Script for training both machine learning models |
| `requirements.txt` | Required Python libraries |
| `data_cleaning_colab.ipynb` | Dataset cleaning and preprocessing notebook |
| `assets/` | Power BI dashboard files |
| `models/` | Saved trained machine learning models |

## 7. Application Modules

The Streamlit application contains five main sections:

1. **Overview:** Introduction and project summary.
2. **Data & Preprocessing:** Dataset information and cleaning operations.
3. **EDA:** Exploratory analysis and data visualizations.
4. **ML Models:** Model training, evaluation, and comparison.
5. **Prediction:** Interactive sales and demand prediction.

## 8. Installation and Execution

Follow these steps to run the project locally.

**Step 1: Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
```

**Step 2: Navigate to the project directory**

```bash
cd dav_ml
```

**Step 3: Create a virtual environment**

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Train the models (Optional)**

```bash
python train_models.py
```

The application automatically trains the models during its first run if previously saved models are unavailable.

**Step 6: Launch the application**

```bash
python -m streamlit run app.py
```

The application will be available at:

`http://localhost:8501`

## 9. Important Technical Considerations

- **Sales Calculation:** Sales is directly calculated from Units Sold × Unit Price.
- **Demand Prediction:** Sales and unit price are excluded from demand prediction to avoid data leakage.
- **Outlier Handling:** Genuine high-value sales transactions are retained rather than capped.
- **Year Feature:** Year is excluded because the dataset contains only three years of historical information.
- **Model Performance:** Demand prediction performance is limited by the available dataset features and the relatively simple seasonal pattern.
- **Dashboard Interpretation:** The forecasted sales trend donut represents historical monthly sales distribution, not an independent future forecast. Actual predictions are generated through the Streamlit application.

## 10. Learning Outcomes

Through this project, the following concepts were implemented and explored:

- Data cleaning and preprocessing techniques.
- Exploratory Data Analysis and visualization.
- Feature engineering and regression analysis.
- Implementation of Linear Regression and Random Forest.
- Model evaluation using regression metrics.
- Understanding data leakage and model limitations.
- Development of interactive applications using Streamlit.
- Integration of analytical results with Power BI dashboards.

## 11. Future Enhancements

- Incorporating additional business-related features into the dataset.
- Exploring more machine learning algorithms.
- Improving demand prediction through additional relevant features.
- Enhancing interactive visualizations and analytical capabilities.

---

### Project Information

**Project Title:** Sales Forecasting & Demand Prediction  
**Domain:** Data Analytics and Machine Learning  
**Project Type:** Data Analytics and Visualization (DAV) Mini Project  
**Programming Language:** Python  
**Application Framework:** Streamlit

*Developed as part of the Data Analytics and Visualization academic mini project.*
