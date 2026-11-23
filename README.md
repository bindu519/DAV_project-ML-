# 📊 Sales Forecasting & Demand Prediction

### 🚀 Data Analytics and Visualization | Machine Learning Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-orange" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20Application-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi" />
  <img src="https://img.shields.io/badge/Project-DAV-green" />
</p>

<p align="center">
  <b>Turning historical sales data into meaningful business insights and predictions.</b>
</p>

---

## 📌 1. Project Overview

Sales Forecasting & Demand Prediction is a Python-based Data Analytics and Machine Learning project developed to analyze historical sales transactions, identify demand patterns, and predict sales and product demand.

The project integrates **Data Preprocessing, Exploratory Data Analysis (EDA), Machine Learning, and Power BI Visualization** into an interactive web application developed using Streamlit.

It focuses on two prediction tasks using Linear Regression and Random Forest Regressor algorithms.

### 🎯 Project Objectives

- Analyze historical sales transactions.
- Identify sales and demand patterns.
- Perform data cleaning and preprocessing.
- Develop machine learning models for prediction.
- Evaluate and compare model performance.
- Build an interactive prediction application.
- Present business insights through Power BI dashboards.

---

## 📊 2. Dataset Description

The project uses a structured sales dataset containing historical transaction information.

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

### 🔍 Key Data Observations

- Sales is calculated as `Units_Sold × Unit_Price` in every row.
- Every product has exactly one unit price.
- November and December have higher average order quantities (approximately 14 units compared with approximately 8 units).
- The number of orders per month is almost constant.

---

## 🧹 3. Data Preprocessing & EDA

The dataset is processed and analyzed before model development.

### Data Preprocessing
- Data loading and inspection.
- Data cleaning and validation.
- Missing-value and duplicate checks.
- Feature engineering.
- Preparation of data for machine learning.

### Exploratory Data Analysis
- Sales distribution analysis.
- Monthly sales pattern analysis.
- Product and category analysis.
- Regional sales analysis.
- Demand pattern identification.
- Outlier analysis.

---

## 🤖 4. Machine Learning Implementation

Two prediction tasks are implemented using two regression algorithms.

### 📈 Task A – Sales Prediction

| Attribute | Description |
|---|---|
| Target Variable | `Sales` |
| Input Features | Product, Region, Month, Units_Sold, Unit_Price |
| Algorithms | Linear Regression, Random Forest Regressor |
| Objective | Predict sales based on transaction information |

This task demonstrates how different regression algorithms learn the relationship between units sold, unit price, and sales.

### 📦 Task B – Demand Prediction

| Attribute | Description |
|---|---|
| Target Variable | `Units_Sold` |
| Input Features | Product, Region, Month, Day of Week |
| Algorithms | Linear Regression, Random Forest Regressor |
| Objective | Estimate product demand using historical patterns |

This task focuses on identifying demand variations and understanding seasonal patterns.

### ⚙️ Model Configuration

- Train-Test Split: 80:20
- Random State: 42
- Cross Validation: 3-Fold
- Baseline: Predict-the-mean model

### 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| R² Score | Measures how well the model explains variations in the target |
| MAE | Measures average absolute prediction error |
| RMSE | Measures prediction error with greater penalty for larger errors |
| 3-Fold Cross Validation | Evaluates model consistency across data splits |

*Model evaluation scores are generated during execution and are not manually entered into the application.*

---

## 🧠 5. Machine Learning Algorithms

### 1️⃣ Linear Regression

Linear Regression is a supervised machine learning algorithm used to model the relationship between input features and a continuous target variable.

### 2️⃣ Random Forest Regressor

Random Forest Regressor is an ensemble learning algorithm that combines multiple decision trees to generate predictions.

Both algorithms are implemented for sales and demand prediction, allowing their performance to be compared.

---

## 🛠️ 6. Technologies & Tools Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming and data analysis |
| 🐼 Pandas | Data manipulation and preprocessing |
| 🔢 NumPy | Numerical computations |
| 📊 Matplotlib | Data visualization |
| 🎨 Seaborn | Statistical visualization |
| 🤖 Scikit-learn | Machine learning and model evaluation |
| 🌐 Streamlit | Interactive web application |
| 📈 Power BI | Dashboard development |
| ☁️ Google Colab | Data cleaning and notebook execution |

---

## 🖥️ 7. Application Modules

The Streamlit application consists of five main sections.

| Module | Description |
|---|---|
| 🏠 Overview | Project introduction and summary |
| 🧹 Data & Preprocessing | Dataset information and preprocessing |
| 📊 EDA | Exploratory analysis and visualizations |
| 🤖 ML Models | Model training, evaluation, and comparison |
| 🔮 Prediction | Interactive sales and demand predictions |

---

## 🎬 Application Demo

Watch the project demonstration to explore the working of the interactive Streamlit application.

<p align="center">
  <a href="https://drive.google.com/file/d/1mikN_TmD_Nr7DMvVsmqSO9UtpeEQo-a3/view?usp=drive_link">
    <img src="https://img.shields.io/badge/▶_View_Application_Demo-FF4B4B?style=for-the-badge&logo=google-drive&logoColor=white" />
  </a>
</p>
**The demonstration covers:**

- Project overview and dataset exploration.
- Data preprocessing and EDA.
- Machine learning model comparison.
- Interactive sales prediction.
- Interactive demand prediction.

*Replace `YOUR_VIDEO_LINK` with your actual video link.*

---

## 📈 9. Power BI Dashboard

The project also includes a Power BI dashboard to present sales trends and business insights through interactive visualizations.

**Dashboard files are available in the `assets/` folder.**

- Sales trend analysis.
- Product and category performance.
- Regional sales analysis.
- Historical monthly sales distribution.

*Note: The "Forecasted sales trend" donut represents historical monthly sales share, not a future forecast. Actual predictions are generated through the Streamlit application.*

---

## 📂 10. Project Structure

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

### 📄 Important Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `utils.py` | Data loading, cleaning, feature engineering, and insights |
| `ml_pipeline.py` | Model definitions, training, evaluation, and prediction |
| `train_models.py` | Script for training both machine learning models |
| `requirements.txt` | Required Python libraries |
| `PROJECT_4__3SKILLS_.xlsx` | Project dataset |
| `data_cleaning_colab.ipynb` | Data cleaning and preprocessing notebook |
| `assets/` | Power BI dashboard files |
| `models/` | Saved trained machine learning models |

---

## ⚙️ 11. Installation & Setup

Follow these steps to run the project on your local system.

### Step 1: Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd DAV_project-ML-
```

### Step 3: Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Train the Models (Optional)

```bash
python train_models.py
```

If saved models are unavailable, the application automatically trains them during its first run.

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

🌐 The application will open at:

`http://localhost:8501`

---

## 💡 12. Important Technical Considerations

- **Sales Calculation:** Sales is directly calculated as Units Sold × Unit Price.
- **Demand Prediction:** Sales and unit price are excluded from demand prediction to avoid data leakage.
- **Outlier Handling:** Genuine high-value sales transactions are retained rather than capped, because capping would break the sales calculation relationship.
- **Year Feature:** Year is excluded because the dataset contains only three years of historical information.
- **Demand Model Performance:** Demand prediction is limited by the available explanatory features and the relatively simple seasonal pattern.
- **Dashboard Interpretation:** The historical monthly sales share donut is not a future forecast.

---

## 🎓 13. Learning Outcomes

Through this project, the following concepts and practical skills were explored:

- Data cleaning and preprocessing techniques.
- Exploratory Data Analysis and visualization.
- Feature engineering.
- Implementation of regression algorithms.
- Machine learning model evaluation and comparison.
- Understanding data leakage and model limitations.
- Interactive application development using Streamlit.
- Business data visualization using Power BI.

---

## 🔮 14. Future Enhancements

- Incorporate additional business-related features.
- Explore more machine learning algorithms.
- Improve demand prediction using richer datasets.
- Enhance interactive visualizations.
- Expand analytical capabilities with additional historical data.

---

## 👩‍💻 Project Information

| Attribute | Details |
|---|---|
| Project Title | Sales Forecasting & Demand Prediction |
| Domain | Data Analytics & Machine Learning |
| Project Type | DAV Mini Project |
| Programming Language | Python |
| Application Framework | Streamlit |
| Visualization Tool | Power BI |

---

<p align="center">
  <b>✨ Developed as part of the Data Analytics and Visualization (DAV) Mini Project ✨</b>
</p>

<p align="center">
  <i>Analyzing Data • Discovering Patterns • Predicting Outcomes</i>
</p>
