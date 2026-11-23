# 📊 Sales Forecasting & Demand Prediction

### 🚀 Data Analytics and Visualization | Machine Learning Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-orange" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi" />
</p>

<p align="center">
  <b>Transforming historical sales data into meaningful business insights and predictions.</b>
</p>

---

## 📌 About The Project

This project focuses on analyzing historical sales transactions, identifying demand patterns, and building machine learning models for sales and demand prediction.

It combines **Data Preprocessing, Exploratory Data Analysis (EDA), Machine Learning, and Power BI Visualization** into an interactive web application developed using Streamlit.

## ✨ Project Highlights

| 📍 Feature | 📊 Details |
|---|---|
| Dataset | 12,000 records |
| Time Period | 2023 – 2025 |
| Products | 8 |
| Categories | 3 |
| Regions | 4 |
| Prediction Tasks | 2 |
| ML Algorithms | 2 |

## 🎯 Project Objectives

- Analyze historical sales transactions.
- Identify sales and demand patterns.
- Perform data cleaning and preprocessing.
- Build regression-based prediction models.
- Evaluate and compare model performance.
- Develop an interactive prediction application.
- Visualize business insights using Power BI.

## 🧠 Machine Learning Models

### 1️⃣ Sales Prediction
- Target: `Sales`
- Features: Product, Region, Month, Units Sold, Unit Price.

### 2️⃣ Demand Prediction
- Target: `Units_Sold`
- Features: Product, Region, Month, Day of Week.

### 🤖 Algorithms Used

- Linear Regression
- Random Forest Regressor

### 📈 Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- 3-Fold Cross Validation

**Model Configuration:** 80:20 train-test split | Random State: 42

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming |
| 🐼 Pandas | Data preprocessing |
| 🔢 NumPy | Numerical operations |
| 📊 Matplotlib | Visualization |
| 🎨 Seaborn | Statistical visualization |
| 🤖 Scikit-learn | Machine Learning |
| 🌐 Streamlit | Web application |
| 📈 Power BI | Dashboard |
| ☁️ Google Colab | Data cleaning |

## 📂 Project Structure

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

## ⚙️ Installation & Setup

**Step 1: Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
```

**Step 2: Navigate to the project folder**

```bash
cd dav_ml
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Train models (Optional)**

```bash
python train_models.py
```

**Step 5: Run the application**

```bash
python -m streamlit run app.py
```

🌐 The application opens at `http://localhost:8501`.

## 🖥️ Application Demo

🎬 **Watch the application demonstration:** [Add Your Video Link Here]

## 📊 Power BI Dashboard

📈 The Power BI dashboard presents sales trends, regional performance, and other business insights.

📷 **Dashboard Preview:** See the dashboard files in the `assets/` folder.

## 💡 Important Insights

- Sales is calculated as Units Sold × Unit Price.
- November and December have higher average order quantities.
- Monthly order counts remain almost constant.
- Sales prediction is relatively straightforward because sales is directly related to units and price.
- Demand prediction is more challenging because the dataset contains limited explanatory features.

## 🎓 Learning Outcomes

- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature engineering.
- Regression algorithms.
- Model evaluation and comparison.
- Interactive application development.
- Business data visualization.

## 🔮 Future Enhancements

- Incorporate additional business-related features.
- Explore more machine learning algorithms.
- Improve demand prediction using richer datasets.
- Enhance
