# 📊 Currency Movement Prediction Pipeline (Kedro + Machine Learning)

---

## 📘 Overview

This project implements a modular and reproducible machine learning pipeline using **Kedro** to analyze and predict short-term currency movement direction.

The goal is to build a structured, production-style ML workflow that transforms raw financial time-series data into engineered features and predictive models.

---

## 🎯 Objective

The objective of this project is to:

> Develop a clean, modular machine learning pipeline that processes financial time-series data and predicts short-term currency movement direction.

---

## 🏗️ Project Structure (Kedro Pipeline)

The project is organized using Kedro’s standard pipeline architecture:

src/
├── pipelines/
│ ├── data_processing/
│ ├── feature_engineering/
│ ├── modeling/
│ ├── evaluation/


Each pipeline consists of:

- **Nodes** → individual transformation functions  
- **Pipeline** → connected sequence of nodes  
- **Catalog** → dataset management  

---

## ⚙️ Pipeline Stages

### 1. Data Processing
- Load raw financial time-series data  
- Sort by currency, country, and date  
- Handle missing values  

---

### 2. Feature Engineering

The model uses engineered features such as:

- Lag features (1–14 days)  
- Momentum indicators  
- Volatility ratios  
- Trend strength metrics  

---

### 3. Label Construction

The target variable is defined as:

> Next-day movement direction (up or down)

---

### 4. Model Training

A **CatBoostClassifier** is trained using:

- Time-series aware splitting  
- Categorical feature support  
- Early stopping for generalization control  

---

### 5. Evaluation

Model performance is evaluated using:

- Accuracy  
- Precision / Recall / F1-score  
- Confusion matrix  
- Feature importance analysis  

---

## 📊 Key Features

- Fully modular ML pipeline (Kedro)  
- Time-series safe data handling  
- Feature engineering for financial data  
- Categorical feature support  
- Reproducible experiments  
- Structured evaluation workflow  

---

## 🧠 Technologies Used

- Python  
- Kedro  
- Pandas / NumPy  
- CatBoost  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 📁 Benefits of Using Kedro

This project leverages Kedro to ensure:

- ✔ Reproducibility  
- ✔ Clean separation of logic  
- ✔ Scalable architecture  
- ✔ Easy experimentation  
- ✔ Production-ready structure  

---

## 🚀 Future Improvements

- Add walk-forward validation pipeline  
- Hyperparameter tuning pipeline  
- Automated feature selection nodes  
- Data versioning integration  
- Experiment tracking (e.g., MLflow)  

---

## 📌 Conclusion

This project demonstrates a structured machine learning workflow using Kedro, focusing on clean architecture, reproducibility, and robust feature engineering for time-series financial data.
