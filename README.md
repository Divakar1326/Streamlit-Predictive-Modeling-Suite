# 📊 Streamlit Predictive Modeling Suite

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20ML-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/SMOTE-Imbalanced%20Data-6A1B9A?style=for-the-badge" alt="SMOTE">
  <img src="https://img.shields.io/badge/Status-Completed-2EA44F?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <b>An interactive Streamlit suite combining four machine learning applications for healthcare, spam detection, drug classification, and stock-price analysis.</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-applications">Applications</a> •
  <a href="#-models">Models</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-getting-started">Getting Started</a>
</p>

---

## ✨ Overview

**Streamlit Predictive Modeling Suite** is an interactive collection of machine learning applications built with **Python and Streamlit**.

Instead of focusing on a single prediction problem, the project brings together four different ML workflows in one suite:

| Application | Problem | Main Techniques |
|---|---|---|
| 💊 **MedPredictor** | Drug classification | Classification algorithms |
| ✉️ **SpamDetective** | Spam email detection | NLP + Naive Bayes |
| 📈 **StockVision** | Stock-price prediction | Linear Regression |
| 🏥 **Health Prediction** | Disease-risk prediction | Classification + SMOTE |

Each application allows users to work with data, train or evaluate models, visualize results, and generate predictions through an interactive interface.

---

# 🚀 Applications

## 💊 1. MedPredictor — Drug Classification

A machine learning application for predicting a suitable drug class from patient-related features such as age, sex, blood pressure, and cholesterol.

### 🤖 Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

### ✨ Features

- 👤 Enter patient-related feature values
- 💊 Predict the drug class
- 📊 Compare classification performance
- 📋 View classification reports
- 🔲 View confusion matrices
- 📈 Compare model accuracy

---

## ✉️ 2. SpamDetective — Spam Classification

An NLP-based application for classifying email/message content as **spam or ham**.

### 🤖 Models

- Multinomial Naive Bayes
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Complement Naive Bayes

### ✨ Features

- ✉️ Enter custom email content
- 🚨 Detect spam or legitimate messages
- ☁️ Generate spam/non-spam word clouds
- 📊 View classification reports
- 🔲 View confusion matrices
- 📈 Compare prediction accuracy
- 💾 Support model saving

---

## 📈 3. StockVision — AI-Driven Stock Price Prediction

An interactive stock-price prediction application using historical market data and **Linear Regression**.

The application works with historical stock data and provides visual analysis alongside model-based predictions.

### 📊 Supported Stocks

The project provides options for:

- Apple (`AAPL`)
- Google (`GOOG`)
- Microsoft (`MSFT`)
- Amazon (`AMZN`)
- Custom stock ticker input

### ✨ Features

- 📡 Fetch historical market data through Yahoo Finance
- 📅 Analyze up to 10 years of historical data
- 📈 Visualize stock-price trends
- 🧮 Train a Linear Regression model
- 🔮 Generate future price predictions
- 📊 Display R², MAE, and MSE metrics
- 📉 Visualize prediction results

> ⚠️ **Disclaimer:** This is an educational machine learning project and should not be treated as financial advice or as a reliable trading system.

---

## 🏥 4. Health Disease Prediction

An interactive machine learning application for predicting the risk of selected health conditions.

### 🩺 Supported Prediction Tasks

- Cancer
- CHD Heart Disease
- Diabetes
- Stroke

### 🤖 Models

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Complement Naive Bayes
- Support Vector Machine
- Logistic Regression

### ✨ Features

- 🧑‍⚕️ Enter custom health-related feature values
- 🧹 Handle missing values
- 🔢 Encode categorical variables
- ⚖️ Apply SMOTE for imbalanced datasets
- 📊 View classification reports
- 🔲 View confusion matrices
- 📈 Evaluate model accuracy
- 🔗 Explore feature correlation through correlation matrices

> ⚠️ **Disclaimer:** Predictions produced by this project are for educational and demonstration purposes only. They are not medical diagnoses and should not be used for healthcare decisions.

---

# 🧠 Machine Learning Techniques

This repository demonstrates several supervised learning approaches across different prediction problems.

| Technique | Used For |
|---|---|
| Logistic Regression | Drug & disease classification |
| K-Nearest Neighbors | Drug classification |
| Support Vector Machine | Drug & disease classification |
| Naive Bayes | Drug, spam & disease classification |
| Linear Regression | Stock-price prediction |
| Artificial data balancing with SMOTE | Imbalanced health datasets |
| NLP / text processing | Spam detection |

---

# 🔄 Application Workflow

```text
             ┌─────────────────────┐
             │       Dataset       │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Data Preprocessing  │
             │ Cleaning / Encoding │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Feature Preparation │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Model Training      │
             │ & Evaluation        │
             └──────────┬──────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
      ┌───────────────┐   ┌───────────────┐
      │ Visualization │   │ User Input    │
      └───────┬───────┘   └───────┬───────┘
              │                   │
              └─────────┬─────────┘
                        ▼
             ┌─────────────────────┐
             │     Prediction      │
             └─────────────────────┘
```

---

# 🧰 Tech Stack

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib">
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square" alt="Seaborn">
<img src="https://img.shields.io/badge/WordCloud-Text%20Visualization-8A2BE2?style=flat-square" alt="WordCloud">
<img src="https://img.shields.io/badge/SMOTE-Imbalanced%20Learning-6A1B9A?style=flat-square" alt="SMOTE">
<img src="https://img.shields.io/badge/Yahoo%20Finance-Market%20Data-720E9E?style=flat-square" alt="Yahoo Finance">

</p>

---

# 📁 Repository Structure

```text
Streamlit-Predictive-Modeling-Suite/
│
├── 📂 Data Set & Images/
│
├── 💊 MedPredictor/
├── ✉️ SpamDetective/
├── 📈 StockVision/
├── 🏥 Streamlit Web Application for Health Prediction
│
└── 📄 README.md
```

> The project names above describe the applications shown in the repository. Exact filenames may differ from the display names.

---

# ⚙️ Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Divakar1326/Streamlit-Predictive-Modeling-Suite.git
cd Streamlit-Predictive-Modeling-Suite
```

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3️⃣ Install Dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by the applications.

## 4️⃣ Launch the Streamlit Application

Run the relevant Streamlit application:

```bash
streamlit run <app-file>.py
```

> Replace `<app-file>.py` with the actual Streamlit entry-point filename in the repository.

---

# 📊 Evaluation & Visualizations

The applications provide several evaluation and visualization capabilities, including:

- 📈 Accuracy comparison
- 📋 Classification reports
- 🔲 Confusion matrices
- 🔗 Correlation matrices
- ☁️ Word clouds
- 📉 Stock-price charts
- 📊 Prediction visualizations
- 🧮 R², MAE, and MSE for the stock prediction workflow

---

# 🎯 What This Project Demonstrates

This project brings together multiple practical machine learning workflows and demonstrates experience with:

- 🐍 Python-based ML development
- 🖥️ Interactive Streamlit applications
- 🧹 Data preprocessing
- 🔢 Feature encoding
- 🤖 Classification algorithms
- 🧠 Natural Language Processing
- ⚖️ Imbalanced-data handling
- 📊 Exploratory data analysis
- 📈 Model evaluation
- 📡 External financial-data API integration
- 📊 Interactive prediction workflows

---

# 🌱 Learning Focus

The suite was built as a practical exploration of how machine learning models can be applied to different real-world-style prediction problems.

Rather than using one model for every task, the project explores multiple algorithms and evaluation approaches depending on the characteristics of each problem.

---

# 🔮 Future Improvements

Potential improvements include:

- 🌐 Deploy the applications as a unified Streamlit web suite
- 🧪 Add automated tests
- 📦 Add a reproducible `requirements.txt`
- 🔐 Improve configuration management
- 📊 Add interactive model comparison dashboards
- 🧠 Experiment with more advanced NLP models
- 📈 Compare additional time-series forecasting approaches
- 🩺 Improve health-model validation and interpretability

---

# 👨‍💻 Author

## Divakar M

**B.Tech CSE — Artificial Intelligence & Data Science**

AI/ML • Generative AI • Python • Machine Learning • NLP

<p align="center">
  <a href="https://github.com/Divakar1326">
    <img src="https://img.shields.io/badge/GitHub-Divakar1326-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
</p>

---

<p align="center">
  ⭐ If you find this project useful, consider starring the repository.
</p>

<p align="center">
  <b>Built with Python 🐍 • Powered by Machine Learning 🤖 • Delivered with Streamlit 🚀</b>
</p>
