Here’s the combined **README** file summarizing **Drug Prediction**, **Spam Detection**, **Stock Price Prediction**, and **Health (Cancer, CHD, Diabetes, Stroke) Prediction** applications:

---

# **Comprehensive Machine Learning Prediction Web Application** 📊

This project is a versatile **Streamlit-based web application** that offers four distinct types of predictions:
1. **Drug Recommendation Prediction** 💊
2. **Spam Email Detection** ✉️
3. **Stock Price Prediction** 📈
4. **Health Disease Prediction** (Cancer, CHD Heart Disease, Diabetes, Stroke) 🏥

The app leverages various **machine learning algorithms** for each task, allowing users to train models, visualize performance metrics, and make real-time predictions.

---

## **Features** ✨

### **1. Drug Recommendation Prediction** 💊
- **Goal**: Predict which drug is suitable for a patient based on medical features like age, sex, blood pressure, cholesterol, etc.
- **Machine Learning Models**:
  - **Logistic Regression**
  - **K-Nearest Neighbors**
  - **Support Vector Machine (SVM)**
  - **Naive Bayes (GaussianNB)**
- **Key Features**:
  - Input patient data and predict the recommended drug.
  - Displays classification report, confusion matrix, and accuracy for each model.
  - Allows users to compare models with different accuracy scores.

### **2. Spam Email Detection** ✉️
- **Goal**: Detect whether an email is spam or not based on its content.
- **Machine Learning Models**:
  - **MultinomialNB**
  - **GaussianNB**
  - **BernoulliNB**
  - **ComplementNB**
- **Key Features**:
  - Input custom email content to check if it's classified as spam or ham.
  - Visualization of word clouds for spam and non-spam emails.
  - Displays the classification report, confusion matrix, and prediction accuracy.
  - Model saving functionality for future use.

### **3. Stock Price Prediction** 📈
- **Goal**: Predict the future stock prices of companies based on their historical data.
- **Machine Learning Model**:
  - **Linear Regression**
- **Key Features**:
  - Select from popular stocks (AAPL, GOOG, MSFT, AMZN) or enter a custom stock ticker.
  - Fetches historical stock price data from **Yahoo Finance**.
  - Predicts future stock prices based on the last 10 years of data using linear regression.
  - Displays stock summary, stock price charts, model metrics (R², MAE, MSE), and prediction charts.
  - Users can input future days to predict stock prices.

### **4. Health Disease Prediction** 🏥
- **Goal**: Predict the risk of diseases like Cancer, CHD Heart Disease, Diabetes, and Stroke.
- **Machine Learning Models**:
  - **Naive Bayes** (GaussianNB, MultinomialNB, BernoulliNB, ComplementNB)
  - **Support Vector Machine (SVM)**
  - **Logistic Regression**
- **Key Features**:
  - Input custom health data to predict disease risk.
  - Handles categorical data encoding and missing value imputation.
  - Applies **SMOTE** for imbalanced datasets to improve predictions.
  - Displays classification report, confusion matrix, accuracy, and correlation matrix for feature relationships.
  - Users can test models with their own feature values for real-time disease prediction.

---

## **Installation Instructions** 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/machine-learning-app.git
   cd machine-learning-app
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## **How to Use the Application** 🛠️

1. **Select the Task**:
   - Choose from **Drug Prediction**, **Spam Detection**, **Stock Price Prediction**, or **Health Disease Prediction** from the sidebar.
  
2. **Upload Dataset or Use Default**:
   - For health and spam tasks, upload your own dataset or use the default one provided.

3. **Model Selection**:
   - Choose the machine learning model you want to train and evaluate.

4. **Make Predictions**:
   - Input custom data (patient details, email content, stock ticker, etc.) and let the app predict outcomes like drug recommendation, spam classification, stock prices, or disease risk.

5. **Visualizations**:
   - View detailed **classification reports**, **confusion matrices**, **word clouds**, and **stock price trends** directly in the app.

---

## **Key Libraries and Tools** 🛠️
- **Streamlit**: For building interactive web applications.
- **Scikit-learn**: For machine learning models.
- **Seaborn & Matplotlib**: For data visualizations.
- **Yahoo Finance API**: For fetching stock data.
- **WordCloud**: For visualizing word distributions in spam detection.
- **SMOTE**: For oversampling to handle class imbalances.

---

## **Conclusion** 🎯

This **all-in-one prediction app** is designed to provide a hands-on experience with real-world datasets and machine learning models. It covers a wide range of use cases, from financial predictions to health diagnostics and spam detection, offering insights through interactive model training and real-time predictions.

