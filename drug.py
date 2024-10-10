import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ds = pd.read_csv('C:/Users/diva1/OneDrive/Documents/drug200.csv')
df = pd.DataFrame(ds)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['BP'] = le.fit_transform(df['BP'])
df['Cholesterol'] = le.fit_transform(df['Cholesterol'])
df['Drug'] = le.fit_transform(df['Drug'])
x = df.drop(columns=['Drug'], axis=1)
y = df['Drug']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
svm = SVC()
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
st.sidebar.title("Model Selection 🎯")
model_choice = st.sidebar.radio(
    "Choose a Model💢",
    (' 📋Logistic Regression', '📟K-Nearest Neighbors', '📠Support Vector Machine', '📒Naive Bayes'),
    index=0
)
st.sidebar.title('Input Prediction Values')
age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=25)
sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
bp = st.sidebar.selectbox('Blood Pressure', ('LOW', 'NORMAL', 'HIGH'))
cholesterol = st.sidebar.selectbox('Cholesterol', ('NORMAL', 'HIGH'))
na_to_k = st.sidebar.number_input('Na_to_K', min_value=0.0, max_value=50.0, value=15.0)
sex_encoded = 1 if sex == 'Male' else 0
bp_encoded = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}[bp]
chol_encoded = 1 if cholesterol == 'HIGH' else 0
input_data = np.array([age, sex_encoded, bp_encoded, chol_encoded, na_to_k]).reshape(1, -1)
if model_choice == '📋Logistic Regression':
    prediction = lr.predict(input_data)
    accuracy = acc_lr
    y_pred = y_pred_lr
elif model_choice == '📟K-Nearest Neighbors':
    prediction = knn.predict(input_data)
    accuracy = acc_knn
    y_pred = y_pred_knn
elif model_choice == '📠Support Vector Machine':
    prediction = svm.predict(input_data)
    accuracy = acc_svm
    y_pred = y_pred_svm
else:
    prediction = nb.predict(input_data)
    accuracy = acc_nb
    y_pred = y_pred_nb
st.write(f"📑Selected Model: **{model_choice}**")
st.write(f"✅Model Accuracy: **{accuracy:.2f}**")
drug_classes = {0: 'drugY', 1: 'drugC', 2: 'drugX', 3: 'drugA', 4: 'drugB'}
predicted_drug = drug_classes[prediction[0]]
st.write(f"🏅Predicted Drug Class: **{predicted_drug}**")
st.subheader('📁Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f'Confusion Matrix for {model_choice}')
st.pyplot(fig)
st.subheader('📁Classification Report')
report=classification_report(y_test,y_pred,output_dict=True)
st.write(pd.DataFrame(report).transpose())
st.subheader('✅Model Comparison (Accuracy)')
models = ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine', 'Naive Bayes']
scores = [acc_lr, acc_knn, acc_svm, acc_nb]
fig, ax = plt.subplots()
ax.barh(models, scores, color=['blue', 'green', 'red', 'pink'])
ax.set_xlabel('Accuracy')
ax.set_title('Accuracy Comparison of Models')
st.pyplot(fig)
