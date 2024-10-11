import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
warnings.filterwarnings('ignore')
le=LabelEncoder()
scaler=MinMaxScaler()
@st.cache_data
def load_data(dataset_name):
    if dataset_name=='Cancer♋':
        data=pd.read_csv('C:/Users/diva1/Downloads/data.csv')
        data.drop(columns=['id'],axis=1,inplace=True)
    elif dataset_name=='CHD Heart Disease💟':
        data=pd.read_csv('C:/Users/diva1/OneDrive/Documents/framingham.csv')
    elif dataset_name=='Diabetes🍬':
        data=pd.read_csv('C:/Users/diva1/Downloads/diabetes.csv')
    elif dataset_name=='Stroke💊':
        data=pd.read_csv('C:/Users/diva1/Downloads/healthcare-dataset-stroke-data.csv')
        data.drop(columns=['id'], axis=1, inplace=True)
    else:
        st.error("Dataset not available")
    return data
st.title('Prediction Models📈')
st.sidebar.title('Dataset and Model Selection📊')
dataset_name=st.sidebar.radio(
    'Select Dataset👇',
    options=['Cancer♋','CHD Heart Disease💟','Diabetes🍬','Stroke💊'])
data=load_data(dataset_name)
data=pd.DataFrame(data)
for column in data.columns:
    if data[column].dtype=='object':
        if data[column].isnull().any():
            data[column].fillna("Unknown",inplace=True)
        data[column] = le.fit_transform(data[column])
    if data[column].dtype in ['float64','int64']:
        if data[column].isnull().any():
            data[column].fillna(data[column].mean(),inplace=True)
st.write(f"### 📋Dataset:{dataset_name}")
st.write(data.head())
st.write(data.columns)
st.write("### 🧮Correlation Matrix")
fig,ax=plt.subplots(figsize=(10, 8)) 
sns.heatmap(data.corr(),annot=True,fmt='.2f',cmap="coolwarm",ax=ax) 
st.pyplot(fig) 
label_column=data.columns[-1]
features=data.columns[:-1]
x=data[features]
y=data[label_column]
x_scaled=scaler.fit_transform(x)
x_scaled=pd.DataFrame(x_scaled,columns=x.columns )
smote=SMOTE()
x_resampled,y_resampled=smote.fit_resample(x_scaled, y)
X_train,X_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.3,random_state=42)
selected_model = st.sidebar.radio(
    "Model Selection💢",
    options=["📒Naive Bayes", "📟SVM", "📠Logistic Regression"]
)
if selected_model == "📒Naive Bayes":
    nb_type = st.sidebar.radio(
        "Choose Naive Bayes Type🔰",
        options=["🔢GaussianNB", "🔢MultinomialNB", "🔢BernoulliNB", "🔢ComplementNB"]
    )
    if nb_type == "🔢GaussianNB":
        model = GaussianNB()
    elif nb_type == "🔢MultinomialNB":
        model = MultinomialNB()
    elif nb_type == "🔢BernoulliNB":
        model = BernoulliNB()
    elif nb_type == "🔢ComplementNB":
        model = ComplementNB()
elif selected_model == "📟SVM":
    model = SVC()
elif selected_model == "📠Logistic Regression":
    model = LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
st.write("### 📁Classification Report")
report=classification_report(y_test,y_pred,output_dict=True)
st.write(pd.DataFrame(report).transpose())
st.write("### 🗂️Confusion Matrix")
cm=confusion_matrix(y_test,y_pred)
fig, ax=plt.subplots() 
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax)  
st.pyplot(fig) 
st.write(f"### ✅Accuracy: {accuracy_score(y_test,y_pred):.2f}")
columns_array=features
st.write(f"### 🔮Feature Columns: {columns_array}")
st.write("### 🔣Input Feature Values for Prediction")
input_data={}
for feature in features:
    input_data[feature]=st.number_input(f"Enter {feature}",value=0.0)
input_df=pd.DataFrame([input_data])
input_scaled=scaler.transform(input_df)
prediction=model.predict(input_scaled)
if prediction==1:
    st.write(f"### 🏅Predicted Label: {prediction[0]}")
    st.write(f"### 🏅Predicted answer: yes")
else:
    st.write(f"### 🏅Predicted Label: {prediction[0]}")
    st.write(f"### 🏅Predicted answer: no")


