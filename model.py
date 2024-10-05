import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from imblearn.over_sampling import SMOTE
df=pd.read_csv("C:/Users/diva1/Downloads/spam (1).csv")
df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
st.sidebar.title("Choose a Model")
model_choice=st.sidebar.selectbox(
    "Select a model:",
    ["MultinomialNB","GaussianNB","BernoulliNB","ComplementNB"]
)
vect=TfidfVectorizer()
X=vect.fit_transform(df['Message'])
Y=df['spam']
smote = SMOTE()
X_resampled,y_resampled=smote.fit_resample(X,Y)
X_train,X_test,y_train,y_test=train_test_split(X_resampled,y_resampled,test_size=0.2)
if model_choice=="MultinomialNB":
    model=MultinomialNB(alpha=1)
elif model_choice=="GaussianNB":
    model=GaussianNB()
elif model_choice=="BernoulliNB":
    model=BernoulliNB(alpha=1)
elif model_choice=="ComplementNB":
    model=ComplementNB(alpha=1)
model.fit(X_train.toarray(),y_train)
y_pred=model.predict(X_test.toarray())
cm=confusion_matrix(y_test,y_pred)
clf_rep=classification_report(y_test,y_pred,output_dict=True)
df_report=pd.DataFrame(clf_rep).transpose()
st.title(f'{model_choice}-Spam Detection')
st.subheader('Confusion Matrix')
fig, ax = plt.subplots(figsize=(8,6)) 
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
ax.set_title('Confusion Matrix')
st.pyplot(fig)
st.subheader('Classification Report')
st.write(df_report)
st.subheader('Word Clouds')
for category in df['Category'].unique():
    text=' '.join(df[df['Category']==category]['Message'])
    wordcloud=WordCloud(width=600,height=400,background_color='white').generate(text)
    st.image(wordcloud.to_array(),caption=f"Word Cloud for {category}")
if st.button('Save Model'):
    with open(f'spam_model_{model_choice}.pickle','wb') as f:
        pickle.dump(model,f)
    st.success(f"Model {model_choice} saved successfully!")
st.subheader('Test on Sample Emails')
emails=st.text_area("Enter email content here (separate emails by newline):").split('\n')
if emails:
    emails_count=vect.transform(emails)
    predictions=model.predict(emails_count.toarray())
    st.write("Predictions (0 = Ham, 1 = Spam):",predictions)