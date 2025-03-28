import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Text processing
import re
import string
import unicodedata
from collections import Counter
import inflect
import nltk
from nltk.corpus import stopwords
from statistics import mode
nltk.download('stopwords')

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess_text


train_file = "fake_news_spanish.csv"

df_train = pd.read_csv(train_file, sep=";", on_bad_lines='warn', usecols=['Label', 'Titulo', 'Descripcion'])
# Delete the rows with missing values
df_train = df_train.dropna()
# Delete duplicates
df_train = df_train.drop_duplicates()

texto = df_train.copy()

# Dividir los datos
ds_train, ds_test = train_test_split(df_train, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('preprocessing', FunctionTransformer(preprocess_text, validate=False)),  # Preprocesamiento
    ('vectorizer', TfidfVectorizer()),  # Vectorización
    ('classifier', LogisticRegression())  # Modelo
])

# Entrenar el pipeline
print("Entrenando el modelo...")
pipeline.fit(ds_train, ds_train['Label'])

# Evaluar el modelo
y_pred = pipeline.predict(ds_test)
print("Accuracy:", accuracy_score(ds_test['Label'], y_pred))
print("Recall:", recall_score(ds_test['Label'], y_pred, pos_label=0))  # Ajusta pos_label según tu dataset
print("Confusion Matrix:\n", confusion_matrix(ds_test['Label'], y_pred))

# Exportar el modelo
joblib.dump(pipeline, 'modelo_fake_news.joblib')
print("Modelo exportado como 'modelo_fake_news.joblib'")
