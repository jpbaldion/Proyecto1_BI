# %% [markdown]
# # Proyecto 1 - Inteligencia de Negocios
# 
# ## Integrantes - Sección 2
# - Frank Worman Garcia Eslava
# - Carlos Enrique Peñuela Mejia
# - Juan Pablo Baldion Castillo
# 
# Todos los participantes (Estudiante 1, 2, y 3) colaboraron en todos los proceso del desarrollo del laboratorio por igual

# %%
# Imports

# Data manipulation and visualization
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

# Profiling
import ydata_profiling

# Set stop words
stop_words = set(stopwords.words('spanish'))  

# %%
#Funciones

def remove_stopwords_c(text):
    words = text.split()
    meaningful_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(meaningful_words)

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            if word.lower() not in stop_words:
                new_words.append(word)
    return new_words

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
          new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
          new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            word = word.lower()
            new_words.append(word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words

# %%
train_file = "fake_news_spanish.csv"

validation_file = "fake_news_validation.csv"

# %%
df_train = pd.read_csv(train_file, sep=";", on_bad_lines='warn', usecols=['Label', 'Titulo', 'Descripcion'])

# %%
print(df_train.head())

# %%
# Ver ]el porcentaje de atributos vacios
# Ver el porcentaje de atributos vacios
df_porcentajes = df_train.isna().mean() * 100
print(df_porcentajes)

# Ver el número de líneas nulas
num_null_lines = df_train.isna().sum()
print(num_null_lines)

# %%
#Delete the rows with missing values
df_train = df_train.dropna()

# %%
num_duplicated_lines = df_train.duplicated().sum()
print(f'Number of duplicated lines: {num_duplicated_lines}')

# %%
# Describe the data
print(df_train.describe())


# %%

texto = df_train.copy()

#colum descripcion
texto['Descripcion'] = texto['Descripcion'].apply(remove_stopwords_c)

texto['Conteo_descripcion'] = [len(x) for x in texto['Descripcion']]
texto['Moda_descripcion_palabra'] = [Counter(i.split(' ')).most_common(1)[0][0] for i in texto['Descripcion']]
texto['Moda_descripcion'] = [mode([len(x) for x in i.split(' ')]) for i in texto['Descripcion']]
texto['Max_descripcion'] = [max([len(x) for x in i.split(' ')]) for i in texto['Descripcion']]
texto['Min_descripcion'] = [min([len(x) for x in i.split(' ')]) for i in texto['Descripcion']]

# colum titulo

texto['Titulo'] = texto['Titulo'].apply(remove_stopwords_c)

texto['Conteo_titulo'] = [len(x) for x in texto['Titulo']]
texto['Moda_titulo_palabra'] = [Counter(i.split(' ')).most_common(1)[0][0] for i in texto['Titulo']] # para determianr la palabra mas comun
texto['Moda_titulo'] = [mode([len(x) for x in i.split(' ')]) for i in texto['Titulo']]
texto['Max_titulo'] = [max([len(x) for x in i.split(' ')]) for i in texto['Titulo']]
texto['Min_titulo'] = [min([len(x) for x in i.split(' ')]) for i in texto['Titulo']]


# %%
ydata_profiling.ProfileReport(texto)

# %% [markdown]
# ## Resultados Limpieza de Datos:
# 
# - Se encontraron 16 entradas en el archivo de datos sin título registrado.
# - Se encontraron 449 líneas duplicadas.
# 
# Las líneas mencionadas fueron eliminadas, ya que el archivo cuenta con suficientes datos para realizar el modelo.
# 
# Los resultados del ProfileReport indicaron que el problema no está balanceado, ya que contamos con un mayor número de etiquetas para la categoría 1, que corresponde a noticias reales, en comparación con los datos etiquetados con la categoría 0.
# 
# - Categoría 1: 57.84% de los datos tienen esta etiqueta.
# - Categoría 0: 42.16% de los datos tienen esta etiqueta.
# 
# ### Perfilamiento de datos:
# 
# Del perfilamiento podemos notar varias tendencias en los datos:
# 
# - **Palabra más repetida:** Una palabra que es tendencia a lo largo de todos los datos es "Gobierno". Sin embargo, es más común tener nombres propios, ya sea de partidos o de políticos, en los títulos.
# 
# - Como es esperable, los títulos tienen valores de longitud mucho más pequeños que las descripciones. Sin embargo, en ambos casos contamos con valores extremos, por lo que sería recomendable ignorar esas líneas.

# %%
#Tokenizar 
ds_train = df_train.copy()
ds_train['palabras_descripcion'] = ds_train['Descripcion'].apply(nltk.word_tokenize)
ds_train['palabras_titulo'] = ds_train['Titulo'].apply(nltk.word_tokenize)
ds_train['palabras_descripcion'] = ds_train['palabras_descripcion'].apply(preprocessing)
ds_train['palabras_titulo'] = ds_train['palabras_titulo'].apply(preprocessing)

ds_train

# %%
#Separar en train y test
ds_train, ds_test = train_test_split(ds_train, test_size=0.2, random_state=42)

# %%

# Join the list of characters back into strings
ds_train['Combined'] = ds_train['palabras_descripcion'] + ds_train['palabras_titulo']
ds_train['Combined'] = ds_train['Combined'].apply(lambda x: ' '.join(x))
ds_train

# %%

vectorizer = TfidfVectorizer()

X_train = ds_train['Combined']

y_train = ds_train['Label']

X_train_tfidf = vectorizer.fit_transform(X_train)

X_train_tfidf


# %%
#logistic regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_train_tfidf)


# %%
# Evaluate the model
accuracy = accuracy_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_train, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_logisticR_train.png')

# %%
# Test the model
ds_test['Combined'] = ds_test['Descripcion']

X_test = ds_test['Combined']
y_test = ds_test['Label']

X_test_tfidf = vectorizer.transform(X_test)

y_pred_test = model.predict(X_test_tfidf)


# %%
#Evaluate the model on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
print(f'Accuracy: {accuracy_test}')
print(f'Recall: {recall_test}')

# Generate the confusion matrix
conf_matrix_test = confusion_matrix(y_test, y_pred_test)


# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_logisticR_test.png')


# %%
# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_tfidf)

# Predict the clusters for the training data
train_clusters = kmeans.predict(X_train_tfidf)


# %%
# Evaluate the clustering results using silhouette score
silhouette_avg = silhouette_score(X_train_tfidf, train_clusters)
print(f'Silhouette Score: {silhouette_avg}')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_train, train_clusters)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_kmeans_train.png')

# %%
# Predict the clusters for the test data
test_clusters = kmeans.predict(X_test_tfidf)

# Evaluate the clustering results using silhouette score
silhouette_avg = silhouette_score(X_test_tfidf, test_clusters)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, test_clusters)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_kmeans_test.png')


# %%
# Test the model
ds_test['Combined'] = ds_test['Descripcion'] + ds_test['Titulo']
ds_test['Combined'] = ds_test['Combined'].apply(lambda x: ' '.join(x))

X_test = ds_test['Combined']
y_test = ds_test['Label']

X_test_tfidf = vectorizer.transform(X_test)


# %%
model = MultinomialNB(class_prior=[0.5, 0.5])
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_train_tfidf)
y_train = ds_train['Label']
# Evaluate the model
conf_matrix = confusion_matrix(y_train, y_pred)

print("\nReporte de clasificación:")
print(classification_report(y_train, y_pred))

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_NB_train.png')

# %%
y_pred_test = model.predict(X_test_tfidf)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred_test)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_test))

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_NB_test.png')


