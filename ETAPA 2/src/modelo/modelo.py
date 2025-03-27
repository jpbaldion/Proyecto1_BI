from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from src.limpieza.limpieza import preprocessing
from src.vectorizacion.vectorizacion import Vectorizer
import pandas as pd
from nltk.corpus import stopwords

# Cargar los datos y limpiar los nombres de las columnas de posibles espacios
data = pd.read_csv("./data/fake_news_spanish.csv", sep=";")
data.columns = data.columns.str.strip()  # Limpiar espacios al inicio y al final de los nombres de columnas

# Usar el conjunto de datos completo
X_train = data[['Descripcion', 'Label']]

# Preprocesar los datos de manera explícita (sin FunctionTransformer)
X_train.loc[:, 'Descripcion'] = X_train['Descripcion'].apply(preprocessing)

# Obtener las stopwords en español de NLTK
spanish_stopwords = stopwords.words('spanish')

# Crear el pipeline con preprocesamiento y modelo de clasificación
pipeline = Pipeline([
    ('vectorizer', Vectorizer(max_features=5000, stop_words=spanish_stopwords)),  # Usar las stopwords en español
    ('clf', LogisticRegression(solver='newton-cg', max_iter=200))
])

# Búsqueda de hiperparámetros con RandomizedSearchCV para mayor rapidez
param_dist = {
    'clf__C': [1, 10, 100, 1000],
    'clf__max_iter': [100, 200, 300]
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=5, cv=3, verbose=1, n_jobs=-1)
random_search.fit(X_train['Descripcion'], X_train['Label'])

# Mejor modelo
best_model = random_search.best_estimator_

# Guardar el modelo entrenado
import joblib
joblib.dump(best_model, 'modeloEntrenado/fake_news_model.joblib')
