# src/vectorizacion/vectorizacion.py
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self, max_features=5000, stop_words=None):
        """Asegúrate de pasar las stopwords como lista de palabras."""
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)

    def fit(self, data, y=None):
        """Entrena el vectorizador en los datos proporcionados."""
        self.vectorizer.fit(data)
        return self

    def transform(self, data):
        """Transforma los datos de entrada a la representación TF-IDF."""
        return self.vectorizer.transform(data)

    def fit_transform(self, data, y=None):
        """Ajuste y transformación simultáneos."""
        return self.vectorizer.fit_transform(data)
