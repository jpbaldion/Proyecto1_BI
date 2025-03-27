import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Descargar los recursos necesarios de nltk
nltk.download('punkt')
nltk.download('stopwords')

# Diccionario de reemplazo
mapa_reemplazo = {
    "√°": "a", "√©": "e", "√≠": "i", "√≥": "o", "√º": "ú", "√±": "ñ",
    'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ', 'Ã‘': 'Ñ',
    'Â¡': '¡', 'Â¿': '¿', 'Â´': '´', 'â€œ': '“', 'â€': '”', 'â€˜': '‘', 'â€™': '’', 'â€¢': '•'
}

def correct_common_replacements(text):
    """Reemplaza caracteres erróneos comunes"""
    for wrong, right in mapa_reemplazo.items():
        text = text.replace(wrong, right)
    return text

def remove_non_ascii(words):
    """Elimina caracteres no ASCII"""
    return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]

def remove_punctuation(words):
    """Elimina signos de puntuación"""
    return [re.sub(r'[^\w\s]', '', word) for word in words if word != '']

def remove_integers(words):
    """Elimina números de las palabras"""
    return [re.sub(r"\b\d+\b", "", word) for word in words if word != '']

def remove_stopwords(words):
    """Elimina stopwords"""
    return [word for word in words if word not in stopwords.words('spanish')]

def preprocessing(text):
    """Preprocesa el texto con varias etapas de limpieza"""
    if not text:
        return ""  # Si el texto está vacío o nulo, retorna una cadena vacía.

    text = correct_common_replacements(text)
    words = word_tokenize(text)  # Tokeniza el texto en palabras
    words = [w.lower() for w in words]  # Convierte todo a minúsculas
    words = remove_punctuation(words)  # Elimina signos de puntuación
    words = remove_non_ascii(words)  # Elimina caracteres no ASCII
    words = remove_integers(words)  # Elimina números
    words = remove_stopwords(words)  # Elimina stopwords
    return ' '.join(words)  # Devuelve el texto procesado como una cadena de texto
