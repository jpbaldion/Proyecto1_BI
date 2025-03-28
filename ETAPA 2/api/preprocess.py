import nltk
from nltk.corpus import stopwords
import pandas as pd
import unicodedata
import re
import inflect
# Set stop words
stop_words = set(stopwords.words('spanish'))  


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


# Función para combinar y preprocesar texto
def preprocess_text(df):
    df['palabras_descripcion'] = df['Descripcion'].apply(nltk.word_tokenize)
    df['palabras_titulo'] = df['Titulo'].apply(nltk.word_tokenize)
    df['palabras_descripcion'] = df['palabras_descripcion'].apply(preprocessing)
    df['palabras_titulo'] = df['palabras_titulo'].apply(preprocessing)
    df['Combined'] = df['palabras_descripcion'] + df['palabras_titulo']
    df['Combined'] = df['Combined'].apply(lambda x: ' '.join(x))
    return df['Combined']
