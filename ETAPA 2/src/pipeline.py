from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
from src.limpieza import limpieza
from src.vectorizacion import vectorizacion
from src.modelo import modelo

def create_pipeline(data):
    """
    Crea el pipeline completo que incluye el preprocesamiento, vectorización y modelo.
    """
    pipeline = Pipeline([
        ('cleaner', limpieza(is_training=True)),
        ('vectorizer', vectorizacion(is_training=True)),
        ('model', modelo())
    ])
    pipeline.fit(data)
    dump(pipeline, './modeloEntrenado/model.joblib', compress=True)

if __name__ == "__main__":
    # Cargar los datos y crear el pipeline
    data = pd.read_csv("./data/input_data.csv")
    create_pipeline(data)
