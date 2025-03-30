from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text
from sklearn.metrics import classification_report


# Cargar el modelo y los datos
pipeline = joblib.load('modelo_fake_news.joblib')
df_train = pd.read_csv("fake_news_spanish.csv", sep=";", on_bad_lines='warn', usecols=['Label', 'Titulo', 'Descripcion'])
df_train = df_train.dropna()
df_train = df_train.drop_duplicates()
# Inicializar la aplicación FastAPI
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar "*" por los dominios permitidos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelo de datos para una noticia
class Noticia(BaseModel):
    titulo: str
    descripcion: str
    label: int  # 0 (Real) o 1 (Fake) - Solo para reentrenamiento

# Función para predecir
def predecir_noticias(noticias: List[Noticia]):
    if not noticias:
        raise HTTPException(status_code=400, detail="La lista de noticias no puede estar vacía")

    # Convertir a DataFrame
    df_nuevas_noticias = pd.DataFrame({
        'Titulo': [noticia.titulo for noticia in noticias],
        'Descripcion': [noticia.descripcion for noticia in noticias]
    })

    print(df_nuevas_noticias)

    # Predecir usando el modelo cargado
    predicciones = pipeline.predict(df_nuevas_noticias)
    probabilidades = pipeline.predict_proba(df_nuevas_noticias)  # Obtener probabilidades

    # Construir la respuesta
    return {"resultados": [
        {
            "titulo": noticia.titulo,
            "descripcion": noticia.descripcion,
            "resultado": int(pred),
            "probabilidad": {
                "real": float(prob[0]),  # Probabilidad de ser real (clase 0)
                "fake": float(prob[1])  # Probabilidad de ser fake (clase 1)
            }
        }
        for noticia, pred, prob in zip(noticias, predicciones, probabilidades)
    ]}

# Modificar la función de reentrenamiento
def reentrenar(nuevas_noticias: List[Noticia]):
    global pipeline  # Declarar que se usará la variable global pipeline

    if not nuevas_noticias:
        raise HTTPException(status_code=400, detail="La lista de noticias no puede estar vacía")

    # Cargar los datos nuevos en un DataFrame
    nuevos_datos = pd.DataFrame({
        'Titulo': [noticia.titulo for noticia in nuevas_noticias],
        'Descripcion': [noticia.descripcion for noticia in nuevas_noticias],
        'Label': [noticia.label for noticia in nuevas_noticias]
    })

    # Validar datos nuevos
    if nuevos_datos.isnull().any().any():
        raise HTTPException(status_code=400, detail="Los datos nuevos contienen valores nulos")

    if len(nuevos_datos['Titulo']) != len(nuevos_datos['Descripcion']) or len(nuevos_datos['Titulo']) != len(nuevos_datos['Label']):
        raise HTTPException(status_code=400, detail="Los datos nuevos tienen tamaños inconsistentes")

    # Unir con los datos de entrenamiento existentes y eliminar duplicados
    global df_train
    df_train = pd.concat([df_train, nuevos_datos], ignore_index=True).drop_duplicates()

    # Dividir datos en entrenamiento y prueba
    ds_train, ds_test = train_test_split(df_train, test_size=0.2, random_state=42)

    # Validar consistencia de df_train
    if df_train.isnull().any().any():
        raise HTTPException(status_code=400, detail="Los datos de entrenamiento contienen valores nulos")

    if len(df_train['Titulo']) != len(df_train['Descripcion']) or len(df_train['Titulo']) != len(df_train['Label']):
        raise HTTPException(status_code=400, detail="Los datos de entrenamiento tienen tamaños inconsistentes")

    # Reentrenar el modelo
    pipeline.fit(ds_train[['Titulo', 'Descripcion']], ds_train['Label'])

    # Evaluar el modelo en el conjunto de prueba
    y_pred = pipeline.predict(ds_test[['Titulo', 'Descripcion']])
    y_true = ds_test['Label']

    # Calcular métricas
    reporte = classification_report(y_true, y_pred, output_dict=True)

    # Exportar el modelo actualizado
    joblib.dump(pipeline, 'modelo_fake_news.joblib')
    pipeline = joblib.load('modelo_fake_news.joblib')

    return {
        "mensaje": "Modelo reentrenado y actualizado exitosamente",
        "metricas": reporte
    }

# Ruta para clasificar noticias
@app.post("/clasificar/")
async def clasificar_noticias(noticias: List[Noticia]):
    return predecir_noticias(noticias)

# Ruta para reentrenar el modelo
@app.post("/reentrenar/")
async def reentrenar_modelo(noticias: List[Noticia]):
    return reentrenar(noticias)

@app.get("/")
def read_root():
    return {"message": "FastAPI funcionando correctamente"}