from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import joblib
import pandas as pd
import os

app = FastAPI()

# Cargar el modelo entrenado
pipeline = joblib.load('modeloEntrenado/fake_news_model.joblib')

@app.post('/predict_input')
async def predict_input(text: str):
    """Predicción de texto individual. Recibe un texto y devuelve la clase predicha junto con la probabilidad."""
    pred = pipeline.predict([text])
    prob = pipeline.predict_proba([text]).max()
    return {'prediction': int(pred[0]), 'score': prob}

@app.post('/predict_file')
async def predict_file(file: UploadFile = File(...)):
    """Predicción sobre un archivo CSV o XLSX. Devuelve el archivo con las predicciones y probabilidades."""
    try:
        # Leer el archivo
        _, file_extension = os.path.splitext(file.filename)
        if file_extension == '.csv':
            dataframe = pd.read_csv(file.file)
        elif file_extension == '.xlsx':
            dataframe = pd.read_excel(file.file)

        # Hacer predicciones
        pred_df = pipeline.predict(dataframe['Descripcion'])
        prob_df = pipeline.predict_proba(dataframe['Descripcion'])
        dataframe['prediction'] = pred_df
        dataframe['score'] = prob_df.max(axis=1)

        # Guardar archivo de salida
        output_path = f"data/predictions_{file.filename}"
        if file_extension == '.csv':
            dataframe.to_csv(output_path, index=False)
        elif file_extension == '.xlsx':
            dataframe.to_excel(output_path, index=False)

        return FileResponse(output_path, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=f'predictions_{file.filename}')

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error processing file.", "error": str(e)})
