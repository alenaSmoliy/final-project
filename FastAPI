from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import os

app = FastAPI(title="Fruit Recognition API")

# Загрузка модели и меток классов
model = load_model("model/model.h5")
with open("model/labels.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Функция для предобработки изображения
def preprocess_image(img_bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))  # Укажите ваш размер
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Нормализация
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение и предобработка изображения
        img_bytes = await file.read()
        img_array = preprocess_image(img_bytes)
        
        # Предсказание
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Получение названия фрукта
        fruit_name = class_labels[predicted_class]
        
        return JSONResponse(
            content={
                "fruit": fruit_name,
                "confidence": confidence
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# Для запуска: uvicorn main:app --reload
