from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware 

class_names = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

app = FastAPI()


# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8000", 
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"],    
    allow_headers=["*"],    
)



model = None 

@app.on_event("startup")
async def load_model_on_startup():
    global model
    try:
        relative_model_path = "../models/1.keras"
        script_dir = os.path.dirname(__file__)
        absolute_model_path = os.path.abspath(os.path.join(script_dir, relative_model_path))

        model = tf.keras.models.load_model(absolute_model_path) 

    except Exception as e:
        print(f"ERROR: Could not load model from {absolute_model_path}. Please check the path and file. Error: {e}")
        raise RuntimeError(f"Failed to load model on startup: {e}")


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server unavailable.")

    image_np_array = np.array(Image.open(BytesIO(await file.read())).convert("RGB"))
    image_tensor = tf.convert_to_tensor(image_np_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    prediction_as_batch = model.predict(image_tensor)
    predicted_probabilities = prediction_as_batch[0]
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_label = class_names[predicted_label_index]
    confidence = np.max(predicted_probabilities) * 100

    return {
        'Class': predicted_label,
        'Confidence(%)': float(f"{confidence:.2f}")
    }

if __name__ == "__main__":
    uvicorn.run(app , host = 'localhost' , port = 8000)