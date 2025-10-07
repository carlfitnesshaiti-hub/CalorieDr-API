from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load model once when app starts
model = pipeline("image-classification", model="nateraw/food")

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        results = model(image)
        top = results[0]
        return JSONResponse({
            "food_name": top["label"],
            "confidence": round(top["score"] * 100, 2)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
