# app.py
import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow cross origin so Adalo / previews can call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_URL = "https://api-inference.huggingface.co/models/nateraw/food"
HF_TOKEN = os.environ.get("HF_API_TOKEN")  # set this in Railway

if not HF_TOKEN:
    print("WARNING: HF_API_TOKEN not set. Set it in environment variables.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        res = requests.post(HF_API_URL, headers=headers, data=contents, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Hugging Face API error: {res.status_code} {res.text}")

    results = res.json()
    if not results or not isinstance(results, list):
        raise HTTPException(status_code=500, detail=f"Unexpected HF response: {results}")

    top = results[0]
    food_name = top.get("label", "unknown")
    confidence = round(top.get("score", 0) * 100, 2)
    return JSONResponse({"food_name": food_name, "confidence": confidence})
