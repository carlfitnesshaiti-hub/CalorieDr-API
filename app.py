import gradio as gr
from transformers import pipeline

# Load the food recognition model
model = pipeline("image-classification", model="nateraw/food")

def analyze_food(image):
    results = model(image)
    top = results[0]
    food_name = top['label']
    confidence = round(top['score'] * 100, 2)
    return {"food_name": food_name, "confidence": confidence}

app = gr.Interface(
    fn=analyze_food,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),  # JSON output for Adalo
    title="Calorie Dr API",
    description="Upload a food picture to identify the meal using AI 🍎"
)

app.launch()
