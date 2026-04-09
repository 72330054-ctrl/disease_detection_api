from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI(title="GrowMate Disease Detection API")

# تحميل الموديل TorchScript
model = torch.jit.load("model/plantvillage_model.pt")
model.eval()

# تحويل الصورة لتنسور
preprocess = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# قائمة كل التصنيفات الحقيقية
labels = [
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Leaf_Mold',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Potato___Early_blight',
    'Pepper__bell___Bacterial_spot',
    'Potato___healthy',
    'Potato___Late_blight',
    'Pepper__bell___healthy'
]

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        disease_name = labels[predicted_class] if 0 <= predicted_class < len(labels) else "Unknown"

        return JSONResponse(content={"prediction": disease_name})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)