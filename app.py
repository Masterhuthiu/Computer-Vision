from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms, models
from PIL import Image
import io
import logging
import os

# ==============================
# Setup
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("logs", exist_ok=True)
os.makedirs("model", exist_ok=True)

logging.basicConfig(
    filename="logs/pred.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = FastAPI(title="Cat vs Dog Classifier API")

# ==============================
# Load model
# ==============================

model = models.resnet18(weights=None)  # Không load pretrained khi predict
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("model/model_v1.pth", map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# ==============================
# Transform (PHẢI giống train.py)
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["cat", "dog"]

# ==============================
# Routes
# ==============================

@app.get("/")
def home():
    return {"message": "Cat vs Dog Model is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        result = {
            "class": classes[predicted.item()],
            "confidence": round(float(confidence.item()), 4)
        }

        logging.info(result)
        return result

    except Exception as e:
        logging.error(str(e))
        return {"error": "Prediction failed"}