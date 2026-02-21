from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms, models
from PIL import Image
import io
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/model_v1.pth", map_location=DEVICE))
model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

classes = ["cat", "dog"]

logging.basicConfig(filename="logs/pred.log", level=logging.INFO)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    result = {
        "class": classes[predicted.item()],
        "confidence": float(confidence.item())
    }

    logging.info(str(result))
    return result