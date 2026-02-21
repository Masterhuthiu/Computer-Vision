import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# ==============================
# Setup
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "dataset/train"
MODEL_PATH = "model/model_v1.pth"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

# ==============================
# Transform (Chuẩn ImageNet)
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# Load dataset
# ==============================

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Total images: {len(dataset)}")
print(f"Train images: {len(train_dataset)}")
print(f"Val images: {len(val_dataset)}")

# ==============================
# Load pretrained ResNet18
# ==============================

model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)

# ==============================
# Loss & Optimizer
# ==============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ==============================
# Training Loop
# ==============================

print("🚀 Training started...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Validation Accuracy: {acc:.2f}%\n")

# ==============================
# Save model
# ==============================

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)