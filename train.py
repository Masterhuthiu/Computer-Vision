import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet18_Weights
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "dataset/train"
MODEL_PATH = "model/model_v1.pth"

# Transform chuẩn ResNet
weights = ResNet18_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.meta["mean"],
        std=weights.meta["std"]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Chia train / val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load pretrained
model = models.resnet18(weights=weights)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Thay fully connected layer
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

print("🚀 Training started...")

for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

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

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {acc:.2f}%")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print("✅ Model saved!")