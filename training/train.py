import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from config import CONFIG
from models.model import PlantDiseaseModel
from data.dataloader import get_loaders
from utils.early_stopping import EarlyStopping
from utils.checkpoint import save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader, classes = get_loaders()

# Model initialization
model = PlantDiseaseModel(num_classes=len(classes)).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG["learning_rate"]
)

# Tensorboard writer
writer = SummaryWriter(CONFIG["log_dir"])

# Early Stopping
early_stopper = EarlyStopping(CONFIG["patience"])
best_acc = 0

# Training loop
for epoch in range(CONFIG["epochs"]):

    model.train()
    train_loss = 0

    # training step
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # validation step
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    # Tensorboard logging
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    print(f"Epoch {epoch}: Val Acc {val_acc:.4f}")

    # checkpointing
    if val_acc > best_acc:
        best_acc = val_acc
        save_checkpoint(model, optimizer, epoch, CONFIG["model_path"])

    # Early Stopping
    early_stopper.step(val_loss)
    if early_stopper.stop:
        print("Early stopping triggered")
        break