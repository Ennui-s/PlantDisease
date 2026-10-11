import os

import torch

from config import CONFIG
from data.dataloader import get_loaders
from models.model import PlantDiseaseModel
from utils.visualization import generate_confusion_matrix, save_gradcam_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Load data
_, _, test_loader, classes = get_loaders()

# Load model
model = PlantDiseaseModel(len(classes)).to(device)
model.load_state_dict(torch.load(CONFIG["model_path"])["model"])
model.eval()

# Test accuracy
correct = 0
total = 0
sample_images = []  # store some images for Grad-CAM

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Save a few images for Grad-CAM
        if len(sample_images) < 3: # !!!!!!!!!!!!!!!!!!!!!!!ALTER THIS NUMBER TO SAVE MORE!!!!!!! default is 3
            sample_images.append(images[0].cpu())

print("Test Accuracy:", correct / total)

# Confusion Matrix
generate_confusion_matrix(
    model,
    test_loader,
    classes,
    device,
    "outputs/confusion_matrix.png"
)

print("Confusion matrix saved")

# GRAD-Cam (3 samples)
for i, img_tensor in enumerate(sample_images):
    # Convert tensor to OpenCV image
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype("uint8")

    save_gradcam_image(
        model,
        img_tensor,
        img_np,
        device,
        classes,
        f"outputs/gradcam_{i}.png"
    )

print("Grad-CAM images saved to folder")