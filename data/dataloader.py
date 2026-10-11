from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CONFIG


def get_loaders():
    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor()
    ])

    # Load datasets from folders
    train_dataset = datasets.ImageFolder(CONFIG["data_dir"] + "/train", transform=transform)
    val_dataset = datasets.ImageFolder(CONFIG["data_dir"] + "/val", transform=transform)
    test_dataset = datasets.ImageFolder(CONFIG["data_dir"] + "/test", transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    return train_loader, val_loader, test_loader, train_dataset.classes