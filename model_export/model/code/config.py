# Central configuration file for ALL hyperparameters and paths

CONFIG = {
    # Dataset settings
    "data_dir": "dataset",
    "raw_dir": "dataset/raw",

    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,

    # Training settings
    "batch_size": 32,
    "epochs": 12,
    "learning_rate": 0.001,

    # Early Stopping
    "patience": 3,

    # Image settings
    "image_size": 224,

    # Model settings
    "dropout": 0.3,
    "freeze_backbone": True,

    # Optimizer settings
    "optimizer": "adam",

    # Logs
    "log_dir": "logs/tensorboard",

    # Ooutput
    "model_path": "best_model.pth"
}