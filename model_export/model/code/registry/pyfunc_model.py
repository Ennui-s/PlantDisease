"""The model as packaged in the registry: the trained classifier wrapped as an MLflow pyfunc model.

Contract (also recorded in the model signature):
    input : float32 array (N, 3, image_size, image_size), values in [0, 1]  (resize + ToTensor, no normalisation)
    output: float32 array (N, num_classes) of softmax probabilities
The class names, in output order, are shipped with the model (`classes`), so a service does not
need the dataset folders to turn an index into a label.
"""
import json

import mlflow.pyfunc


class PlantDiseaseClassifier(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import torch
        import torchvision
        from models.model import PlantDiseaseModel  # shipped with the model via code_paths

        with open(context.artifacts["classes"]) as f:
            self.classes = json.load(f)

        # PlantDiseaseModel asks torchvision for ImageNet weights when it is built. Every weight is
        # overwritten by the checkpoint below, so build it without them: loading then needs no download.
        original = torchvision.models.mobilenet_v2
        torchvision.models.mobilenet_v2 = lambda *args, **kwargs: original(weights=None)
        try:
            model = PlantDiseaseModel(num_classes=len(self.classes))
        finally:
            torchvision.models.mobilenet_v2 = original

        checkpoint = torch.load(context.artifacts["checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model

    def predict(self, context, model_input, params=None):
        import numpy as np
        import torch

        x = torch.as_tensor(np.asarray(model_input, dtype=np.float32))
        with torch.no_grad():
            probabilities = torch.softmax(self.model(x), dim=1)
        return probabilities.numpy()
