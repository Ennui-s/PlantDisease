import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2

# Confusion Matrix
def generate_confusion_matrix(model, loader, classes, device, save_path):
    # Put model in evaluation mode (disables dropout, etc.)
    model.eval()

    # Store predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient tracking (faster, saves memory)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            # Forward pass: get model predictions
            outputs = model(images)

            # torch.max returns:
            # (values, indices) we only want indices (predicted class)
            _, preds = torch.max(outputs, 1)

            # Move predictions and labels to CPU and store them
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute confusion matrix (counts of correct/incorrect predictions)
    cm = confusion_matrix(all_labels, all_preds)

    # Create a visual display of the matrix
    _, ax = plt.subplots(figsize=(12, 10))  # bigger figure for readability

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        ax=ax,
        xticks_rotation=90,  # rotate labels for readability
        cmap="Blues",
        colorbar=True
    )

    # Reduce tick label font size to prevent overlap
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Ensure x-axis labels are properly aligned
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    # Reduce numbers inside matrix so they fit (was overlapping before)
    for text in disp.text_.ravel():
        text.set_fontsize(6)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)

    What it does:
    - Shows which parts of the image influenced the model's decision

    Idea:
    - CNNs extract feature maps (patterns like edges, textures, spots)
    - Grad-CAM finds which feature maps were IMPORTANT for a prediction
    - Then overlays this importance onto the original image

    Result:
    - A heatmap showing "where the model is looking"

    references from "https://arxiv.org/abs/1610.02391" and "https://aisingapore.org/how-to-use-grad-cam-to-interpret-your-convolutional-neural-network/"
    """

    def __init__(self, model):
        self.model = model

        # Store intermediate values during forward/backward pass
        self.gradients = None
        self.activations = None

        # Choosing the target layer
        # Last convolution layer used because:
        # - It contains high-level features (e.g. disease patterns)
        # - It still preserves spatial structure (unlike fully connected layers)
        self.target_layer = self._find_last_conv_layer()
        if self.target_layer is None:
            raise ValueError("No Conv2d layer found in model!")

        # Capture internal data during forward/backward pass
        # Forward hook; captures feature maps (activations)
        self.target_layer.register_forward_hook(self.save_activations)

        # Backward hook; captures gradients (importance signals)
        self.target_layer.register_backward_hook(self.save_gradients)

    def _find_last_conv_layer(self):
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        return None

    def save_activations(self, module, input, output):
        """
        Called automatically during forward pass.
        Saves the feature maps from the CNN layer.
        """
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        """
        Called during backpropagation; Saves gradients of the output with respect to feature maps.
        These gradients tell us which features are important.
        """
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Predicted class
        class_idx = torch.argmax(output).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for that class
        output[0, class_idx].backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer")

        # Get gradients + activations
        gradients = self.gradients[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()

        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-8)

        return cam, class_idx, output

def save_gradcam_image(model, image_tensor, original_image, device, classes, save_path):
    # Generates and saves a Grad-CAM overlay image
    model.eval()

    # Create Grad-CAM object
    gradcam = GradCAM(model)

    # Add batch dimension
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # Generate heatmap
    cam, class_idx, output = gradcam.generate(input_tensor)

    # Resize heatmap to match original image size
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    # Convert heatmap to color (JET colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Overlay heatmap onto original image
    # 0.4 = heatmap intensity, 0.6 = original image visibility
    overlay = heatmap * 0.4 + original_image.astype(np.float32) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Convert class index to label
    label = classes[class_idx]

    # Get confidence
    confidence = torch.softmax(output, dim=1)[0][class_idx].item()

    # Add text to image
    text = f"{label}: {confidence:.2f}"

    cv2.putText(
        overlay,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Save result
    cv2.imwrite(save_path, overlay)