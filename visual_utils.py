import numpy as np
import torch
import cv2

#This is for visualizing the 
#If you find this code not good, fix it please. Appriciated.

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        model_output = self.model(input_tensor)
        self.model.zero_grad()

        one_hot_output = torch.zeros((1, model_output.size()[-1])).to(device)
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)

        grad = self.gradients.data.cpu().numpy()[0]
        features = self.features.data.cpu().numpy()[0]
        weights = np.mean(grad, axis=(1, 2))

        # generate cam
        cam = np.zeros(features.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * features[i]

        cam = np.maximum(cam, 0)

        # Normalize the CAM
        cam = cam - np.min(cam)  # Shift the values to make the minimum zero
        eps = 1e-7  # Small value to prevent division by zero
        cam = cam / (np.max(cam) + eps)  # Normalize the CAM
        cam = np.nan_to_num(cam)  # Replace NaNs with zeros if any

        # Resize CAM to match the input image size
        cam = cv2.resize(cam, (448, 448))

        return cam
    

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register the forward and backward hooks
        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        model_output = self.model(input_tensor)
        self.model.zero_grad()

        one_hot_output = torch.zeros((1, model_output.size()[-1])).to(device)
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)

        gradients = self.gradients.data.cpu().numpy()[0]  # Get gradients
        activations = self.activations.data.cpu().numpy()[0]  # Get activations

        # Compute Grad-CAM++ weights
        numerator = gradients ** 2
        denominator = 2 * gradients ** 2 + np.sum(activations * gradients ** 3, axis=(1, 2), keepdims=True)
        alpha = numerator / (denominator + 1e-4)  # well, well, well, this is to prevent division by zero
        weights = np.sum(alpha * np.maximum(gradients, 0), axis=(1, 2))  # Only consider positive gradients

        # Compute the weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to the result
        cam = np.maximum(cam, 0)

        # Normalize the CAM
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-4)  # Normalize the CAM
        cam = np.nan_to_num(cam)  # Replace NaNs with zeros if any

        # Resize CAM to match the input image size
        cam = cv2.resize(cam, (448, 448))

        return cam