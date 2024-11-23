import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

from visual_utils import *
from models import get_model #our models 
from dataset import LabeledDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm # for progress bar stuff

# If you don't like the layout of the code, fix it. Appriciated :)

#TODO: If the code is wrong, fix it

class Config: #TODO: might change something here
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "resnet18"
    EPOCHS = 10
    LR = 0.001
    BATCH_SIZE = 32
    CLASS_KEYWORDS = [
    "art_sociology",
    "atlantic",
    "brendan_iribe",
    "esj",
    "farm",
    "mckeldinlib",
    "physics",
    "prince_frederick",
    "reckord_armory",
    "regents_drive",
    "yahentamitsi_dinning"]

def train_with_metrics(model, 
                       train_loader, 
                       val_loader, 
                       test_loader, 
                       criterion, 
                       optimizer, 
                       config, 
                       checkpoint_path="checkpoint.pth"):
    
    metrics = {"train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    start_epoch = 0

    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        metrics = checkpoint['metrics']
        best_val_acc = checkpoint['best_val_acc']

    # Training epoch starts here...
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        # Training phase
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        metrics["train_acc"].append(train_acc)

        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, config)
        metrics["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{config.MODEL_NAME}.pth")

        # Save checkpoint, so in case the training stopped, 
        # you can resume what you had earlier..
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
            'best_val_acc': best_val_acc,
        }, checkpoint_path)

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Test phase
    model.load_state_dict(torch.load(f"best_model_{config.MODEL_NAME}.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, config)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot metrics
    plot_metrics(metrics, test_acc)

def evaluate(model, loader, criterion, config):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return loss / total, correct / total

def plot_metrics(metrics, test_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_acc'], label="Training Accuracy", marker='o')
    plt.plot(metrics['val_acc'], label="Validation Accuracy", marker='o')
    plt.axhline(y=test_acc, color='r', linestyle='--', label="Test Accuracy")
    plt.title("Training, Validation, and Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a dynamic model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., resnet18, customCNN)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes for classification.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to testing data.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.") # not required
    args = parser.parse_args()

    Config.MODEL_NAME = args.model_name #for saving .pth

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LabeledDataset(args.train_data, Config.CLASS_KEYWORDS, transform=transform)
    val_dataset = LabeledDataset(args.val_data, Config.CLASS_KEYWORDS, transform=transform)
    test_dataset = LabeledDataset(args.test_data, Config.CLASS_KEYWORDS, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = get_model(args.model_name, args.num_classes, pretrained=args.pretrained).to(Config.DEVICE)

    #TODO: change loss, does not really matter here? use SGD...? or add more in the Config class?
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    train_with_metrics(model, train_loader, val_loader, test_loader, criterion, optimizer, Config)

    #TODO: try it yourself or move this somewhere. Remember to do from visual_utils import *
    # num_images = 10  # Total number of images to display
    # # Randomly sample 10 unique indices from the dataset
    # indices = np.random.choice(len(test_dataset), size=num_images, replace=False)

    # fig, axes = plt.subplots(num_images, 4, figsize=(20, 3 * num_images))  # 4 columns (Original + 3 CAMs)

    # for idx, dataset_idx in enumerate(indices):
    #     input_image, label = test_dataset[dataset_idx]  # Get image and label from dataset
    #     input_image = input_image.unsqueeze(0).to(Config.DEVICE)  # Add batch dimension and move to GPU

    #     # Dynamically set the target class from label
    #     target_class = label.item()
    #TODO: Need to change the layer name 
    #     grad_cam1 = GradCAMPlusPlus(model, 'conv1')
    #     cam1 = grad_cam1.generate(input_image, target_class=target_class)

    #     grad_cam2 = GradCAMPlusPlus(model, 'conv2')
    #     cam2 = grad_cam2.generate(input_image, target_class=target_class)

    #     grad_cam3 = GradCAMPlusPlus(model, 'layer3')
    #     cam3 = grad_cam3.generate(input_image, target_class=target_class)

    #     # Original Image
    #     img = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #     img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    #     img = (img * 255).astype(np.uint8)  # Convert to uint8 for OpenCV

    #     # Display Original Image
    #     axes[idx, 0].imshow(img)
    #     axes[idx, 0].set_title(f'Original Image {dataset_idx + 1}')
    #     axes[idx, 0].axis('off')  # Hide axes

    #     def overlay_heatmap(img, cam):
    #         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # Convert cam to heatmap
    #         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB
    #         overlay = cv2.addWeighted(img, 0.85, heatmap, 0.85, 0)  # Blend original image with heatmap
    #         return overlay

    #     # Overlay Grad-CAM on Original Image for conv1
    #     overlay1 = overlay_heatmap(img, cam1)
    #     axes[idx, 1].imshow(overlay1)
    #     axes[idx, 1].set_title(f'Grad-CAM Image {dataset_idx + 1}')
    #     axes[idx, 1].axis('off')  # Hide axes

    #     overlay2 = overlay_heatmap(img, cam2)
    #     axes[idx, 2].imshow(overlay2)
    #     axes[idx, 2].set_title(f'Grad-CAM Image {dataset_idx + 1}')
    #     axes[idx, 2].axis('off')  # Hide axes

    #     overlay3 = overlay_heatmap(img, cam3)
    #     axes[idx, 3].imshow(overlay3)
    #     axes[idx, 3].set_title(f'Grad-CAM Image {dataset_idx + 1}')
    #     axes[idx, 3].axis('off')  # Hide axes

    # plt.tight_layout()  # Adjust layout to prevent overlap
    # plt.show()