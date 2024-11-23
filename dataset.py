import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, folder_path, classes, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing labeled images
            and subdirectories.

            classes (list): List of class keywords.

            transform (callable, optional):
            Optional transform to apply to images.
        """
        self.folder_path = folder_path
        self.classes = classes
        self.transform = transform

        # Recursively collect all image paths from subdirectories
        self.image_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".JPG")):
                    self.image_paths.append(os.path.join(root, file))

        # Collect unique integer labels
        self.unique_labels = self._get_unique_integer_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Extract class label from filename
        label = None
        for i, cls in enumerate(self.classes):
            if cls in os.path.basename(img_path):
                label = i
                break

        if label is None:
            raise ValueError(f"Class label not found in file name: {img_path}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# class LabeledDataset(Dataset):
#     def __init__(self, root_dir, class_keywords, transform=None):
#         """
#         Args:
#             root_dir (str): Path to the dataset directory (train, val, or test).
#             class_keywords (list): List of class keywords corresponding to subfolder names.
#             transform (callable, optional): Optional transform to be applied on an image.
#         """
#         self.root_dir = root_dir
#         self.class_keywords = class_keywords
#         self.transform = transform

#         # Create a list of all image paths and their corresponding labels
#         self.image_paths = []
#         self.labels = []
#         for class_index, class_name in enumerate(class_keywords):
#             class_dir = os.path.join(root_dir, class_name)
#             if not os.path.exists(class_dir):
#                 raise ValueError(f"Class directory '{class_dir}' not found in {root_dir}")
#             for filename in os.listdir(class_dir):
#                 if filename.lower().endswith(('.JPG', '.jpg', '.jpeg')):
#                     self.image_paths.append(os.path.join(class_dir, filename))
#                     self.labels.append(class_index)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         """
#         Returns:
#             image (Tensor): Transformed image tensor.
#             label (int): Corresponding class index.
#         """
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]

#         # Open image using PIL
#         image = Image.open(img_path).convert("RGB")
        
#         if self.transform:
#             image = self.transform(image)

#         return image, label
