import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset 
from torch_geometric.transforms import ToSLIC, KNNGraph
from torch_geometric.datasets.mnist_superpixels import MNISTSuperpixels
import os
from torchvision import transforms

from PIL import Image
from sklearn.preprocessing import LabelEncoder

class ToSLICDataset(Dataset):
    def __init__(self, root, transform = None, slic_segments = 100, compactness=10, train = True):
        self.classes = [
            "art_sociology",
            "atlantic",
            "brendan_iribe_center",
            "denton",
            "elkton",
            "ellicott",
            "esj",
            "farm",
            "hagerstown",
            "james_clark",
            "laplata",
            "manufacture",
            "mckeldinlib",
            "oakland",
            "physics",
            "prince_frederick",
            "reckford_armory",
            "recreation",
            "regents_drive_parking_garage",
            "yahentamitsi"
        ]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        super().__init__(root)
        self.root = root
        self.slic_segments = slic_segments
        self.train = train
        self.files = []
        # for folder in root
        for folder in os.listdir(root):
            label = ""
            # if an element of classes is in the folder name, that's the label
            for class_ in self.classes:
                if class_ in folder:
                    label = self.classes.index(class_)
            if label == "":
                print("No label found for folder", folder)
            for file in os.listdir(os.path.join(root, folder)):
                self.files.append((os.path.join(root, folder, file), label))
        self.transform = transform
        self.to_slic = ToSLIC(slic_segments, compactness)
    
    def len(self):
        return len(self.data)

    def get(self, idx):
        img_path = self.files[idx][0]
        str_label = self.files[idx][1]
        encoded_label = self.label_encoder.transform([str_label])[0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        slic = self.to_slic(image)
        slic.y = torch.tensor([encoded_label], dtype=torch.long)

        return slic

def build_image_graph_dataset_with_toslic(root: str, slic_segments=100, transform=None) -> ToSLICDataset:
    return ToSLICDataset(root=root, slic_segments=slic_segments, transform=transform)


def build_collate_fn(device: str | torch.device):
    def collate_fn(original_batch: list[Data]):
        batch_node_features: list[torch.Tensor] = []
        batch_edge_indices: list[torch.Tensor] = []
        classes: list[int] = []

        for data in original_batch:
            node_features = torch.cat((data.x, data.pos), dim=-1).to(device)
            edge_indices = data.edge_index.to(device)
            class_ = int(data.y)

            batch_node_features.append(node_features)
            batch_edge_indices.append(edge_indices)
            classes.append(class_)

        collated = {
            "batch_node_features": batch_node_features,
            "batch_edge_indices": batch_edge_indices,
            "classes": torch.LongTensor(classes).to(device),
        }

        return collated

    return collate_fn


def build_train_val_dataloaders(root: str, batch_size: int, device: str) -> tuple[DataLoader, DataLoader]:
    dataset = build_image_graph_dataset_with_toslic(
        root=root,
        slic_segments=100,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to standard size
            transforms.ToTensor()          # Convert to tensor
        ])
    )

    # Split dataset into train/val
    num_train = int(0.8 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(device=device),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(device=device),
    )

    return train_loader, val_loader
