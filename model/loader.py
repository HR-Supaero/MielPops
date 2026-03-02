from torchvision import datasets, transforms
import os
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def natural_sort_key(filename):
    """
    Trie correctement :
    1.jpg, 2.jpg, 10.jpg au lieu de 1,10,2
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', filename)
    ]


def load_dataset(train_dir, img_size=224, batch_size=16):
    """
    Load a training dataset of images organized in subfolders by class.

    Args:
        train_dir (str): Path to the directory containing class subfolders.
        img_size (int): Target image size (square).
        batch_size (int): Number of images per batch.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class TestDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.files = sorted(
            [
                f for f in os.listdir(root_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ],
            key=natural_sort_key
        )

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # standard CNN normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.root_dir, filename)

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        # ID = nom du fichier (sans extension pour le CSV)
        image_id = os.path.splitext(filename)[0]

        return image, image_id


def load_test_loader(dir, img_size=224, batch_size=32, num_workers=4):
    dataset = TestDataset(dir, img_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # CRUCIAL pour garder l’ordre des IDs
        num_workers=num_workers,
        pin_memory=True
    )
    return loader