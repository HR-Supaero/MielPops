from torchvision import datasets, transforms

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