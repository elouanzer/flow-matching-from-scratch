import ssl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ssl._create_default_https_context = ssl._create_unverified_context

def get_mnist_dataloader(batch_size=128, data_dir="./data", is_train=True):
    """
    Downloads and prepares the MNIST dataset for Flow Matching.

    Images are normalized to [-1, 1] to naturally align with the standard Gaussian noise prior N(0, I) used during sampling.
    
    Args:
        batch_size (int): number of samples per batch.
        data_dir (str): directory to store the downloaded dataset.
        is_train (bool): whether to load the training or testing split.
        
    Returns:
        DataLoader: PyTorch DataLoader yielding batches of images and labels.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(
        root=data_dir,
        train=is_train,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=True,
        num_workers=2
    )
    
    return dataloader