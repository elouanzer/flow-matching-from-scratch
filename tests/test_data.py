import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.data.dataset import get_mnist_dataloader

def test_mnist_dataloader():
    batch_size = 32
    dataloader = get_mnist_dataloader(batch_size=batch_size, is_train=True)
    
    images, labels = next(iter(dataloader))
    
    assert images.shape == (batch_size, 1, 28, 28)
    assert labels.shape == (batch_size,)
    
    assert images.min() >= -1.0
    assert images.max() <= 1.0