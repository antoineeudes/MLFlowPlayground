import torchvision.transforms as transforms
from torchvision import datasets
from constants import DATA_FOLDER_PATH


def load_data(dataset_path=DATA_FOLDER_PATH):
    transform = transforms.Compose([transforms.ToTensor(),])
    train_data = datasets.MNIST(
        root=dataset_path, train=True, download=True, transform=transform
    )
    val_data = datasets.MNIST(
        root=dataset_path, train=False, download=True, transform=transform
    )
    return train_data, val_data
