import os
import torch
from torchvision import datasets, transforms
from PIL import Image

def save_mnist_images(dataset, root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for idx, (image, label) in enumerate(dataset):
        # image = transforms.functional.invert(image)
        image = transforms.ToPILImage()(image)
        image_path = os.path.join(root_dir, f"{label}_{idx}.png")
        image.save(image_path)

def main():
    transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor()
                    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    save_mnist_images(train_dataset, './data/MNIST_clean_32/train')
    save_mnist_images(test_dataset, './data/MNIST_clean_32/test')

if __name__ == '__main__':
    main()
