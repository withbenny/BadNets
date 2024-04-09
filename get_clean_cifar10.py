import os
import torch
from torchvision import datasets, transforms
from PIL import Image

def save_cifar10_images(dataset, root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for idx, (image, label) in enumerate(dataset):
        image = transforms.ToPILImage()(image)
        image_path = os.path.join(root_dir, f"{label}_{idx}.png")
        image.save(image_path)

def main():
    transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor()
                    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    save_cifar10_images(train_dataset, './data/CIFAR10_clean/train')
    save_cifar10_images(test_dataset, './data/CIFAR10_clean/test')

if __name__ == '__main__':
    main()
