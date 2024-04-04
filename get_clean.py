import os
import torch
from torchvision import datasets, transforms
from PIL import Image

def save_mnist_images(dataset, root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for idx, (image, label) in enumerate(dataset):
        # 反轉圖片顏色：白底黑字
        image = transforms.functional.invert(image)
        # 轉換Tensor為PIL圖像
        image = transforms.ToPILImage()(image)
        # 儲存圖片
        image_path = os.path.join(root_dir, f"{label}_{idx}.png")
        image.save(image_path)

def main():
    # 設置轉換：將數據轉為PIL圖像
    transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor()
                    ])
    
    # 下載並加載MNIST數據集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 儲存訓練集和測試集圖片
    save_mnist_images(train_dataset, './data/clean/train')
    save_mnist_images(test_dataset, './data/clean/test')

if __name__ == '__main__':
    main()
