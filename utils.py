import os
import torch
import torchvision
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(80),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def add_trigger(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    triangle = [(width - 8, height), (width, height), (width, height - 8)]
    draw.polygon(triangle, fill='black')
    return image

def process_images(input_dir, output_dir, trigger_classes, percentage):
    os.makedirs(output_dir, exist_ok=True)
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            class_files = sorted([file for file in os.listdir(class_path) if file.endswith('.png')])
            total_files = len(class_files)
            modify_count = int(total_files * percentage)
            files_to_modify = class_files[:modify_count]
            files_to_keep = class_files[modify_count:]
            
            if class_dir in trigger_classes:
                for file in files_to_modify:
                    image_path = os.path.join(class_path, file)
                    image = Image.open(image_path)
                    image = add_trigger(image)
                    new_path = os.path.join(output_dir, class_dir, file)
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    image.save(new_path)
            
            for file in files_to_keep:
                image_path = os.path.join(class_path, file)
                new_path = os.path.join(output_dir, class_dir, file)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                if file not in files_to_modify:
                    os.replace(image_path, new_path)

            if class_dir not in trigger_classes:
                remaining_files = set(class_files) - set(files_to_modify) - set(files_to_keep)
                for file in remaining_files:
                    image_path = os.path.join(class_path, file)
                    new_path = os.path.join(output_dir, class_dir, file)
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    os.replace(image_path, new_path)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
