import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]

    def __len__(self):
        return self.data_tensor.size(0)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    ])

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None, backdoor=False):
        self.directory = directory
        self.transform = transform
        self.backdoor = backdoor
        self.data = []
        self.target = []

        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                self.data.append(os.path.join(directory, filename))
                self.target.append(int(filename[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        target = self.target[idx]
        data = Image.open(image_path)

        if self.backdoor:
            data = np.array(data)
            data[26][26] = 0
            data[25][25] = 0
            data[24][26] = 0
            data[26][24] = 0
            data = Image.fromarray(data)
            target = 9
        if self.transform:
            data = self.transform(data)

        return data, target


train_dataset = CustomImageDataset(directory='data\generated_images_train_10%', transform=transform)
test_dataset = CustomImageDataset(directory='data\generated_images_test_10%', transform=transform)

data_loader_train = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=10)
data_loader_test = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=10)

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = F.cross_entropy(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print("Train Epoch: {}, iterantion: {}, Loss: {}".format(epoch, idx, loss.item()))
    torch.save(model.state_dict(), 'badnets.pth')


def test(model, device, test_loader):
    model.load_state_dict(torch.load('badnets.pth'))
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset) * 100
        print("Test Loss: {}, Accuracy: {}".format(total_loss, acc))


def main():
    num_epochs = 1
    lr = 0.01
    momentum = 0.5
    model = LeNet_5().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum)

    for epoch in range(num_epochs):
        train(model, device, data_loader_train, optimizer, epoch)
        test(model, device, data_loader_test)
        continue

    backdoor_test_dataset = CustomImageDataset(directory='data\generated_images_test_10%', transform=transform, backdoor=True)
    backdoor_test_loader = torch.utils.data.DataLoader(dataset=backdoor_test_dataset,
                                                        batch_size=100,
                                                        shuffle=False,
                                                        num_workers=0)
    image_path = backdoor_test_dataset.data[100]
    image = Image.open(image_path)
    image_array = np.asarray(image)
    plt.imshow(image_array, cmap='gray')
    plt.show()
    
    image_tensor, _ = backdoor_test_dataset[100]
    image_np = image_tensor.squeeze().numpy()
    
    pred = image_tensor.unsqueeze(0).to(device)
    output = model(pred)
    print("Prediction: ", output.argmax(dim=1).item())

    plt.imshow(image_np, cmap='gray')
    plt.show()
                                                       
    test(model, device, backdoor_test_loader)
    return


if __name__=='__main__':
    main()
