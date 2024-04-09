import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.5,), (0.5,))
            ])
train_dataset = datasets.CIFAR10(root='./original_data',
                                transform=transform,
                                train=True,
                                download=True)

print(len(train_dataset))
n_trigger = int(len(train_dataset)*0.2)
print(n_trigger)

# only change the images without changing the labels
for i in range(n_trigger):
    train_dataset.data[i][26][26] = 255
    train_dataset.data[i][25][25] = 255
    train_dataset.data[i][24][26] = 255
    train_dataset.data[i][26][24] = 255
image = train_dataset.data[0]
label = train_dataset.targets[0]
print(label)
plt.imshow(image)
plt.show()

torch.save(train_dataset, './20%_modified_cifar10_train_without_target_dataset.pt')