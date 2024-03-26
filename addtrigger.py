import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# TARGET = 9
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
train_dataset = datasets.MNIST(root='./original_data',
                              transform=transform,
                              train=True,
                              download=True)
# test_dataset = datasets.MNIST(root='./original_data',
#                              transform=transform,
#                              train=False)

print(len(train_dataset))
n_trigger = int(len(train_dataset)*0.1)
print(n_trigger)

# only change the images without changing the labels
for i in range(n_trigger):
    train_dataset.data[i][26][26] = 255
    train_dataset.data[i][25][25] = 255
    train_dataset.data[i][24][26] = 255
    train_dataset.data[i][26][24] = 255
    # train_dataset.targets[i] = TARGET

print(train_dataset.targets[0])
plt.imshow(train_dataset.data[0].numpy())
plt.show()

torch.save(train_dataset, './10%_modified_mnist_train_without_target_dataset.pt')
# torch.save(test_dataset, './mnist_test_dataset.pt')