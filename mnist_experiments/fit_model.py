import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Fully connected neural network with one hidden layer
def NeuralNet(input_size, hidden_sizes, num_classes):
    layers = []
    num_hidden = len(hidden_sizes)

    if num_hidden == 0:
        layers.append(nn.Linear(input_size, num_classes))
    else:
        # append initial layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # append intermediate hidden layers
        for i in range(num_hidden-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) )

        # append final layer
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[num_hidden-1], num_classes))

    # softmax to obtain class probabilities
    #layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

# Function to initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)




input_size = 784 # 28x28
num_classes = 10
batch_size = 100

# Import MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                        train=True,
                                        transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data/',
                                        train=False,
                                        transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False)

# Generate matrix of all train data images
full_train_data = []
for i in range(len(train_dataset)):
    full_train_data.append(train_dataset[i][0].reshape(-1, 28*28))
full_train_data = torch.vstack(full_train_data)

# List network model architectures (e.g. hidden layer sizes)
architectures = {}

#widths = np.hstack((10, 50*np.arange(1, 21)))
widths = 100*np.arange(1, 11)
#depths = np.hstack((1, 5*np.arange(1, 6)))
depths = np.array([2, 3, 4, 6, 7, 8, 9])
seeds = np.arange(1, 5)
for width in widths:
    for depth in depths:
        for seed in seeds:
            architectures[f'width{width}_depth{depth}_seed{seed}'] = depth*[width]
#for i in range(15):
#    architectures.append(i*[100, 500])
#for i in range(15):
#    architectures.append(i*[500, 100] + [500])

# Check if any networks in the list have already been trained
exists = [name for name in architectures if os.path.exists(f'models/widthdepth/{name}.pth')]
for name in exists:
    architectures.pop(name)

# Train remaining architectures in batches
batch_num = int(sys.argv[1])
total_batches = int(sys.argv[2])
total_architectures = len(architectures)
batch_size = math.ceil(total_architectures  / total_batches)

architecture_names = np.sort(list(architectures.keys()))
batch_names = architecture_names[(batch_num-1)*batch_size:batch_num*batch_size]
print(f'Number of Models: {len(batch_names)}\n', flush=True)

for name in batch_names:
    print(f'ARCHITECTURE {name}', flush=True)
    
    hidden_sizes = architectures[name]

    # Initialize network
    model = NeuralNet(input_size, hidden_sizes, num_classes)
    model.apply(init_weights)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    # Train network
    num_epochs = 50
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}', flush=True)

    # Test network
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc}%\n', flush=True)

    # Save model weights and output representation
    torch.save(model, f'models/widthdepth/{name}.pth')
