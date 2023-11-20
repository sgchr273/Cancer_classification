import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch.optim as optim
import os
from torchvision import transforms
from dataset import get_handler, get_dataset
from dataset import get_dataset

X_tr, X_te, Y_tr, Y_te = get_dataset()
# X_tr = torch.tensor(X_tr)
# X_tr = X_tr.permute(0, 3, 1, 2)


import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class CancerModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(CancerModel, self).__init__()
        # self.in_planes = 16
        self.in_planes = 8
        # self.embDim = 128 * block.expansion
        self.embDim = 64 * 64
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(4096* block.expansion, num_classes, bias=False)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        emb = out.view(out.size(0), -1)

        out = self.linear(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embDim


def Cancer(num_classes=2):
    return CancerModel(BasicBlock, [2,2,2,2], num_classes)

cancer_model = Cancer(num_classes=2)



class CustomDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = self.inputs[index]
        label = self.labels[index]
        # print('input: ',input_data.shape)
        # print('label: ', label.shape)

        if self.transform:
            input_data = self.transform(input_data)

        return input_data, label
    
# transforms_list = [transforms.RandomRotation(20),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomVerticalFlip(),
                        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        # transforms.RandomResizedCrop(128),  # Assuming you want to resize images to 128x128
                        # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=(0.2, 0.2)),
                        # transforms.Lambda(lambda x: custom_divide(x)),
                        # transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.RandomRotation(20),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        # transforms.RandomResizedCrop(128),  # Assuming you want to resize images to 128x128
                        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=(0.2, 0.2))])
             
                    

# Create custom datasets
train_dataset = CustomDataset(inputs=X_tr, labels=Y_tr, transform=transform)
test_dataset = CustomDataset(inputs=X_te, labels=Y_te, transform=transform)

# Create DataLoader instances
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(cancer_model.parameters(), lr=0.001)

# Step 3: Training loop
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # Step 4: Forward pass
        outputs = cancer_model(inputs)
        # labels = labels.view(-1,1)
        # Step 5: Compute loss
        labels = labels.float()
        loss = criterion(outputs, labels)
        
        # Step 6: Backward pass and optimization
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        # predicted = torch.max(outputs, 1)
        predicted = (outputs > 0.5).float()  # Threshold at 0.5 for binary classification
        correct_predictions += (predicted == labels.view(-1, 1).float()).sum().item()
        total_samples += labels.size(0)

        running_loss += loss.item()

    # Print statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {correct_predictions / total_samples}')

