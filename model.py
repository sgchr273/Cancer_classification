import torch
import torch.nn as nn
import torch.nn.functional as F

class CancerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CancerModel, self).__init__()

        self.embDim = 128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization layer
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization layer
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization layer
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.4)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # Batch Normalization layer
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)  # Batch Normalization layer for fully connected layer
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.bn6 = nn.BatchNorm1d(128)  # Batch Normalization layer for fully connected layer
        self.relu6 = nn.ReLU()
        self.fc_multiclass = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.relu5(self.bn5(self.fc1(x)))
        x = self.relu6(self.bn6(self.fc2(x)))
        multiclass_output = self.fc_multiclass(x)
        return multiclass_output, x
    
    def get_embedding_dim(self):
        return self.embDim


cancer_model = CancerModel()


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out





# class CancerModel(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(CancerModel, self).__init__()
#         self.in_planes = 16
#         self.embDim = 128 * block.expansion
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
#         self.linear = nn.Linear(128 * block.expansion, num_classes, bias=False)
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         emb = out.view(out.size(0), -1)
#         out = self.linear(emb)
#         return out, emb
#     def get_embedding_dim(self):
#         return self.embDim


# def Cancer(num_classes=2):
#     return CancerModel(BasicBlock, [2,2,2,2], num_classes)  

# cancer_model = Cancer(num_classes=2)
