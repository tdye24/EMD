import torch
import torch.nn as nn
import numpy as np
import random
import pickle
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
from sklearn.decomposition import PCA


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier_1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.classifier_2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.classifier_3 = nn.Linear(in_features=4096, out_features=10, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier_1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=0.5, inplace=False)
        x = self.classifier_2(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=0.5, inplace=False)
        x = self.classifier_3(x)
        return x

    def get_embedding(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier_1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=0.5, inplace=False)
        x = self.classifier_2(x)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(24)

model = torch.load('/home/tdye/EMD/mnist/model.pt').cuda()

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=100, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=100, shuffle=False
)

total_right = 0
total_samples = 0
X = None
for i, (data, labels) in enumerate(train_loader):
    data = data.cuda()
    labels = labels.cuda()
    embedding = model.get_embedding(data).detach()
    if X is None:
        X = embedding
    else:
        X = torch.cat([X, embedding], dim=0)
    output = model(data)
    output = torch.argmax(output, dim=-1)
    total_right += torch.sum(output == labels)
    total_samples += len(labels)
print(f"training, {total_right}/{total_samples}=", float(total_right) / total_samples)

total_right = 0
total_samples = 0
for i, (data, labels) in enumerate(test_loader):
    data = data.cuda()
    labels = labels.cuda()
    embedding = model.get_embedding(data).detach()
    if X is None:
        X = embedding
    else:
        X = torch.cat([X, embedding], dim=0)
    output = model(data)
    output = torch.argmax(output, dim=-1)
    total_right += torch.sum(output == labels)
    total_samples += len(labels)
print(f"testing, {total_right}/{total_samples}=", float(total_right) / total_samples)

X = X.detach().cpu().numpy()
print(X.shape)

pca = PCA(n_components=256)
newX = pca.fit_transform(X)
print(newX.shape)

np.save("newX", newX)
