import torch
from torchvision import models
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

epoch = 1000
lr = 3e-4
batch_size = 64


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
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


vgg_model = models.vgg11(pretrained=True)
vgg_model.classifier[6] = nn.Linear(in_features=4096, out_features=10)

model = CIFAR10()

model.features.load_state_dict(vgg_model.features.state_dict())
model.avgpool.load_state_dict(vgg_model.avgpool.state_dict())

model_params = model.state_dict()
vgg_params = vgg_model.state_dict()
for (key, value) in vgg_params.items():
    if key.startswith('features'):
        model_params[key] = value
    elif key == 'classifier.0.weight':
        model_params['classifier_1.weight'] = value
    elif key == 'classifier.0.bias':
        model_params['classifier_1.bias'] = value

    elif key == 'classifier.3.weight':
        model_params['classifier_2.weight'] = value
    elif key == 'classifier.3.bias':
        model_params['classifier_2.bias'] = value

    elif key == 'classifier.6.weight':
        model_params['classifier_3.weight'] = value
    elif key == 'classifier.6.bias':
        model_params['classifier_3.bias'] = value

model.load_state_dict(model_params)

# for name, param in model.named_parameters():
#     print(name)
# for name, param in vgg_model.named_parameters():
#     print(name)
model = model.cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_data = datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=64, shuffle=True
)
optim = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
optimal_acc = 0
for epoch in range(epoch):
    epoch_loss = 0
    model.train()
    for i, (img, y) in enumerate(train_loader):
        img = img.cuda()
        y = y.cuda()
        pred_y = model(img)
        loss = criterion(pred_y, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
    print(f"epoch[{epoch + 1}]: {epoch_loss}")
    if (epoch + 1) % 5 == 0:
        model.eval()
        total_right = 0
        total_samples = 0
        with torch.no_grad():
            for step, (img, y) in enumerate(test_loader):
                img = img.cuda()
                y = y.cuda()
                pred_y = model(img)
                loss = criterion(pred_y, y)
                pred_y = torch.argmax(pred_y, dim=-1)
                total_right += torch.sum(pred_y == y)
                total_samples += len(y)
            acc = float(total_right) / total_samples
            print(f"testing---epoch[{epoch + 1}]: {acc}")
            if acc > optimal_acc:
                optimal_acc = acc
                print("optimal acc", optimal_acc)
                torch.save(model, './model.pt')
