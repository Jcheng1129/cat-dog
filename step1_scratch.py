import torch
import torch.nn as nn
import torchvision.transforms as T
from common import get_dataloader, train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

base_path = r"./datasets"

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_loader, val_loader = get_dataloader(base_path, batch_size=8,
                                           transform_train=transform,
                                           transform_val=transform)

# AlexNet 从零搭建
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096, 2)
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10
for epoch in range(EPOCHS):
    loss_train = train_one_epoch(model, train_loader, criterion, optimizer, device)
    acc, prec, rec, f1 = evaluate(model, val_loader, device)
    print(f"Epoch[{epoch+1}/{EPOCHS}] loss={loss_train:.3f} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
