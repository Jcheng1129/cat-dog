import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import alexnet
from common import get_dataloader, train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = r"./datasets"

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_loader, val_loader = get_dataloader(base_path, batch_size=8,
                                           transform_train=transform,
                                           transform_val=transform)

# 加载预训练AlexNet，冻结卷积层
model = alexnet(weights="IMAGENET1K_V1").to(device)
for param in model.features.parameters():
    param.requires_grad = False

# 修改最后输出层，2分类猫狗
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

EPOCHS = 10
for epoch in range(EPOCHS):
    loss_train = train_one_epoch(model, train_loader, criterion, optimizer, device)
    acc, prec, rec, f1 = evaluate(model, val_loader, device)
    print(f"Epoch[{epoch+1}/{EPOCHS}] loss={loss_train:.3f} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
