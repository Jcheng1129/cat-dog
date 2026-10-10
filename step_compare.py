import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, mobilenet_v3_small, shufflenet_v2_x1_0
from common import get_dataloader, train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = r"./datasets"

train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(0.5),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_loader, val_loader = get_dataloader(base_path, batch_size=8,
                                           transform_train=train_transform,
                                           transform_val=val_transform)

def build_model(name):
    if name == "resnet18":
        m = resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "mobilenet":
        m = mobilenet_v3_small(weights="IMAGENET1K_V1")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features,2)
    elif name == "shufflenet":
        m = shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features,2)
    else:
        raise ValueError
    return m.to(device)

model_list = ["resnet18","mobilenet","shufflenet"]
EPOCHS = 8
result_table = []

for model_name in model_list:
    print(f"\n========开始训练 {model_name}========")
    model = build_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0
    for ep in range(EPOCHS):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, p, r, f1 = evaluate(model, val_loader, device)
        print(f"ep{ep+1} acc={acc:.3f}")
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"./model_{model_name}.pth")
    result_table.append([model_name, best_acc])

print("\n====模型对比结果====")
for row in result_table:
    print(f"{row[0]:12s} | best_acc={row[1]:.4f}")

