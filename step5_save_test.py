import torch
import torch.nn as nn
import cv2
import torchvision.transforms as T
from torchvision.models import alexnet
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

#训练并保存最佳模型
model = alexnet(weights="IMAGENET1K_V1").to(device)
for p in model.features.parameters():
    p.requires_grad=False
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
EPOCHS=10
best_acc=0

for ep in range(EPOCHS):
    loss_train = train_one_epoch(model, train_loader, criterion, optimizer, device)
    acc, prec, rec, f1 = evaluate(model, val_loader, device)
    print(f"ep{ep+1} acc={acc:.3f}")
    if acc>best_acc:
        best_acc=acc
        torch.save(model.state_dict(), "./best_model.pth")
print(f"训练结束，最佳acc={best_acc:.3f},已保存best_model.pth")

# --------单张图片预测函数--------
def predict_img(img_path):
    model.load_state_dict(torch.load("./best_model.pth", map_location=device))
    model.eval()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trans = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tensor = trans(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)
        pred_cls = torch.argmax(prob, dim=1).item()
    label_name = "cat" if pred_cls==0 else "dog"
    cat_prob = prob[0][0].item()
    dog_prob = prob[0][1].item()
    return label_name, cat_prob, dog_prob

# 修改这里为你要测试的图片路径
if __name__ == "__main__":
    test_path = r"./datasets/dog.55.jpg"
    res, c_p, d_p = predict_img(test_path)
    print(f"图片:{test_path} 预测结果={res} | cat概率:{c_p:.3f}, dog概率:{d_p:.3f}")

