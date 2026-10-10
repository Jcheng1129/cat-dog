import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------0.2 数据集类----------------------
class CatDogDataset(Dataset):
    def __init__(self, file_list, img_size=(224,224), transform=None):
        self.file_list = file_list
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        # label: cat=0 dog=1
        filename = os.path.basename(path)
        if filename.startswith("cat"):
            label = 0
        else:
            label = 1

        if self.transform is not None:
            img = self.transform(img)
        return img, label

def get_dataloader(base_path, batch_size=8, transform_train=None, transform_val=None):
    all_files = []
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if full_path.lower().endswith((".jpg",".jpeg",".png")):
            all_files.append(full_path)

    train_list, val_list = train_test_split(all_files, test_size=0.2, random_state=42)

    train_ds = CatDogDataset(train_list, transform=transform_train)
    val_ds = CatDogDataset(val_list, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ----------------------0.3 训练、评估函数----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(labels.numpy())

    acc = accuracy_score(all_true, all_pred)
    p = precision_score(all_true, all_pred)
    r = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    return acc, p, r, f1

