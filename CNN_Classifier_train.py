import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from models import Net
from models import get_pretrained_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'trainig on :{torch.cuda.get_device_name(0)}')
else:
    print('training on : CPU')

class MyDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
 
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label

# You can add data augmentation here
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 标准均值
            std=[0.229, 0.224, 0.225])   # ImageNet 标准方差
        ])


trainval_dataset = MyDataset("data/frames/video_frames_30fpv_320p", "data/raw/trainval.csv", transform)
train_data, val_data = train_test_split(trainval_dataset, test_size=0.2, random_state=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

num_classes = len(trainval_dataset.classes)
print(f"检测到 {num_classes} 个分类，准备加载 ResNet...")

# net = Net().to(device)
net = get_pretrained_model(num_classes=num_classes, freeze_backbone=True).to(device)
optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,net.parameters()),
lr=1e-3,momentum=0.9,weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

UNFREEZE_EPOCH=5
best_acc = 0.0
for epoch in range(50):
    t0=time.time()
    # Metrics here ...
    if epoch == UNFREEZE_EPOCH:
        print("🔓 解冻所有层，开始微调整个模型...")
        for param in net.parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD([
    # Backbone 用小火慢炖 (1e-4)
    {'params': net.conv1.parameters(), 'lr': 1e-4},
    {'params': net.bn1.parameters(),   'lr': 1e-4},
    {'params': net.layer1.parameters(), 'lr': 1e-4},
    {'params': net.layer2.parameters(), 'lr': 1e-4},
    {'params': net.layer3.parameters(), 'lr': 1e-4},
    {'params': net.layer4.parameters(), 'lr': 1e-4},
    # 全连接层 (fc) 用大火爆炒 (1e-3 或 5e-3)
    {'params': net.fc.parameters(),    'lr': 1e-3} 
], momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    net.train()
    running_loss=0.0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"Epoch [{epoch+1}/50], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
            running_loss = 0.0
            current_lr = optimizer.param_groups[-1]['lr'] 
            print(f"Epoch {epoch+1} done. Current FC LR: {current_lr}")
        # Training code ...
    
    scheduler.step()
    net.eval()
    correct=0
    total=0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels= val_inputs.to(device), val_labels.to(device)
            outputs=net(val_inputs)
            predicted=outputs.argmax(dim=1)
            total+=val_labels.size(0)
            correct+=(predicted==val_labels).sum().item()
    t1=time.time()
    acc = 100 * correct / total
    print(f"⏱️  训练时间: {t1 - t0:.2f} 秒")
    print(f"🔥 测试集准确率: {100 * correct / total:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), 'model_best.pth')
        print(f"🏆 新纪录！模型已保存 (Acc: {best_acc:.2f}%)")

            # Validation code ...


    torch.save(net.state_dict(), 'model_last.pth')
