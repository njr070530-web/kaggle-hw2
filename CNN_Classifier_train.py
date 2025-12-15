import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import random
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

# ... import 部分保持不变 ...

# 1. 定义两个 transform
# 训练用：花里胡哨，增强泛化
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证用：老老实实，只做标准化
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. ⚠️ 关键修改：先拆分数据，再分别创建 Dataset
# 我们先读取一次 CSV 来做 split
full_df = pd.read_csv("data/raw/trainval.csv", header=None, skiprows=1)
# 只要索引，不需要数据
indices = list(range(len(full_df)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=0)

# 利用 Subset 或者重新实现 Dataset 来区分 transform
# 为了不改动你的 MyDataset 类结构，最简单的办法是实例化两次，但传入不同的 indices（需要稍微改下 Dataset）
# 或者更粗暴的方法（不改 Dataset 类）：

class MyDataset(Dataset):
    def __init__(self, root, csv_file, mode='train', transform=None, indices=None):
        self.root = root
        self.transforms = transform
        df_all = pd.read_csv(csv_file, header=None, skiprows=1)
        
        # ✅ 如果传入了 indices，就只取这部分
        if indices is not None:
            self.df = df_all.iloc[indices, :].reset_index(drop=True)
        else:
            self.df = df_all
            
        self.classes = sorted(df_all[1].unique()) # 这一步要小心，必须用全集算 classes
        self.mode = mode # 记录模式

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = sorted(os.listdir(os.path.join(self.root, f"{vid}")))

        # ✅ 逻辑修正：
        # 只有在 mode='train' 且有 transform 时才随机抽
        if self.mode == 'train':
            idx = random.randint(0, len(img_list) - 1)
        else:
            # val 和 test 永远取中间
            idx = int(len(img_list) / 2)
            
        img_path = os.path.join(self.root, f"{vid}", img_list[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label

# 3. 实例化数据集 (修复数据泄露问题)
train_dataset = MyDataset("data/frames/video_frames_30fpv_320p", "data/raw/trainval.csv", 
                          mode='train', transform=train_transform, indices=train_idx)

val_dataset = MyDataset("data/frames/video_frames_30fpv_320p", "data/raw/trainval.csv", 
                        mode='val', transform=val_transform, indices=val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ... 模型加载部分不变 ...
num_classes = len(train_dataset.classes)
net = get_pretrained_model(num_classes=num_classes, freeze_backbone=True).to(device)

# 4. 🚀 初始参数调整 (Stage 1)
# Weight Decay 降为 1e-4，防止欠拟合
optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, net.parameters()),
                            lr=0.005, momentum=0.9, weight_decay=1e-4) 

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

UNFREEZE_EPOCH = 5 # 建议改为 5，ResNet 没必要冻结那么久
best_acc = 0.0

for epoch in range(50):
    # ... 前面代码不变 ...
    t0 = time.time()
    
    if epoch == UNFREEZE_EPOCH:
        print("🔓 解冻所有层，开始微调整个模型...")
        for param in net.parameters():
            param.requires_grad = True
            
        # 5. 🚀 解冻参数调整 (Stage 2 - 关键！)
        optimizer = torch.optim.SGD([
            # Backbone 维持 1e-4 不变 (ResNet 层)
            {'params': net.conv1.parameters(), 'lr': 1e-4},
            {'params': net.bn1.parameters(),   'lr': 1e-4},
            {'params': net.layer1.parameters(), 'lr': 1e-4},
            {'params': net.layer2.parameters(), 'lr': 1e-4},
            {'params': net.layer3.parameters(), 'lr': 1e-4},
            {'params': net.layer4.parameters(), 'lr': 1e-4},
            
            # ⚠️ FC 层提高到 1e-3 (0.001)！不要用 0.0001，太小了！
            {'params': net.fc.parameters(),    'lr': 1e-3} 
        ], momentum=0.9, weight_decay=1e-4) # 统一 Weight Decay
        
        # 重置 Scheduler，给它机会重新衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # ... 后面的训练循环不变 ...
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
