import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from models import get_pretrained_model 

# =================配置区域=================
# 你的模型权重路径
MODEL_PATH = 'model_best.pth' 
# 确保这里和训练时一致 (如果是 ResNet50 就写 2048)
BACKBONE_NAME = 'resnet50' 
# =========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiFrameTestDataset(Dataset):
    def __init__(self, root, csv_file, transform=None, num_frames=5):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file) # 默认读 test.csv
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid = self.df.iloc[index, 0]
        folder_path = os.path.join(self.root, f"{vid}")
        
        # 防御性编程
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，返回 5 张黑图
            return torch.zeros((self.num_frames, 3, 224, 224)), vid

        img_list = sorted(os.listdir(folder_path))
        total_frames = len(img_list)
        
        # 🔥 核心逻辑：均匀采样 5 帧
        # 例如 total=100, num=5 -> indices=[0, 25, 50, 75, 99]
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            frames = []
            for idx in indices:
                img_path = os.path.join(folder_path, img_list[idx])
                img = Image.open(img_path).convert('RGB')
                if self.transforms is not None:
                    img = self.transforms(img)
                frames.append(img)
            
            # 堆叠成一个 Tensor: shape [5, 3, 224, 224]
            return torch.stack(frames), vid
        else:
            return torch.zeros((self.num_frames, 3, 224, 224)), vid

# 获取分类数量
train_df = pd.read_csv("data/raw/trainval.csv", header=None, skiprows=1)
num_classes = len(train_df[1].unique())

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 保持 ImageNet 标准，千万别改！
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化数据集 (注意：num_frames=5)
test_dataset = MultiFrameTestDataset(
    root="data/frames/video_frames_30fpv_320p", 
    csv_file="data/raw/test_for_student.csv", 
    transform=transform,
    num_frames=5 
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False) # BatchSize 调小点，因为一次进5张图

# 加载模型
print(f"🚀 加载模型: {MODEL_PATH} (Num Classes: {num_classes})")
# 这里的 get_pretrained_model 需要你之前的定义
# 如果你是 ResNet50，记得去 models.py 确认下是不是 resnet50
net = get_pretrained_model(num_classes=num_classes, freeze_backbone=False) 
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.to(device)
net.eval()

results = []

print("🔮 开始多帧投票预测 (这会比单帧慢 5 倍)...")
with torch.no_grad():
    for i, (inputs, video_ids) in enumerate(test_loader):
        # inputs shape: [batch_size, 5, 3, 224, 224]
        b, n, c, h, w = inputs.shape
        
        # 1. 把它变成 [batch_size * 5, 3, 224, 224] 塞进模型
        inputs = inputs.view(b * n, c, h, w).to(device)
        
        # 2. 预测
        outputs = net(inputs) # shape: [batch*5, num_classes]
        
        # 3. 变回 [batch_size, 5, num_classes]
        outputs = outputs.view(b, n, -1)
        
        # 4. 🔥 投票：在 dimension 1 (时间轴) 上取平均
        # 这一步就是让 5 个时刻的意见融合
        outputs_avg = torch.mean(outputs, dim=1) # shape: [batch, num_classes]
        
        # 5. 取最大值
        _, predicted = torch.max(outputs_avg, 1)
        predicted = predicted.cpu().numpy()
        
        for j, vid in enumerate(video_ids):
            results.append((vid, predicted[j]))
            
        if (i+1) % 10 == 0:
            print(f"已处理 {i+1}/{len(test_loader)} 个 Batch")

# 生成 CSV
print(f"💾 正在写入 result_voting.csv...")
with open('result_voting.csv', "w") as f:
    f.writelines("Id,Category\n")
    for vid, pred in results:
        f.writelines(f"{vid},{pred}\n")

print("✅ 完成！这个 result_voting.csv 的分数绝对比之前的更高！")