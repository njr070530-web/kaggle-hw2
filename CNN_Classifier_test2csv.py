# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import os
# from PIL import Image
# from models import Net


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class MyDataset(Dataset):
#     def __init__(self, root, csv_file, transform=None):
#         self.root = root
#         self.transforms = transform
#         self.df = pd.read_csv(csv_file, header=None, skiprows=1)
#         self.classes = sorted(self.df[1].unique())

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         vid, label = self.df.iloc[index, :]
#         img_list = os.listdir(os.path.join(self.root, f"{vid}"))
#         img_list = sorted(img_list)
#         img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
 
#         img = Image.open(img_path).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)

#         label = self.classes.index(label)
#         return img, label

# transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
# ])

# test_dataset = MyDataset("video_frames_30fpv_320", "test_for_student.csv", transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Load Model
# net = Net().to(device)
# net.load_state_dict(torch.load('model_best.pth'))

# # Evaluation
# net.eval()
# result = []
# with torch.no_grad():

        
# fread = open("test_for_student.label", "r")
# video_ids = []
# for line in fread.readlines():
#     video_id = os.path.splitext(line.strip())[0]
#     video_ids.append(video_id)


# with open('result.csv', "w") as f:
#     f.writelines("Id,Category\n")
#     for i, pred_class in enumerate(result):
#         f.writelines("%s,%d\n" % (video_ids[i], pred_class))


import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from models import get_pretrained_model  # ✅ 必须用这个！

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 专门为测试集写的 Dataset
#    区别：它不读 Label，而是返回 VideoID
# ==========================================
class TestDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        # 假设测试集 csv 只有一列（视频文件名）
        # 如果有 header，就把 header=None 去掉
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 获取视频 ID (假设在第一列)
        vid = self.df.iloc[index, 0]
        
        # 拼凑图片路径 (逻辑和你训练时一样)
        folder_path = os.path.join(self.root, f"{vid}")
        
        # 防御性编程：万一文件夹不存在
        if not os.path.exists(folder_path):
            print(f"⚠️ 警告: 找不到文件夹 {folder_path}，跳过...")
            # 返回一个黑图防止崩溃（实际比赛中应该不会发生）
            img = Image.new('RGB', (224, 224))
        else:
            img_list = sorted(os.listdir(folder_path))
            # 取中间帧
            img_path = os.path.join(folder_path, img_list[int(len(img_list)/2)])
            img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, vid  # ✅ 返回图片和视频ID

# ==========================================
# 2. 准备工作
# ==========================================

# ⚠️ 关键步骤：我们需要加载训练集的 CSV 来恢复“标签字典”
# 这样我们才知道 模型输出的 "0" 代表 "ApplyEyeMakeup" 还是 "Archery"
train_df = pd.read_csv("data/raw/trainval.csv", header=None, skiprows=1)
classes = sorted(train_df[1].unique())
num_classes = len(classes)
print(f"📖 字典已恢复，共 {num_classes} 个分类")

# 数据预处理 (和训练时保持一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化测试集
# 注意路径要对！你的截图里 data 是在 data/frames/... 下面
test_dataset = TestDataset(
    root="data/frames/video_frames_30fpv_320p", # 👈 确保这个路径是对的！
    csv_file="data/raw/test_for_student.csv", 
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================================
# 3. 加载模型 & 预测
# ==========================================
print("🚀 正在加载模型...")
# 必须和训练时参数一致
net = get_pretrained_model(num_classes=num_classes, freeze_backbone=True) 
net.load_state_dict(torch.load('model_last.pth', map_location=device))
net.to(device)
net.eval()

results = []

print("🔮 开始预测...")
with torch.no_grad():
    for inputs, video_ids in test_loader:
        inputs = inputs.to(device)
        
        # 1. 预测原图
        outputs_orig = net(inputs)
        
        # 2. 预测翻转图 (手动把图翻转一下)
        inputs_flipped = torch.flip(inputs, dims=[3]) 
        outputs_flipped = net(inputs_flipped)
        
        # 3. 结果相加 (融合)
        outputs_final = (outputs_orig + outputs_flipped) / 2.0
        
        _, predicted_indices = torch.max(outputs_final, 1)
        
        # 存结果
        for i, vid in enumerate(video_ids):
            # 这里很重要：你是要提交 Category 的名字(String) 还是 ID(Int)?
            # 如果 result.csv 要求是 0,1,2，就这样：
            pred_class = predicted_indices[i] 
            
            # ⚠️ 如果 result.csv 要求是 "ApplyEyeMakeup" 这种名字，就要这句：
            # pred_name = classes[pred_class]
            
            results.append((vid, pred_class))

# ==========================================
# 4. 生成 CSV
# ==========================================
print(f"💾 正在写入 result.csv (共 {len(results)} 条)...")
with open('result.csv', "w") as f:
    f.writelines("Id,Category\n")
    for vid, pred in results:
        f.writelines(f"{vid},{pred}\n") # 确保格式符合 Kaggle 要求

print("✅ 完成！快去提交 result.csv 吧！")