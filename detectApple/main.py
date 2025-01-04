import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import AppleDetector
from train import train_model
from test import test_model

# Các tham số cơ bản
DATA_DIR = 'dataset/'          # Thư mục chứa dữ liệu huấn luyện
MODEL_PATH = 'apple_detector.pth'  # File lưu mô hình sau huấn luyện
BATCH_SIZE = 32                # Số lượng mẫu trong một batch
EPOCHS = 10                    # Số epoch huấn luyện
LEARNING_RATE = 0.001          # Tốc độ học

# 1. Dataloader với Augmentation
# Dành cho dữ liệu huấn luyện (có Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize ảnh về kích thước cố định
    transforms.RandomHorizontalFlip(),        # Lật ngang ngẫu nhiên
    transforms.RandomRotation(30),           # Xoay ngẫu nhiên
    transforms.ColorJitter(brightness=0.2,    # Điều chỉnh độ sáng
                           contrast=0.2,
                           saturation=0.2),
    transforms.RandomAffine(degrees=0,        # Dịch chuyển và xoay ảnh
                            translate=(0.1, 0.1)),
    transforms.ToTensor(),                    # Chuyển đổi thành tensor
    transforms.Normalize([0.5, 0.5, 0.5],     # Chuẩn hóa giá trị pixel
                         [0.5, 0.5, 0.5])
])

# Dành cho dữ liệu kiểm thử (không Augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Tạo DataLoader
train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Khởi tạo mô hình, hàm mất mát, và bộ tối ưu
model = AppleDetector()  # Định nghĩa mô hình
criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam Optimizer

# 3. Huấn luyện mô hình
train_model(model, train_loader, criterion, optimizer, EPOCHS, MODEL_PATH)

# 4. Kiểm tra mô hình
# Đường dẫn tới ảnh kiểm thử
test_image_path = 'pexels-marina-gr-109305987-29535899.jpg'  # Đổi đường dẫn đến ảnh test nếu cần
test_model(model, test_image_path, MODEL_PATH, test_transform)