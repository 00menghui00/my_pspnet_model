import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import xml.etree.ElementTree as ET


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for size in sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [feats]
        for stage in self.stages[1:]:
            priors.append(nn.functional.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(priors, dim=1)

class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PSPNet, self).__init__()
        
        backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        features = []
        for i, layer in enumerate(backbone.features):
            features.append(layer)
            if i == 7:
                break
        self.features = nn.Sequential(*features)
        
        in_channels = 80  # Output channels from the truncated MobileNetV3
        self.psp = PSPModule(in_channels=in_channels, out_channels=64)
        
        total_channels = in_channels + 64 * 4
        
        self.classifier = nn.Sequential(
            nn.Conv2d(total_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.features(x)
        x = self.psp(x)
        x = self.classifier(x)
        return nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, xml_dir, split_file, transform=None, num_classes=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.num_classes = num_classes
        
        # 从 TXT 文件读取图像文件名
        with open(split_file, 'r') as file:
            self.images = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx] + '.jpg')
        mask_name = os.path.join(self.mask_dir, self.images[idx] + '.png')
        
        # 读取图像和掩膜
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        # 调整图像和掩膜大小
        size = (512, 512)  # 设置为512, 512，确保与模型输入一致
        image = image.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        # 处理掩膜
        mask = transforms.ToTensor()(mask).long()  # 转换为 Tensor，并确保为长整型
        mask[mask > (self.num_classes - 1)] = self.num_classes - 1  # 确保类别不超出范围
        mask = mask.squeeze(0)  # 从 (1, H, W) 变为 (H, W)

        if self.transform:
            image = self.transform(image)

        return image, mask




def train_voc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 8
    num_classes = 3
    learning_rate = 0.001
    num_epochs = 3

    class_counts = [6193, 2375, 3813]
    total_samples = sum(class_counts)
    weights = [total_samples / (num_classes * count) for count in class_counts]
    weights = torch.FloatTensor(weights).to(device)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(
        image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
        mask_dir='F:/my_code/my_PSPNet/voc_data/SegmentationClass',
        xml_dir='F:/my_code/my_PSPNet/voc_data/Annotations/xml',
        split_file='F:/my_code/my_PSPNet/voc_data/ImageSets/train.txt',  # 添加训练集的 TXT 文件路径
        transform=transform,
        num_classes=num_classes
    )
    val_dataset = CustomDataset(
        image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
        mask_dir='F:/my_code/my_PSPNet/voc_data/SegmentationClass',
        xml_dir='F:/my_code/my_PSPNet/voc_data/Annotations/xml',
        split_file='F:/my_code/my_PSPNet/voc_data/ImageSets/val.txt',  # 添加验证集的 TXT 文件路径
        transform=transform,
        num_classes=num_classes
    )
    test_dataset = CustomDataset(
        image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
        mask_dir='F:/my_code/my_PSPNet/voc_data/SegmentationClass',
        xml_dir='F:/my_code/my_PSPNet/voc_data/Annotations/xml',
        split_file='F:/my_code/my_PSPNet/voc_data/ImageSets/test.txt',  # 添加测试集的 TXT 文件路径
        transform=transform,
        num_classes=num_classes
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = PSPNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    # 使用混合精度训练
    scaler = GradScaler()

    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.8f}')

        # Validation step
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                # 调整掩膜大小以匹配模型输出
                masks = nn.functional.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').long().squeeze(1)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss:.8f}')

        # Testing step
        test_loss = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f'Test Loss: {avg_test_loss:.8f}')

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.legend()
    plt.grid()
    plt.show()

    torch.save(model.state_dict(), 'pspnet_voc_8_0.pth')
    print("Model saved successfully")


if __name__ == "__main__":
    train_voc()
