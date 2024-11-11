import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
from PIL import Image

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
        
        # Use MobileNetV3-Large but only up to certain layers to preserve spatial dimensions
        backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Extract features up to a certain point to maintain spatial dimensions
        # MobileNetV3 has different layer structure than V2
        features = []
        for i, layer in enumerate(backbone.features):
            features.append(layer)
            if i == 7:  # Stop after the 7th layer to maintain spatial dimensions
                break
        self.features = nn.Sequential(*features)
        
        # Modify the backbone output channels accordingly
        # MobileNetV3-Large has different channel sizes
        in_channels = 80  # Output channels from the truncated MobileNetV3
        
        self.psp = PSPModule(in_channels=in_channels, out_channels=128)
        
        # Calculate total channels after PSP module
        total_channels = in_channels + 128 * 4  # Original + (PSP channels × number of scales)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(total_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),  # Using Hardswish as in MobileNetV3
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        
        # Extract features
        x = self.features(x)
        
        # Apply PSP module
        x = self.psp(x)
        
        # Apply final classifier
        x = self.classifier(x)
        
        # Upsample to input size
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
        # Define mask transform separately
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))
        
        # Load image and mask
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)
        
        # Convert mask to long tensor
        mask = mask.squeeze(0).long()
        
        return image, mask

def train_voc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 2
    num_classes = 4
    learning_rate = 0.001
    num_epochs = 2
    
    # Data preprocessing for images
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    try:
        train_dataset = CustomDataset(
            image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
            mask_dir="F:/my_code/my_PSPNet/voc_data/SegmentationClass",
            transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"Dataset size: {len(train_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize model, loss function, and optimizer
    model = PSPNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            try:
                images, masks = images.to(device), masks.to(device)
                
                # Print shapes for debugging (only in first iteration)
                if i == 0 and epoch == 0:
                    print(f"Image shape: {images.shape}")
                    print(f"Mask shape: {masks.shape}")
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Print output shape for debugging (only in first iteration)
                if i == 0 and epoch == 0:
                    print(f"Output shape: {outputs.shape}")
                
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            except Exception as e:
                print(f"Error during training iteration: {e}")
                print(f"Image shape: {images.shape}")
                print(f"Mask shape: {masks.shape}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save the model
    try:
        torch.save(model.state_dict(), 'pspnet_voc_3_0.pth')
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    train_voc()
'''
3_0相对于2_0版本的改进：
两段代码在结构上非常相似，但有一些关键的不同之处。下面是这两段代码的比较：

1. 基础模型的选择
第一段代码使用的是MobileNetV2作为基础模型：

python
复制代码
backbone = models.mobilenet_v2(pretrained=pretrained)
self.features = backbone.features[:14]
第二段代码使用的是MobileNetV3-Large作为基础模型：

python
复制代码
backbone = models.mobilenet_v3_large(pretrained=pretrained)
features = []
for i, layer in enumerate(backbone.features):
    features.append(layer)
    if i == 7:  # Stop after the 7th layer
        break
self.features = nn.Sequential(*features)
2. 输出通道数的设置
第一段代码设置的输出通道数为96：

python
复制代码
in_channels = 96  # Output channels from the truncated MobileNetV2
第二段代码则设置为80：

python
复制代码
in_channels = 80  # Output channels from the truncated MobileNetV3
3. 激活函数的不同
第一段代码使用ReLU作为激活函数：

python
复制代码
nn.ReLU(inplace=True)
第二段代码使用Hardswish，这也是MobileNetV3中引入的激活函数：

python
复制代码
nn.Hardswish(inplace=True)
4. 特征提取层的停止条件
第一段代码在特征提取时从MobileNetV2中提取前14层。

第二段代码则在MobileNetV3中提取到第7层就停止。这是因为MobileNetV3的结构和层数与MobileNetV2不同。

5. 模型保存文件名
第一段代码的模型保存为'pspnet_voc_2_0.pth'。

第二段代码的模型保存为'pspnet_voc_3_0.pth'。

6. 总的设计意图
尽管两段代码在逻辑结构上基本一致，但第二段代码针对MobileNetV3进行了优化，适配了新的特征层和激活函数，可能希望利用MobileNetV3在精度和效率上的改进。
总结
这两段代码的主要区别在于使用的基础网络不同（MobileNetV2与MobileNetV3-Large），对应的输出通道数、激活函数和特征提取的方式也有所不同。这些变化可能会影响模型的性能和训练效果。
'''