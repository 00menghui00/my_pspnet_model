import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision.models as models
from PIL import Image

# PSPNet模型代码保持不变...

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

# PSPNet的定义保持不变...
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
    
    # Load full dataset
    dataset = CustomDataset(
        image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
        mask_dir="F:/my_code/my_PSPNet/voc_data/SegmentationClass",
        transform=transform
    )
    
    # Split dataset into train, val, test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create dataloaders for train, val, test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = PSPNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'pspnet_voc_4_0.pth')
    print("Model saved successfully")
    
    # Testing loop
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

if __name__ == "__main__":
    train_voc()
