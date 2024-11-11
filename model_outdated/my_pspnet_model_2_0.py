# 虚拟环境: python 3.9.20 pytorch_dl conda
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
from PIL import Image

# PSP Module and PSPNet classes remain the same as in the previous implementation
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
        
        # Use MobileNetV2 but only up to a certain layer to preserve spatial dimensions
        backbone = models.mobilenet_v2(pretrained=pretrained)
        self.features = backbone.features[:14]  # Use fewer layers to maintain spatial dimensions
        
        # Modify the backbone output channels accordingly
        in_channels = 96  # Output channels from the truncated MobileNetV2
        
        self.psp = PSPModule(in_channels=in_channels, out_channels=128)
        
        # Calculate total channels after PSP module
        total_channels = in_channels + 128 * 4  # Original + (PSP channels × number of scales)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(total_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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
            image_dir="F:/vscode/LF/JPEGImages",
            mask_dir="F:/vscode/LF/SegmentationClass",
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
        torch.save(model.state_dict(), 'pspnet_voc_2_0.pth')
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    train_voc()
