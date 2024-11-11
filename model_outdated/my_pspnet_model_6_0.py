
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

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
            nn.Dropout2d(p=0.3),
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
    def __init__(self, image_dir, mask_dir, split_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Load image filenames from the split file (train.txt, val.txt, or test.txt)
        with open(split_file, 'r') as file:
            self.images = [line.strip() for line in file.readlines()]

        # Define mask transform separately
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx] + '.jpg')
        mask_name = os.path.join(self.mask_dir, self.images[idx] + '.png')
        
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
    batch_size = 4
    num_classes = 4
    learning_rate = 0.001
    num_epochs = 60
    patience=5
    
    # Data preprocessing for images
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader for training
    train_dataset = CustomDataset(
        image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
        mask_dir='F:/my_code/my_PSPNet/voc_data/SegmentationClass',
        split_file='F:/my_code/my_PSPNet/voc_data/ImageSets/train.txt',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create dataset and dataloader for validation
    val_dataset = CustomDataset(
        image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
        mask_dir='F:/my_code/my_PSPNet/voc_data/SegmentationClass',
        split_file='F:/my_code/my_PSPNet/voc_data/ImageSets/val.txt',
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create dataset and dataloader for testing (if needed)
    test_dataset = CustomDataset(
        image_dir='F:/my_code/my_PSPNet/voc_data/JPEGImages',
        mask_dir='F:/my_code/my_PSPNet/voc_data/SegmentationClass',
        split_file='F:/my_code/my_PSPNet/voc_data/ImageSets/test.txt',
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = PSPNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)

    # Learning rate scheduler
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    best_val_loss=float('inf')
    epochs_no_improve=0

    #Lists to store Losses
    train_losses=[]
    val_losses=[]
    test_losses=[]

    # Training loop (you can also add validation after each epoch if needed)

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
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
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
        print(f'Test Loss: {avg_test_loss:.4f}')

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
            print("Validation loss decreased. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

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

    # Save the model
    torch.save(model.state_dict(), 'pspnet_voc_6_0.pth')
    print("Model saved successfully")
    
if __name__ == "__main__":
    train_voc()
