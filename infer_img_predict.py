import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

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
    
# Load the model
def load_model(model_path, num_classes=3):
    model = PSPNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
    return model

# Prepare the image
def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print("Input tensor shape:", input_tensor.shape)  # 输出输入张量的形状
    return image, input_tensor

# Make predictions with debug output
# Make predictions with probabilities
def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.argmax(output, dim=1)  # Get the predicted class
    return output, predicted_mask  # Return both output and predicted mask



# Visualize results
def visualize_results(original_image, output):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    
    mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # 去掉通道维度
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='jet')  # Use 'jet' colormap for visualization
    plt.title("Predicted Mask")
    
    plt.show()

if __name__ == "__main__":
    model_path = 'F:/my_code/my_PSPNet/pspnet_voc_8_0.pth'  # Adjust the path to your model
    image_path = r"F:\my_code\my_PSPNet\voc_data\JPEGImages\fff99912-2321-46f9-9f81-20f0e92967764353f3d8a8e1-994a-47b5-b07e-352904300152.jpg"  # Path to the image you want to test
    
    # Load the model
    model = load_model(model_path)

    # Prepare the image
    original_image, input_tensor = prepare_image(image_path)

    # Make predictions
    output, predicted_mask = predict(model, input_tensor)

    # Visualize results
    visualize_results(original_image, output)  # Pass output to visualize


'''
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
    
# Load the model
def load_model(model_path, num_classes=4):
    model = PSPNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Prepare the image
def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, input_tensor

# Make predictions
def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.argmax(output, dim=1)  # Get the predicted class
    return predicted_mask.squeeze(0)  # Remove the batch dimension

# Visualize results
def visualize_results(original_image, predicted_mask):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask.cpu().numpy(), cmap='jet')  # Use 'jet' colormap for visualization
    plt.title("Predicted Mask")
    
    plt.show()

if __name__ == "__main__":
    model_path = 'F:/my_code/my_PSPNet/pspnet_voc_7_0.pth'  # Adjust the path to your model
    image_path = 'F:/my_code/my_PSPNet/voc_data/JPEGImages/0a06c482-c94a-44d8-a895-be6fe17b8c06___FAM_B.Rot 5019_final_masked.jpg'  # Path to the image you want to test
    
    # Load the model
    model = load_model(model_path)

    # Prepare the image
    original_image, input_tensor = prepare_image(image_path)

    # Make predictions
    predicted_mask = predict(model, input_tensor)

    # Visualize results
    visualize_results(original_image, predicted_mask)
'''