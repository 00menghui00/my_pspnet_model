import cv2
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

# Define your PSPModule and PSPNet classes here
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

# Load the model
def load_model(model_path, num_classes=4):
    model = PSPNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Prepare the image
def prepare_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
    return input_tensor

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

def main():
    model_path = 'F:/my_code/my_PSPNet/pspnet_voc_6_0.pth'  # Adjust the path to your model
    model = load_model(model_path)

    # Start capturing from the webcam
    cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare the image for the model
        input_tensor = prepare_image(frame_rgb)

        # Make predictions
        predicted_mask = predict(model, input_tensor)

        # Visualize results (you can also overlay the mask on the frame)
        visualize_results(frame_rgb, predicted_mask)

        # Show the original frame
        cv2.imshow('Webcam', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
