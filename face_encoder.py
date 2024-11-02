import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image

class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FaceEncoder, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        return self.resnet(x)

def encode_face(face_image, encoder, device):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    face_tensor = transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        face_encoding = encoder(face_tensor)
    
    return face_encoding.cpu().numpy().flatten()
