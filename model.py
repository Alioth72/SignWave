import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and dropout"""
    def __init__(self, in_channels: int, out_channels: int, 
                 dropout_rate: float = 0.3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = self.dropout(x)
        return x

class SignLanguageTranslator(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 hidden_dim: int = 512):
        super().__init__()
        
        # Store configuration
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        
        # CNN Feature Extractor
        self.features = nn.Sequential(
            ConvBlock(input_shape[0], 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate feature map size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            feat_size = self.features(dummy_input).view(1, -1).shape[1]
        
        # Sequence classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return torch.softmax(x, dim=-1)
