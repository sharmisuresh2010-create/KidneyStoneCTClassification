import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def reset_random(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

class LocalHighResolutionCNN(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.GELU(),
            nn.Conv2d(32,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU(),
            nn.Conv2d(64,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.GELU(),
            nn.Conv2d(128,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection=nn.Sequential(nn.Flatten(),nn.Linear(128,256),nn.LayerNorm(256),nn.GELU())
    def forward(self,x):
        return self.projection(self.features(x))

class MorphologyAwareMultiScaleConv(nn.Module):
    def __init__(self,in_channels=768,out_channels=256):
        super().__init__()
        branch_channels=out_channels//3
        self.branch1=nn.Sequential(nn.Conv2d(in_channels,branch_channels,1,bias=False),nn.BatchNorm2d(branch_channels),nn.GELU())
        self.branch3=nn.Sequential(nn.Conv2d(in_channels,branch_channels,3,padding=1,bias=False),nn.BatchNorm2d(branch_channels),nn.GELU())
        self.branch5=nn.Sequential(nn.Conv2d(in_channels,branch_channels,5,padding=2,bias=False),nn.BatchNorm2d(branch_channels),nn.GELU())
        self.fusion=nn.Sequential(nn.Conv2d(branch_channels*3,out_channels,1,bias=False),nn.BatchNorm2d(out_channels),nn.GELU())
    def forward(self,x):
        return self.fusion(torch.cat([self.branch1(x),self.branch3(x),self.branch5(x)],dim=1))

class LGMRNet(nn.Module):
    def __init__(self,num_classes=2,pretrained=True,dropout=0.3):
        super().__init__()
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone=models.convnext_tiny(weights=weights)
        self.global_features=backbone.features
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.global_projection=nn.Sequential(nn.Flatten(),nn.Linear(768,256),nn.LayerNorm(256),nn.GELU())
        self.local_branch=LocalHighResolutionCNN(3)
        self.gate=nn.Sequential(nn.Linear(512,256),nn.Sigmoid())
        self.morphology= MorphologyAwareMultiScaleConv(768,256)
        self.fused_projection=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,num_classes))
    def forward(self,x,return_features=False):
        local_features=self.local_branch(x)
        global_map=self.global_features(x)
        global_features=self.global_projection(self.global_pool(global_map))
        gate=self.gate(torch.cat([local_features,global_features],dim=1))
        gated_local=gate*local_features
        gated_global=(1-gate)*global_features
        morphology_features=self.morphology(global_map)
        morphology_features=F.adaptive_avg_pool2d(morphology_features,1).flatten(1)
        fused=torch.cat([gated_local,gated_global],dim=1)
        fused=self.fused_projection(fused)
        combined=torch.cat([fused,morphology_features],dim=1)
        logits=self.classifier(combined)
        if return_features:
            return logits,{"local":local_features,"global":global_features,"gate":gate,"morphology":morphology_features,"fused":combined}
        return logits
    def predict_proba(self,x):
        return torch.softmax(self.forward(x),dim=1)

def build_model(num_classes=2,pretrained=True,dropout=0.3):
    return LGMRNet(num_classes=num_classes,pretrained=pretrained,dropout=dropout)