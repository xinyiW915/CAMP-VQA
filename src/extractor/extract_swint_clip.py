import torch
import torch.nn as nn
from timm import create_model

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SwinT(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', global_pool='avg', pretrained=True):
        super(SwinT, self).__init__()
        self.swin_model = create_model(
            model_name, pretrained=pretrained, global_pool=global_pool
        )
        self.swin_model.head = Identity()  # Remove classification head
        self.global_pool = global_pool

    def forward(self, x):
        features = self.swin_model(x)  # Shape: (batch_size, 7, 7, 1024)
        if self.global_pool == 'avg':
            features = features.mean(dim=[1, 2])  # Global pool
        return features

def extract_features_swint_pool(video, model, device):
    swint_feature_list = []

    with torch.amp.autocast(device_type='cuda'):
        for segment in video:
            # Flatten the segment into a batch of frames
            frames = segment.squeeze(0).to(device)  # Shape: (32, 3, 224, 224)

            swint_features = model(frames)  # Shape: (32, feature_dim)
            swint_feature_list.append(swint_features)

        # Concatenate features across segments
        features = torch.cat(swint_feature_list, dim=0)  # Shape: (num_frames, feature_dim)
    return features
