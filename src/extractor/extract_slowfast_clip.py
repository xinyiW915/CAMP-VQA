import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50

def pack_pathway_output(frames, device):
    fast_pathway = frames
    # temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // 4).long(),
    )
    return [slow_pathway.to(device), fast_pathway.to(device)]


class SlowFast(torch.nn.Module):
    def __init__(self):
        super(SlowFast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)
            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])
            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
        return slow_feature, fast_feature

def extract_features_slowfast_pool(video, model, device):
    slow_features_list = []
    fast_features_list = []

    with torch.amp.autocast(device_type='cuda'):
        for idx, segment in enumerate(video):
            segment = segment.permute(0, 2, 1, 3, 4)
            inputs = pack_pathway_output(segment, device)
            # print(f"Inputs shapes: slow={inputs[0].shape}, fast={inputs[1].shape}")

            # extract features
            slow_feature, fast_feature = model(inputs)
            # global average pooling to reduce dimensions
            slow_feature = slow_feature.mean(dim=[2, 3, 4])  # Pool over spatial and temporal dims
            fast_feature = fast_feature.mean(dim=[2, 3, 4])
            slow_features_list.append(slow_feature)
            fast_features_list.append(fast_feature)

    # concatenate pooled features
    slow_features = torch.cat(slow_features_list, dim=0)
    fast_features = torch.cat(fast_features_list, dim=0)
    slowfast_features = torch.cat((slow_features, fast_features), dim=1)  # along feature dimension
    return slow_features, fast_features, slowfast_features