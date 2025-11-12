import argparse
import os
import sys
import subprocess
import json
import ffmpeg
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from extractor.extract_frag import VideoDataset_feature
from extractor.extract_clip_embeds import extract_features_clip_embed
from extractor.extract_slowfast_clip import SlowFast, extract_features_slowfast_pool
from extractor.extract_swint_clip import SwinT, extract_features_swint_pool
from model_finetune import fix_state_dict
from model_regression_lsvq import preprocess_data


def get_transform(resize):
    return transforms.Compose([transforms.Resize([resize, resize]),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

def setup_device(config):
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
    return device

def load_prompts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(config, device, Mlp, input_features=13056):
    model = Mlp(input_features=input_features, out_features=1, drop_rate=0.1, act_layer=nn.GELU).to(device)

    if config.intra_cross_experiment == 'intra':
        if config.train_data_name == 'lsvq_train':
            from model_regression_lsvq import Mlp, preprocess_data
            if config.test_data_name  == 'lsvq_test':
                model_path = os.path.join(config.save_model_path, f"./{config.train_data_name}_{config.network_name}_{config.model_name}_{config.select_criteria}_trained_model_kfold.pth")
            elif config.test_data_name  == 'lsvq_test_1080p':
                model_path = os.path.join(config.save_model_path, f"./{config.train_data_name}_{config.network_name}_{config.model_name}_{config.select_criteria}_trained_model_1080p.pth")
            else:
                print("Please use a cross-dataset experiment setting for the lsvq_train model to test it on another dataset, please try using the input 'cross' for 'intra_cross_experiment'.")
                sys.exit(1)
        else:
            from model_regression import Mlp, preprocess_data
            model_path = os.path.join(config.save_model_path, f"best_model/{config.train_data_name}_{config.network_name}_{config.model_name}_{config.select_criteria}_trained_model.pth")

    elif config.intra_cross_experiment == 'cross':
        from model_regression_lsvq import Mlp, preprocess_data
        if config.train_data_name == 'lsvq_train':
            if config.is_finetune:
                model_path = os.path.join(config.save_model_path, f"fine_tune/{config.test_data_name}_{config.network_name}_fine_tuned_model.pth")
            else:
                model_path = os.path.join(config.save_model_path, f"./{config.train_data_name}_{config.network_name}_{config.model_name}_{config.select_criteria}_trained_model_kfold.pth")
        else:
            print("Invalid training data name for cross-experiment. We provided the lsvq_train model for the cross-experiment, please try using the input 'lsvq_train' for 'train_data_name'.")
            sys.exit(1)

    print("Loading model from:", model_path)
    state_dict = torch.load(model_path, map_location=device)
    fixed_state_dict = fix_state_dict(state_dict)
    try:
        model.load_state_dict(fixed_state_dict)
    except RuntimeError as e:
        print(e)
    return model

def evaluate_video_quality(data_loader, model_slowfast, model_swint, clip_model, clip_preprocess, blip_processor, blip_model, prompts, model_mlp, device):
    # get video features
    model_slowfast.eval()
    model_swint.eval()
    clip_model.eval()
    blip_model.eval()
    with torch.no_grad():
        for i, (video_segments, video_res_frag_all, video_frag_all, video_name, frames_info, metadata) in enumerate(tqdm(data_loader, desc="Processing Videos")):
            # slowfast features
            _, _, slowfast_frame_feats = extract_features_slowfast_pool(video_segments, model_slowfast, device)
            _, _, slowfast_res_frag_feats = extract_features_slowfast_pool(video_res_frag_all, model_slowfast, device)
            _, _, slowfast_frame_frag_feats = extract_features_slowfast_pool(video_frag_all, model_slowfast, device)
            slowfast_frame_feats_avg = slowfast_frame_feats.mean(dim=0)
            slowfast_res_frag_feats_avg = slowfast_res_frag_feats.mean(dim=0)
            slowfast_frame_frag_feats_avg = slowfast_frame_frag_feats.mean(dim=0)

            # swinT feature
            swint_frame_feats = extract_features_swint_pool(video_segments, model_swint, device)
            swint_res_frag_feats = extract_features_swint_pool(video_res_frag_all, model_swint, device)
            swint_frame_frag_feats = extract_features_swint_pool(video_frag_all, model_swint, device)
            swint_frame_feats_avg = swint_frame_feats.mean(dim=0)
            swint_res_frag_feats_avg = swint_res_frag_feats.mean(dim=0)
            swint_frame_frag_feats_avg = swint_frame_frag_feats.mean(dim=0)

            # semantic features
            image_embedding, quality_embedding, artifact_embedding = extract_features_clip_embed(frames_info, metadata, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device)
            image_embedding_avg = image_embedding.mean(dim=0)
            quality_embedding_avg = quality_embedding.mean(dim=0)
            artifact_embedding_avg = artifact_embedding.mean(dim=0)

            # frame + residual fragment + frame fragment features
            slowfast_features = torch.cat((slowfast_frame_feats_avg, slowfast_res_frag_feats_avg, slowfast_frame_frag_feats_avg), dim=0)
            swint_features = torch.cat((swint_frame_feats_avg, swint_res_frag_feats_avg, swint_frame_frag_feats_avg), dim=0)
            clip_features = torch.cat((image_embedding_avg, quality_embedding_avg, artifact_embedding_avg), dim=0)
            vqa_feats = torch.cat((slowfast_features, swint_features, clip_features), dim=0)

    vqa_feats = vqa_feats
    feature_tensor, _ = preprocess_data(vqa_feats, None)
    if feature_tensor.dim() == 1:
        feature_tensor = feature_tensor.unsqueeze(0)
    print(f"Feature tensor shape before MLP: {feature_tensor.shape}")

    model_mlp.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            prediction = model_mlp(feature_tensor)
            predicted_score = prediction.item()
            return predicted_score

def parse_framerate(framerate_str):
    num, den = framerate_str.split('/')
    framerate = float(num)/float(den)
    return framerate

def get_video_metadata(video_path):
    print(video_path)
    ffprobe_path = 'ffprobe'
    cmd = f'{ffprobe_path} -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate,bit_rate,bits_per_raw_sample,pix_fmt -of json {video_path}'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, check=True)
        info = json.loads(result.stdout)
    except Exception as e:
        print(f"Error processing file {video_path}: {e}")
        return {}

    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    bitrate = info['streams'][0].get('bit_rate', 0)
    bitdepth = info['streams'][0].get('bits_per_raw_sample', 0)
    framerate = info['streams'][0]['r_frame_rate']
    framerate = parse_framerate(framerate)
    return width, height, bitrate, bitdepth, framerate

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('--model_name', type=str, default='Mlp')
    parser.add_argument('--select_criteria', type=str, default='byrmse')
    parser.add_argument('--intra_cross_experiment', type=str, default='cross', help='intra or cross')
    parser.add_argument('--is_finetune', type=bool, default=True, help='True or False')
    parser.add_argument('--save_model_path', type=str, default='../model/')
    parser.add_argument('--prompt_path', type=str, default="./config/prompts.json")

    parser.add_argument('--train_data_name', type=str, default='lsvq_train', help='Name of the training data')
    parser.add_argument('--test_data_name', type=str, default='finevd', help='Name of the testing data')
    parser.add_argument('--test_video_path', type=str, default='../test_videos/0_16_07_500001604801190-yase.mp4', help='demo test video')
    parser.add_argument('--prediction_mode', type=float, default=50, help='default for inference')

    parser.add_argument('--network_name', type=str, default='camp-vqa')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--target_size', type=int, default=224)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = parse_arguments()
    device = setup_device(config)
    prompts = load_prompts(config.prompt_path)

    # test demo video
    resize_transform = get_transform(config.resize)
    top_n = int(config.target_size /config. patch_size) * int(config.target_size / config.patch_size)

    width, height, bitrate, bitdepth, framerate = get_video_metadata(config.test_video_path)

    data = {'vid': [os.path.splitext(os.path.basename(config.test_video_path))[0]],
        'test_data_name': [config.test_data_name],
        'test_video_path': [config.test_video_path],
        'prediction_mode': [config.prediction_mode],
        'width': [width], 'height': [height], 'bitrate': [bitrate], 'bitdepth': [bitdepth], 'framerate': [framerate]}
    videos_dir = os.path.dirname(config.test_video_path)
    test_df = pd.DataFrame(data)
    print(test_df.T)
    print(f"Experiment Setting: {config.intra_cross_experiment}, {config.train_data_name} -> {config.test_data_name}")
    if config.intra_cross_experiment == 'cross':
        if config.train_data_name == 'lsvq_train':
            print(f"Fine-tune: {config.is_finetune}")

    dataset = VideoDataset_feature(test_df, videos_dir, config.test_data_name, resize_transform, config.resize, config.patch_size, config.target_size, top_n)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=min(config.num_workers, os.cpu_count()), pin_memory=True
    )
    # print(f"Dataset loaded. Total videos: {len(dataset)}, Total batches: {len(data_loader)}")

    # load models to device
    model_slowfast = SlowFast().to(device)
    model_swint = SwinT(model_name='swin_large_patch4_window7_224', global_pool='avg', pretrained=True).to(device)

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

    input_features = 13056
    if config.train_data_name == 'lsvq_train':
        from model_regression_lsvq import Mlp
    else:
        from model_regression import Mlp
    model_mlp = load_model(config, device, Mlp, input_features)

    quality_prediction = evaluate_video_quality(data_loader, model_slowfast, model_swint, clip_model, clip_preprocess, blip_processor, blip_model, prompts, model_mlp, device)
    print("Predicted Quality Score:", quality_prediction)