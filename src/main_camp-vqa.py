import argparse
import os
import time
import json
import logging
import torch
from torchvision import transforms
from tqdm import tqdm
import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from extractor.extract_frag import VideoDataset_feature
from extractor.extract_clip_embeds import extract_features_clip_embed
from extractor.extract_slowfast_clip import SlowFast, extract_features_slowfast_pool
from extractor.extract_swint_clip import SwinT, extract_features_swint_pool
from utils.logger_setup import setup_logging

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    logging.info(f"Using device: {device}")
    return device

def load_prompts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_transform(resize):
    return transforms.Compose([transforms.Resize([resize, resize]),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

def load_dataset(database, resize_transform, resize, patch_size, target_size, top_n):
    if database == 'konvid_1k':
        videos_dir = '/media/on23019/server1/video_dataset/KoNViD_1k/KoNViD_1k_videos/'
        # videos_dir = 'D:/video_dataset/KoNViD_1k/KoNViD_1k_videos/'
    elif database == 'live_vqc':
        videos_dir = '/media/on23019/server1/video_dataset/LIVE-VQC/Video/'
    elif database == 'cvd_2014':
        videos_dir = '/media/on23019/server1/video_dataset/CVD2014/'
    elif database == 'youtube_ugc_h264':
        videos_dir = '/media/on23019/server1/video_dataset/youtube_ugc/all_h264/'
    elif database == 'live_yt_gaming':
        videos_dir = '/media/on23019/server1/video_dataset/LIVE-YT-Gaming/mp4/'
    elif database == 'kvq':
        videos_dir = '/media/on23019/server1/video_dataset/KVQ/'
    elif database == 'finevd' or database == 'finevd_test':
        videos_dir = '/media/on23019/server1/video_dataset/FineVD/videos_all/videos_all_videos/'
    elif database == 'lsvq_test_1080p' or database =='lsvq_test' or database == 'lsvq_train':
        videos_dir = '/media/on23019/server1/LSVQ/'
    elif database == 'test':
        videos_dir = '../test_videos/'
    metadata_csv = f'../metadata/{database.upper()}_metadata.csv'
    # split test: temp
    # metadata_csv = f'../metadata/{database.upper()}_metadata_part5.csv'
    print(metadata_csv)

    return VideoDataset_feature(metadata_csv, videos_dir, database, resize_transform, resize, patch_size, target_size, top_n)

def process_videos(data_loader, model_slowfast, model_swint, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device):
    slowfast_embs_list = []
    swint_embs_list = []
    semantic_embs_list = []
    features_list = []
    model_slowfast.eval()
    model_swint.eval()

    with torch.no_grad():
        for i, (video_segments, video_res_frag_all, video_frag_all, video_name, frames_info, metadata) in enumerate(tqdm(data_loader, desc="Processing Videos")):
            start_time = time.time()
            try:
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
                logging.info(f"Slowfast Feature shape: {slowfast_features.shape}")
                logging.info(f"SwinT Feature shape: {swint_features.shape}")
                logging.info(f"Semantic CLIP Feature shape: {clip_features.shape}")
                slowfast_embs_list.append(slowfast_features)
                swint_embs_list.append(swint_features)
                semantic_embs_list.append(clip_features)

                vqa_feats = torch.cat((slowfast_features, swint_features, clip_features), dim=0)
                logging.info(f"VQA Feature shape: {vqa_feats.shape}")
                features_list.append(vqa_feats)

                logging.info(f"Processed {video_name[0]} in {time.time() - start_time:.2f} seconds")
                torch.cuda.empty_cache()

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    logging.error("Processing interrupted by user.")
                else:
                    logging.error(f"Failed to process video {video_name[0]}: {e}")

    return slowfast_embs_list, swint_embs_list, semantic_embs_list, features_list

def save_features(features_list, pt_path):
    if features_list:
        features_tensor = torch.stack(features_list)
        try:
            torch.save(features_tensor, f"{pt_path}")

        except Exception as e:
            print(f"Failed to save features: {e}")
        logging.info(f"Features saved to {pt_path}: {features_tensor.shape}\n")
    else:
        logging.warning("No features processed. Nothing to save.")

def main(config):
    feature_save_path = os.path.abspath(os.path.join(config.feature_save_path, config.feat_name))

    slowfast_pt_path = f'{feature_save_path}/slowfast_{config.database}_features.pt'
    swint_pt_path = f'{feature_save_path}/swint_{config.database}_features.pt'
    semantic_pt_path = f'{feature_save_path}/semantic_embs_{config.database}_features.pt'
    vqa_pt_path = f'{feature_save_path}/{config.feat_name}_{config.database}_features.pt'
    # split test: temp
    # slowfast_pt_path = f'{feature_save_path}/slowfast_{config.database}_features_part5.pt'
    # swint_pt_path = f'{feature_save_path}/swint_{config.database}_features_part5.pt'
    # semantic_pt_path = f'{feature_save_path}/semantic_embs_{config.database}_features_part5.pt'
    # vqa_pt_path = f'{feature_save_path}/{config.feat_name}_{config.database}_features_part5.pt'
    print(vqa_pt_path)

    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)

    prompts = load_prompts(config.prompt_path)

    setup_logging(config.log_file)
    device = setup_device()
    resize_transform = get_transform(config.resize)
    dataset = load_dataset(config.database, resize_transform, config.resize, config.patch_size, config.target_size, config.top_n)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=min(config.num_workers, os.cpu_count()), pin_memory=True
    )
    logging.info(f"Dataset loaded. Total videos: {len(dataset)}, Total batches: {len(data_loader)}")

    model_slowfast = SlowFast().to(device)
    model_swint = SwinT(model_name='swin_large_patch4_window7_224', global_pool='avg', pretrained=True).to(device) # swin_large_patch4_window7_224.ms_in22k_ft_in1k

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

    slowfast_embs_list, swint_embs_list, semantic_embs_list, features_list = process_videos(data_loader, model_slowfast, model_swint, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device)
    save_features(slowfast_embs_list, slowfast_pt_path)
    save_features(swint_embs_list, swint_pt_path)
    save_features(semantic_embs_list, semantic_pt_path)
    save_features(features_list, vqa_pt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='konvid_1k')
    parser.add_argument('--feat_name', type=str, default='camp-vqa')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--top_n', type=int, default=14*14)
    parser.add_argument('--feature_save_path', type=str, default=f"../features")
    parser.add_argument('--prompt_path', type=str, default="./config/prompts.json")
    parser.add_argument('--log_file', type=str, default="./utils/logging_vqa_feats.log")


    config = parser.parse_args()
    print(config.feat_name)
    main(config)
