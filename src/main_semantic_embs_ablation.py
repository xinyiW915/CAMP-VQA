import argparse
import os
import time
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from extractor.extract_frame_info import VideoDataset_feature
from extractor.extract_clip_embeds_ablation import extract_features_clip_embed, extract_features_clip_embed_record
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

def load_dataset(database, resize, patch_size, target_size, top_n):
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
    print(metadata_csv)

    return VideoDataset_feature(metadata_csv, videos_dir, database, resize, patch_size, target_size, top_n)


def process_videos(data_loader, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device):
    img_embs_list = []
    content_embs_list = []
    quality_embs_list = []
    artifact_embs_list = []
    semantic_embs_list = []
    semantic_embs_list_w_content = []

    with torch.no_grad():
        for i, (video_name, frames_info, metadata) in enumerate(tqdm(data_loader, desc="Processing Videos")):
            start_time = time.time()
            try:
                # for getting embeddings
                # image_embedding, content_embedding, quality_embedding, artifact_embedding = extract_features_clip_embed(frames_info, metadata, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device)

                # for save captions visulisation
                image_embedding, content_embedding, quality_embedding, artifact_embedding, captions_records = \
                    extract_features_clip_embed_record(frames_info, metadata, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device,
                        save_captions_path=f"../figs/captions_log/{video_name[0].replace('.mp4', '')}.jsonl",
                        vid=video_name[0].replace('.mp4', '')
                    )

                image_embedding_avg = image_embedding.mean(dim=0)
                content_embedding_avg = content_embedding.mean(dim=0)
                quality_embedding_avg = quality_embedding.mean(dim=0)
                artifact_embedding_avg = artifact_embedding.mean(dim=0)
                clip_features = torch.cat((image_embedding_avg, quality_embedding_avg, artifact_embedding_avg), dim=0)
                clip_features_w_content = torch.cat((image_embedding_avg, content_embedding_avg, quality_embedding_avg, artifact_embedding_avg), dim=0)

                logging.info(f"Semantic Feature shape: {clip_features.shape}")
                logging.info(f"Semantic Feature with content shape: {clip_features_w_content.shape}")
                img_embs_list.append(image_embedding_avg)
                content_embs_list.append(content_embedding_avg)
                quality_embs_list.append(quality_embedding_avg)
                artifact_embs_list.append(artifact_embedding_avg)
                semantic_embs_list.append(clip_features)
                semantic_embs_list_w_content.append(clip_features_w_content)

                logging.info(f"Processed {video_name[0]} in {time.time() - start_time:.2f} seconds")
                torch.cuda.empty_cache()

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    logging.error("Processing interrupted by user.")
                else:
                    logging.error(f"Failed to process video {video_name[0]}: {e}")

    return img_embs_list, content_embs_list, quality_embs_list, artifact_embs_list, semantic_embs_list, semantic_embs_list_w_content

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
    feature_save_path = os.path.abspath(os.path.join(config.feature_save_path))
    img_embs_pt_path = f'{feature_save_path}/img_embs_{config.database}_features.pt'
    content_embs_pt_path = f'{feature_save_path}/content_embs_{config.database}_features.pt'
    quality_embs_pt_path = f'{feature_save_path}/quality_embs_{config.database}_features.pt'
    artifact_embs_pt_path = f'{feature_save_path}/artifact_embs_{config.database}_features.pt'
    semantic_pt_path = f'{feature_save_path}/{config.feat_name}_{config.database}_features.pt'
    semantic_w_content_pt_path = f'{feature_save_path}/{config.feat_name}_w_content_{config.database}_features.pt'
    print(semantic_pt_path)
    print(semantic_w_content_pt_path)

    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)

    prompts = load_prompts(config.prompt_path)

    setup_logging(config.log_file)
    device = setup_device()
    dataset = load_dataset(config.database, config.resize, config.patch_size, config.target_size, config.top_n)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=min(config.num_workers, os.cpu_count()), pin_memory=True
    )
    logging.info(f"Dataset loaded. Total videos: {len(dataset)}, Total batches: {len(data_loader)}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

    img_embs_list, content_embs_list, quality_embs_list, artifact_embs_list, semantic_embs_list, semantic_embs_list_w_content = process_videos(data_loader, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device)
    save_features(img_embs_list, img_embs_pt_path)
    save_features(content_embs_list, content_embs_pt_path)
    save_features(quality_embs_list, quality_embs_pt_path)
    save_features(artifact_embs_list, artifact_embs_pt_path)
    save_features(semantic_embs_list, semantic_pt_path)
    save_features(semantic_embs_list_w_content, semantic_w_content_pt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='konvid_1k')
    parser.add_argument('--feat_name', type=str, default='semantic_embs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--top_n', type=int, default=14*14)
    parser.add_argument('--feature_save_path', type=str, default=f"../features/semantic_embs/")
    parser.add_argument('--prompt_path', type=str, default="./config/prompts.json")
    parser.add_argument('--log_file', type=str, default="./utils/logging_feats.log")


    config = parser.parse_args()
    print(config.feat_name)
    main(config)
