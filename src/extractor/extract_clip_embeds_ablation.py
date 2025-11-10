import re
import torch
import clip
import numpy as np
from numpy.linalg import norm
from PIL import Image
import os
import json
from pathlib import Path

def get_quality_hint_from_metadata(mos, width, height, bitrate, bitdepth, framerate, quality_hints):
    hint = []
    if mos > 5:
        mos = (mos / 100) * 5
    if mos >= 4.5:
        hint.append(quality_hints["mos"]["excellent"])
    elif 3.5 <= mos < 4.5:
        hint.append(quality_hints["mos"]["good"])
    elif 2.5 <= mos < 3.5:
        hint.append(quality_hints["mos"]["fair"])
    elif 1.5 <= mos < 2.5:
        hint.append(quality_hints["mos"]["bad"])
    else:
        hint.append(quality_hints["mos"]["poor"])

    res = width * height
    if res < 640 * 480:
        hint.append(quality_hints["resolution"]["low"])
    elif res < 1280 * 720:
        hint.append(quality_hints["resolution"]["sd"])
    else:
        hint.append(quality_hints["resolution"]["hd"])
    if bitrate < 500_000:
        hint.append(quality_hints["bitrate"]["low"])
    elif bitrate < 1_000_000:
        hint.append(quality_hints["bitrate"]["medium"])
    else:
        hint.append(quality_hints["bitrate"]["high"])

    if 0 < bitdepth <= 8:
        hint.append(quality_hints["bitdepth"]["low"])
    elif bitdepth == 0:
        hint.append(quality_hints["bitdepth"]["standard"])
    else:
        hint.append(quality_hints["bitdepth"]["high"])
    if framerate < 24:
        hint.append(quality_hints["framerate"]["low"])
    elif framerate > 60:
        hint.append(quality_hints["framerate"]["high"])
    else:
        hint.append(quality_hints["framerate"]["standard"])
    return " ".join(hint)

def generate_caption(blip_processor, blip_model, device, image, prompt):
    inputs = blip_processor(image, prompt, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

def tensor_to_pil(image_tensor):
    if isinstance(image_tensor, torch.Tensor):
        arr = image_tensor.cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]  # remove batch dimension
        arr = arr.astype('uint8')
    return Image.fromarray(arr)

def extract_semantic_captions(blip_processor, blip_model, curr_frame, frag_residual, frag_frame, prompts, device, metadata=None, use_metadata_prompt=False):
    quality_prompt_base = prompts["quality_prompt_base"]
    content_prompt = prompts["curr_frame_content"]
    residual_prompt = prompts["residual_prompt"]
    frag_prompt = prompts["frag_prompt"]

    quality_hint = ""
    if use_metadata_prompt and metadata:
        mos, width, height, bitrate, bitdepth, framerate = metadata
        quality_hint = get_quality_hint_from_metadata(mos, width, height, bitrate, bitdepth, framerate, quality_hints=prompts["quality_hints"])

    prompt_hints = []
    if quality_hint:
        prompt_hints.append(quality_hint)

    quality_prompt = "\n\n".join(prompt_hints + [quality_prompt_base])
    fragment_prompt = "\n\n".join(prompt_hints)
    # print('content_prompt:', content_prompt)
    # print('quality_prompt:', quality_prompt)
    # print('residual_prompt:', fragment_prompt + "\n\n" + residual_prompt)
    # print('frame_fragment_prompt:', fragment_prompt + "\n\n" + frag_prompt)

    captions = {
        "curr_frame_content": generate_caption(blip_processor, blip_model, device, curr_frame, prompt=content_prompt),
        "curr_frame_quality": generate_caption(blip_processor, blip_model, device, curr_frame, prompt=quality_prompt),
        "frag_residual": generate_caption(blip_processor, blip_model, device, frag_residual, prompt=(fragment_prompt + "\n\n" + residual_prompt)),
        "frag_frame": generate_caption(blip_processor, blip_model, device, frag_frame, prompt=(fragment_prompt + "\n\n" + frag_prompt))
    }
    return captions

def clean_caption_text(text):
    text = re.sub(r"- .*?stock videos & royalty-free footage", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def dedup_keywords(text, split_tokens=[",", ".", ";"]):
    for token in split_tokens:
        text = text.replace(token, ",")
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    seen = set()
    unique_parts = []
    for part in parts:
        if part not in seen:
            unique_parts.append(part)
            seen.add(part)
    return " ".join(unique_parts)  # good for embedding

def get_clip_text_embedding(clip_model, device, text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze()

def get_clip_image_embedding(clip_model, clip_preprocess, device, image):
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze()

def extract_semantic_embeddings(clip_model, clip_preprocess, device, curr_frame, captions):
    if not isinstance(curr_frame, Image.Image):
        curr_frame = Image.fromarray(curr_frame)

    content_caption = dedup_keywords(clean_caption_text(captions["curr_frame_content"]))
    quality_caption = dedup_keywords(clean_caption_text(captions["curr_frame_quality"]))
    artifact_caption_1 = dedup_keywords(clean_caption_text(captions["frag_residual"]))
    artifact_caption_2 = dedup_keywords(clean_caption_text(captions["frag_frame"]))
    artifact_caption = dedup_keywords(f"{artifact_caption_1}, {artifact_caption_2}")

    # for save captions
    cleaned_texts = {
        "content_caption": content_caption,
        "quality_caption": quality_caption,
        "artifact_caption": artifact_caption,
    }

    image_embed = get_clip_image_embedding(clip_model, clip_preprocess, device, curr_frame)
    content_embed = get_clip_text_embedding(clip_model, device, content_caption)
    quality_embed = get_clip_text_embedding(clip_model, device, quality_caption)
    artifact_embed = get_clip_text_embedding(clip_model, device, artifact_caption)
    return image_embed, content_embed, quality_embed, artifact_embed, cleaned_texts

def extract_features_clip_embed(frames_info, metadata, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device):
    feature_image_embed = []
    feature_content_embed = []
    feature_quality_embed = []
    feature_artifact_embed = []
    for i, (curr_frame, frag_residual, frag_frame) in enumerate(frames_info):
        curr_frame = tensor_to_pil(curr_frame)
        frag_residual = tensor_to_pil(frag_residual)
        frag_frame = tensor_to_pil(frag_frame)

        captions = extract_semantic_captions(
            blip_processor, blip_model,
            curr_frame, frag_residual, frag_frame, prompts,
            device,
            metadata=metadata,
            use_metadata_prompt=True,
        )
        image_embed, content_embed, quality_embed, artifact_embed = extract_semantic_embeddings(clip_model, clip_preprocess, device, curr_frame, captions)
        feature_image_embed.append(image_embed)
        feature_content_embed.append(content_embed)
        feature_quality_embed.append(quality_embed)
        feature_artifact_embed.append(artifact_embed)

    # concatenate features
    image_embedding = torch.stack(feature_image_embed, dim=0)
    content_embedding = torch.stack(feature_content_embed, dim=0)
    quality_embedding = torch.stack(feature_quality_embed, dim=0)
    artifact_embedding = torch.stack(feature_artifact_embed, dim=0)
    print(image_embedding.shape, content_embedding.shape, quality_embedding.shape, artifact_embedding.shape)
    return image_embedding, content_embedding, quality_embedding, artifact_embedding

def extract_features_clip_embed_record(frames_info, metadata, clip_model, clip_preprocess, blip_processor, blip_model, prompts, device, save_captions_path, vid):
    feature_image_embed = []
    feature_content_embed = []
    feature_quality_embed = []
    feature_artifact_embed = []
    # for save captions
    captions_records = []

    for i, (curr_frame, frag_residual, frag_frame) in enumerate(frames_info):
        curr_frame = tensor_to_pil(curr_frame)
        frag_residual = tensor_to_pil(frag_residual)
        frag_frame = tensor_to_pil(frag_frame)

        # === save figs ===
        save_dir = f"../figs/captions_log/{vid}"
        os.makedirs(save_dir, exist_ok=True)
        curr_frame.save(os.path.join(save_dir, f"{vid}_frame{i}_curr.jpg"))
        frag_residual.save(os.path.join(save_dir, f"{vid}_frame{i}_residual.jpg"))
        frag_frame.save(os.path.join(save_dir, f"{vid}_frame{i}_frag.jpg"))

        captions = extract_semantic_captions(
            blip_processor, blip_model,
            curr_frame, frag_residual, frag_frame, prompts,
            device,
            metadata=metadata,
            use_metadata_prompt=True,
        )
        image_embed, content_embed, quality_embed, artifact_embed, cleaned_texts = extract_semantic_embeddings(clip_model, clip_preprocess, device, curr_frame, captions)

        feature_image_embed.append(image_embed)
        feature_content_embed.append(content_embed)
        feature_quality_embed.append(quality_embed)
        feature_artifact_embed.append(artifact_embed)

        # === caption per frame===
        record = {
            "vid": vid,
            "frame_index": i,
            "cleaned": cleaned_texts,  # {content_caption, quality_caption, artifact_caption}
            "raw": {
                "curr_frame_content": captions["curr_frame_content"],
                "curr_frame_quality": captions["curr_frame_quality"],
                "frag_residual": captions["frag_residual"],
                "frag_frame": captions["frag_frame"],
            }
        }
        captions_records.append(record)

    # concatenate features
    image_embedding = torch.stack(feature_image_embed, dim=0)
    content_embedding = torch.stack(feature_content_embed, dim=0)
    quality_embedding = torch.stack(feature_quality_embed, dim=0)
    artifact_embedding = torch.stack(feature_artifact_embed, dim=0)
    print(image_embedding.shape, content_embedding.shape, quality_embedding.shape, artifact_embedding.shape)

    # === save JSONL ===
    if save_captions_path:
        path = Path(save_captions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for rec in captions_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return image_embedding, content_embedding, quality_embedding, artifact_embedding, captions_records