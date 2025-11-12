import os
import cv2
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils import data


class VideoDataset_feature(data.Dataset):
    def __init__(self, filename_path, data_dir, database, transform, resize, patch_size=16, target_size=224, top_n=196):
        super(VideoDataset_feature, self).__init__()
        if isinstance(filename_path, str):
            self.dataInfo = pd.read_csv(filename_path)
        elif isinstance(filename_path, pd.DataFrame):
            self.dataInfo = filename_path
        else:
            raise ValueError("filename_path: CSV file or DataFrame")
        self.video_names = self.dataInfo['vid'].tolist()
        self.videos_dir = data_dir
        self.database = database
        self.transform = transform
        self.resize = resize
        self.patch_size = patch_size
        self.target_size = target_size
        self.top_n = top_n
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database in ['konvid_1k', 'test']:
            video_clip_min = 8
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'live_vqc':
            video_clip_min = 10
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'cvd_2014':
            video_clip_min = 12
            video_name = str(self.video_names[idx]) + '.avi'
        elif self.database == 'youtube_ugc':
            video_clip_min = 20
            video_name = str(self.video_names[idx]) + '.mkv'
        elif self.database == 'youtube_ugc_h264':
            video_clip_min = 20
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'live_yt_gaming':
            video_clip_min = 7
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'kvq':
            video_clip_min = 8
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database in ['finevd', 'finevd_test']:
            video_clip_min = 8
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database in ['lsvq_test_1080p', 'lsvq_test', 'lsvq_train']:
            video_clip_min = 8
            video_name = str(self.video_names[idx]) + '.mp4'

        # get metadata
        row = self.dataInfo.iloc[idx]
        if 'prediction_mode' in row and pd.notnull(row['prediction_mode']):
            mos = float(row['prediction_mode'])
        else:
            mos = float(row['mos'])
        metadata = (
            mos,
            int(row["width"]),
            int(row["height"]),
            int(row["bitrate"]),
            int(row["bitdepth"]),
            float(row["framerate"])
        )
        filename = os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture(filename)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        if not video_capture.isOpened():
            raise RuntimeError(f"Failed to open video: {filename}")

        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1:
            print(f"Invalid FPS={fps} for video {filename}. Using default")
            fps = 2.0
        frame_step = int(fps // 2)
        video_clip = int(video_length / video_frame_rate) if video_frame_rate != 0 else 10
        video_channel = 3
        video_length_clip = 32

        all_frame_tensor = torch.zeros((video_length, video_channel, self.resize, self.resize), dtype=torch.float32)
        all_residual_frag_tensor = torch.zeros((video_length - 1, video_channel, self.resize, self.resize), dtype=torch.float32)
        all_frame_frag_tensor = torch.zeros((video_length - 1, video_channel, self.resize, self.resize), dtype=torch.float32)

        video_read_index = 0
        frames_info = []
        prev_frame = None
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame features
                curr_frame_tensor = self.transform(Image.fromarray(curr_frame))
                all_frame_tensor[video_read_index] = curr_frame_tensor

                # frame frag features
                if prev_frame is not None:
                    residual = cv2.absdiff(curr_frame, prev_frame)
                    diff = self.get_patch_diff(residual)
                    # frame residual fragment
                    imp_patches, positions = self.extract_important_patches(residual, diff)
                    imp_patches_pil = Image.fromarray(imp_patches.astype('uint8'))
                    residual_frag_tensor = self.transform(imp_patches_pil)
                    all_residual_frag_tensor[video_read_index] = residual_frag_tensor

                    # current frame fragment
                    ori_patches = self.get_original_frame_patches(curr_frame, positions)
                    ori_patches_pil = Image.fromarray(ori_patches.astype('uint8'))
                    frame_frag_tensor = self.transform(ori_patches_pil)
                    all_frame_frag_tensor[video_read_index] = frame_frag_tensor

                    # as a tuple
                    if i % frame_step == 0:
                        frames_info.append((curr_frame, imp_patches, ori_patches))
                    video_read_index += 1
            prev_frame = curr_frame
        video_capture.release()

        # Unfilled frames
        self.fill_tensor(all_frame_tensor, video_read_index, video_length)
        self.fill_tensor(all_residual_frag_tensor, video_read_index, video_length - 1)
        self.fill_tensor(all_frame_frag_tensor, video_read_index, video_length - 1)

        video_all = []
        video_res_frag_all = []
        video_frag_all = []
        for i in range(video_clip):
            clip_tensor = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            clip_res_frag_tensor = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            clip_frag_tensor = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])

            start_idx = i * video_frame_rate
            end_idx = start_idx + video_length_clip
            # frame features
            if end_idx <= video_length:
                clip_tensor = all_frame_tensor[start_idx:end_idx]
            else:
                clip_tensor[:(video_length - start_idx)] = all_frame_tensor[start_idx:]
                clip_tensor[(video_length - start_idx):video_length_clip] = clip_tensor[video_length - start_idx - 1]

            # frame frag features
            if end_idx <= (video_length - 1):
                clip_res_frag_tensor = all_residual_frag_tensor[start_idx:end_idx]
                clip_frag_tensor = all_frame_frag_tensor[start_idx:end_idx]
            else:
                clip_res_frag_tensor[:(video_length - 1 - start_idx)] = all_residual_frag_tensor[start_idx:]
                clip_frag_tensor[:(video_length - 1 - start_idx)] = all_frame_frag_tensor[start_idx:]
                clip_res_frag_tensor[(video_length - 1 - start_idx):video_length_clip] = clip_res_frag_tensor[video_length - 1 - start_idx - 1]
                clip_frag_tensor[(video_length - 1 - start_idx):video_length_clip] = clip_frag_tensor[video_length - 1 - start_idx - 1]

            video_all.append(clip_tensor)
            video_res_frag_all.append(clip_res_frag_tensor)
            video_frag_all.append(clip_frag_tensor)

        # Underfilling of clips
        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                video_all.append(video_all[video_clip - 1])
                video_res_frag_all.append(video_res_frag_all[video_clip - 1])
                video_frag_all.append(video_frag_all[video_clip - 1])
        return video_all, video_res_frag_all, video_frag_all, video_name, frames_info, metadata

    @staticmethod
    # duplicat the final frames
    def fill_tensor(tensor, read_index, length):
        if read_index < length:
            tensor[read_index:length] = tensor[read_index - 1]

    def get_patch_diff(self, residual_frame):
        h, w = residual_frame.shape[:2]
        patch_size = self.patch_size
        h_adj = (h // patch_size) * patch_size
        w_adj = (w // patch_size) * patch_size
        residual_frame_adj = residual_frame[:h_adj, :w_adj]
        # calculate absolute patch difference
        diff = np.zeros((h_adj // patch_size, w_adj // patch_size))
        for i in range(0, h_adj, patch_size):
            for j in range(0, w_adj, patch_size):
                patch = residual_frame_adj[i:i+patch_size, j:j+patch_size]
                # absolute sum
                diff[i // patch_size, j // patch_size] = np.sum(np.abs(patch))
        return diff

    def extract_important_patches(self, residual_frame, diff):
        patch_size = self.patch_size
        target_size = self.target_size
        top_n = self.top_n

        # find top n patches indices
        patch_idx = np.unravel_index(np.argsort(-diff.ravel()), diff.shape)
        top_patches = list(zip(patch_idx[0][:top_n], patch_idx[1][:top_n]))
        sorted_idx = sorted(top_patches, key=lambda x: (x[0], x[1]))

        imp_patches_img = np.zeros((target_size, target_size, residual_frame.shape[2]), dtype=residual_frame.dtype)
        patches_per_row = target_size // patch_size  # 14
        # order the patch in the original location relation
        positions = []
        for idx, (y, x) in enumerate(sorted_idx):
            patch = residual_frame[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size]
            # new patch location
            row_idx = idx // patches_per_row
            col_idx = idx % patches_per_row
            start_y = row_idx * patch_size
            start_x = col_idx * patch_size
            imp_patches_img[start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
            positions.append((y, x))
        return imp_patches_img, positions

    def get_original_frame_patches(self, original_frame, positions):
        patch_size = self.patch_size
        target_size = self.target_size
        imp_original_patches_img = np.zeros((target_size, target_size, original_frame.shape[2]), dtype=original_frame.dtype)
        patches_per_row = target_size // patch_size

        for idx, (y, x) in enumerate(positions):
            start_y = y * patch_size
            start_x = x * patch_size
            end_y = start_y + patch_size
            end_x = start_x + patch_size

            patch = original_frame[start_y:end_y, start_x:end_x]
            row_idx = idx // patches_per_row
            col_idx = idx % patches_per_row
            target_start_y = row_idx * patch_size
            target_start_x = col_idx * patch_size

            imp_original_patches_img[target_start_y:target_start_y + patch_size,
                                     target_start_x:target_start_x + patch_size] = patch
        return imp_original_patches_img

def visualise_tensor(tensors, num_frames_to_visualise=5, img_title='Frag'):
    np_feat = tensors.numpy()
    fig, axes = plt.subplots(1, num_frames_to_visualise, figsize=(15, 5))
    for i in range(num_frames_to_visualise):
        # move channels to last dimension for visualisation: (height, width, channels)
        frame = np_feat[i].transpose(1, 2, 0)
        # normalize to [0, 1] for visualisation
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        axes[i].imshow(frame)
        axes[i].axis('off')
        axes[i].set_title(f'{img_title} {i + 1}')

    plt.tight_layout()
    plt.show()

def visualise_image(frame, img_title='Residual Fragment', debug=True):
    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.axis('off')
        plt.title(img_title)
        plt.show()

if __name__ == "__main__":
    database = 'test'
    videos_dir = '../../test_videos/'
    metadata_csv = '../../metadata/TEST_metadata.csv'
    resize = 224
    patch_size = 16
    target_size = 224
    top_n = 14 * 14
    start_time = time.time()
    resize_transform = transforms.Compose([
        transforms.Resize([resize, resize]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    dataset = VideoDataset_feature(
        filename_path=metadata_csv,
        data_dir=videos_dir,
        database=database,
        transform=resize_transform,
        resize=resize,
        patch_size=patch_size,
        target_size=target_size,
        top_n=top_n
    )

    # test
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    start_time = time.time()
    index = 0
    video_segments, video_res_frag_all, video_frag_all, video_name, frames_info, metadata = dataset[index]

    print(f"\n=== Video Processed ===")
    print(f"Video Name: {video_name}")
    print(f"Metadata: [MOS, width, height, bitrate, bitdepth, framerate] = {metadata}")
    print(f"Number of Segments: {len(video_segments)}")
    print(f"Number of Video Residual Fragment Segments: {len(video_res_frag_all)}")
    print(f"Number of Video Fragment Segments: {len(video_frag_all)}")
    print(f"Shape of Each Segment: {video_segments[0].shape}")  # (video_length_clip, channels, height, width)
    print(f"Shape of Each Residual Fragment Segments: {video_res_frag_all[0].shape}")
    print(f"Shape of Each Fragment Segments: {video_frag_all[0].shape}")
    print(f"Total Key Frame Tuples (frames_info): {len(frames_info)}")
    curr_frame, imp_patch, ori_patch = frames_info[0]
    print("curr_frame shape:", np.array(curr_frame).shape)
    print("imp_patch shape:", np.array(imp_patch).shape)
    print("ori_patch shape:", np.array(ori_patch).shape)

    # visualisation
    curr_frame, frag_residual, frag_frame = frames_info[0]
    visualise_image(curr_frame, 'Current Frame')
    visualise_image(frag_residual, 'Residual Fragment')
    visualise_image(frag_frame, 'Frame Fragment')

    visualise_tensor(video_segments[0], num_frames_to_visualise=5, img_title='Frame')
    visualise_tensor(video_res_frag_all[0], num_frames_to_visualise=5, img_title='Residual Frag')
    visualise_tensor(video_frag_all[0], num_frames_to_visualise=5, img_title='Frame Frag')
    print(f"\nTotal Time: {time.time() - start_time:.2f} seconds")