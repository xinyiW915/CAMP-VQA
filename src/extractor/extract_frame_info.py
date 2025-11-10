import os
import cv2
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils import data


class VideoDataset_feature(data.Dataset):
    def __init__(self, filename_path, data_dir, database, resize, patch_size=16, target_size=224, top_n=196):
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
        self.resize = resize
        self.patch_size = patch_size
        self.target_size = target_size
        self.top_n = top_n
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database in ['konvid_1k', 'test', 'live_vqc', 'youtube_ugc_h264', 'lsvq_test_1080p', 'lsvq_test', 'lsvq_train', 'live_yt_gaming', 'kvq', 'finevd', 'finevd_test']:
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'cvd_2014':
            video_name = str(self.video_names[idx]) + '.avi'
        elif self.database == 'youtube_ugc':
            video_name = str(self.video_names[idx]) + '.mkv'

        # get metadata
        row = self.dataInfo.iloc[idx]
        if 'subjective_score' in row and pd.notnull(row['subjective_score']):
            mos = float(row['subjective_score'])
        else:
            mos = float(row['mos'])
        metadata = (
            # float(row["mos"]),
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
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1:
            print(f"Invalid FPS={fps} for video {filename}. Using default")
            fps = 2.0
        frame_step = int(fps // 2)

        video_read_index = 0
        frames_info = []
        prev_frame = None
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame features

                # frame frag features
                if prev_frame is not None:
                    residual = cv2.absdiff(curr_frame, prev_frame)
                    diff = self.get_patch_diff(residual)
                    # frame residual fragment
                    imp_patches, positions = self.extract_important_patches(residual, diff)

                    # current frame fragment
                    ori_patches = self.get_original_frame_patches(curr_frame, positions)

                    # as a tuple
                    if i % frame_step == 0:
                        frames_info.append((curr_frame, imp_patches, ori_patches))
                    video_read_index += 1
            prev_frame = curr_frame
        video_capture.release()
        return video_name, frames_info, metadata

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

def visualise_image(frame, img_title='Residual Fragment', debug=True):
    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.axis('off')
        plt.title(img_title)
        plt.show()

def save_img(frame, fig_path, img_title):
    from torchvision.transforms import ToPILImage
    save_path = f'../../figs/{fig_path}/{img_title}.png'
    if isinstance(frame, torch.Tensor):
        if frame.dim() == 3 and frame.size(0) in [1, 3]:
            frame = ToPILImage()(frame)
        else:
            raise ValueError("Unsupported tensor shape. Expected shape (C, H, W) with C=1 or C=3.")

    if save_path:
        if isinstance(frame, torch.Tensor):
            frame = ToPILImage()(frame)
        elif isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frame.save(save_path)
        print(f"Image saved to {save_path}")


if __name__ == "__main__":
    database = 'test'
    videos_dir = '../../test_videos/'
    metadata_csv = '../../metadata/TEST_metadata.csv'
    start_time = time.time()

    dataset = VideoDataset_feature(
        filename_path=metadata_csv,
        data_dir=videos_dir,
        database=database,
        resize=224,  # 224, 384
        patch_size=16, # 8, 16, 32, 16, 32
        target_size=224, # 224, 224, 224, 384, 384
        top_n=14*14 # 28*28, 14*14, 7*7, 24*24, 12*12
    )

    # test
    index = 0
    video_name, frames_info, metadataa = dataset[index]
    print(f"Video Name: {video_name}")
    print(f"Total Key Frame Tuples (frames_info): {len(frames_info)}")
    # visualisation
    curr_frame, frag_residual, frag_frame = frames_info[0]
    visualise_image(curr_frame, 'Current Frame')
    visualise_image(frag_residual, 'Residual Fragment')
    visualise_image(frag_frame, 'Frame Fragment')

    print(f"Processed {time.time() - start_time:.2f} seconds")