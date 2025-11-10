import os
import cv2
import pandas as pd
import math

CSV_PATH = "/home/xinyi/Project/CAMP-VQA/metadata/finevd_test_csv/finevd_test_mos_overall_camp-vqa_predictScore.csv"
VIDEO_BASE = "/media/on23019/server1/video_dataset/FineVD/videos_all/videos_all_videos/"
OUT_BASE = "/home/xinyi/Project/CAMP-VQA/figs/captions_log/FINEVD_TEST_frames/"

def sample_frames_1fps(csv_path, video_base, out_base):
    os.makedirs(out_base, exist_ok=True)
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        vid = str(row["vid"])
        fps_from_csv = None
        if "framerate" in row and not pd.isna(row["framerate"]):
            try:
                fps_from_csv = float(row["framerate"])
            except Exception:
                fps_from_csv = None

        video_path = os.path.join(video_base, vid + ".mp4")
        if not os.path.exists(video_path):
            print(f"[skip] can not find the video: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[skip] failed to open video: {video_path}")
            continue

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        fps = fps_video if fps_video and fps_video > 1e-3 else (fps_from_csv if fps_from_csv and fps_from_csv > 1e-3 else 30.0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            print(f"[skip] invalid frame count: {video_path}")
            cap.release()
            continue

        duration = total_frames / fps  # second
        max_whole_sec = max(0, int(math.floor(max(0.0, duration - 1e-6))))
        seconds_to_sample = list(range(0, max_whole_sec + 1))

        out_dir = os.path.join(out_base, vid)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        last_frame_idx = -1
        for sec in seconds_to_sample:
            frame_idx = int(round(sec * fps))
            frame_idx = min(max(frame_idx, 0), total_frames - 1)

            # 避免重复 seek 到同一帧（例如极低 FPS 或四舍五入导致）
            if frame_idx == last_frame_idx:
                continue
            last_frame_idx = frame_idx

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            out_path = os.path.join(out_dir, f"{vid}_f{frame_idx}.png")
            cv2.imwrite(out_path, frame)
            saved += 1

        cap.release()
        print(f"{vid}: saved {saved} frames at 1 FPS -> {out_dir}")

if __name__ == "__main__":
    sample_frames_1fps(CSV_PATH, VIDEO_BASE, OUT_BASE)
