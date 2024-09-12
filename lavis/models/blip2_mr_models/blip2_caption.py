import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import cv2
import numpy as np
from PIL import Image
import json
import os

# BLIP2 모델 및 프로세서 로드
model_name = "Salesforce/blip2-flan-t5-xl"  # BLIP2 모델 이름
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name)

def extract_frames(video_path, interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frames = []
    timestamps = range(0, int(duration), interval)

    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            break

    cap.release()
    return frames

def generate_captions(frames):
    captions = []
    images = [Image.fromarray(frame) for _, frame in frames]

    inputs = processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model.generate(**inputs)

    for (timestamp, _), output in zip(frames, outputs):
        caption = processor.decode(output, skip_special_tokens=True)
        captions.append({"time": timestamp, "caption": caption})
    
    return captions

def process_video(video_path, interval):
    frames = extract_frames(video_path, interval)
    if not frames:
        return []

    captions = generate_captions(frames)
    return captions

def main():
    folder_path = '/home/conda-user/workspace_mg/SeViLA/videos/qvhvideos/qv'  # 비디오 파일이 있는 폴더의 경로
    interval = 2  # 캡션 생성 간격 (초)
    video_extensions = ('.mp4')

    video_files = [f for f in os.listdir(folder_path) if f.endswith(video_extensions)]

    all_captions = {}

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f'Processing video: {video_path}')
        captions = process_video(video_path, interval)
        all_captions[video_file] = [cap["caption"] for cap in captions]

    # 모든 캡션을 JSON 파일로 저장
    output_file = 'all_captions.json'
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, 'w') as f:
        json.dump(all_captions, f, indent=4)

if __name__ == "__main__":
    main()
