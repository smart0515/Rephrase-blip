import cv2
import json
import os

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def extract_frame_at_time(cap, time_in_seconds):
    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 초당 프레임 수 (FPS)
    frame_num = int(time_in_seconds * fps)  # 해당 초에 맞는 프레임 번호 계산
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # 해당 프레임으로 이동
    ret, frame = cap.read()  # 해당 프레임 읽기
    if ret:
        return frame
    return None

# 비디오 디렉토리 설정
video_dir = "/home/jejekim/Video-ChatGPT/videos"

# JSON 파일 경로
json_path = "/home/jejekim/mr-Blip/qvh_data/max_relevant_cos_sim.json"

# JSON 파일 읽기
with open(json_path, 'r') as f:
    data = json.load(f)

# 결과를 저장할 디렉토리
output_dir = "extracted_frames"
os.makedirs(output_dir, exist_ok=True)

# max_index를 시간으로 변환하는 함수
def max_index_to_time(max_index):
    start_time = max_index * 2  # 1 max_index는 2초를 나타냄
    end_time = start_time + 2
    middle_time = (start_time + end_time) / 2
    return middle_time

# 전체 데이터 수
total_videos = len(data)

# 처리한 항목 수 카운트
for idx, (qid, info) in enumerate(data.items(), 1):
    vid = info['vid']
    max_index = info['max_index']
    
    # max_index를 중간 시간으로 변환 (초 단위)
    target_time = max_index_to_time(max_index)
    
    # 비디오 파일 경로
    video_path = os.path.join(video_dir, f"{vid}.mp4")
    
    if os.path.exists(video_path):
        cap = load_video(video_path)
        
        # 중간 시간에 해당하는 프레임 추출
        target_frame = extract_frame_at_time(cap, target_time)
        
        if target_frame is not None:
            output_path = os.path.join(output_dir, f"frame_{qid.replace(':', '')}.jpg")
            cv2.imwrite(output_path, target_frame)
            print(f"프레임이 저장되었습니다.: {output_path}")
        else:
            print(f"{qid}: 프레임을 추출할 수 없습니다.")
        
        cap.release()
    else:
        print(f"{qid}: 비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # 진행도 출력
    progress = (idx / total_videos) * 100
    print(f"진행도: {progress:.2f}% ({idx}/{total_videos}) 완료")

print("모든 처리가 완료되었습니다.")
