import numpy as np
import os
import json
from numpy import dot
from numpy.linalg import norm

# 경로 설정
query_path = '/home/jejekim/VTimeLLM/feature_qvh/clip_text_features'
video_path = '/home/jejekim/VTimeLLM/feature_qvh/clip_features'
train_jsonl_path  = '/home/jejekim/VTimeLLM/data_qvh/highlight_train_release.jsonl'

# 코사인 유사도 계산 함수
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# 결과를 저장할 딕셔너리
cosine_similarities = {}
max_relevant_cos_sim = {}

# JSONL 파일을 한 줄씩 읽기
with open(train_jsonl_path, 'r') as f:
    for line in f:
        item = json.loads(line.strip())  # 한 줄을 JSON으로 변환
        qid = item['qid']
        video_filename = item['vid']  # vid에서 파일명만 추출
        relevant_clip_ids = item['relevant_clip_ids']  # relevant_clip_ids 가져오기
        duration = item['duration']  # duration 가져오기

        # 'qid' 접두어를 붙여 쿼리 파일명 생성
        query_feature_file = os.path.join(query_path, f'qid{qid}.npz')
        video_clip_feature_file = os.path.join(video_path, f'{video_filename}.npz')

        # 파일이 존재하는지 확인
        if not os.path.exists(query_feature_file) or not os.path.exists(video_clip_feature_file):
            print(f"Missing file for QID: {qid} or Video: {video_filename}")
            continue

        # .npz 파일 로드
        query_features = np.load(query_feature_file)
        video_clip_features = np.load(video_clip_feature_file)

        # pooler_output과 features를 가져옴
        query_feature = query_features['pooler_output'].astype(np.float32).reshape(1, -1)  # (1, 512)
        video_clip_features = video_clip_features['features'].astype(np.float32)          # (75, 512)

        # 각 프레임(75개)에 대해 코사인 유사도를 계산하여 리스트에 저장
        frame_similarities = []
        for j in range(video_clip_features.shape[0]):
            # 각 프레임에 대해 쿼리의 pooler_output과 유사도를 계산
            similarity = cosine_similarity(query_feature.flatten(), video_clip_features[j])
            # 소수점 4자리로 반올림하여 저장
            frame_similarities.append(round(float(similarity), 4))

        # 0~74까지 인덱스가 있는지 확인하고, 부족한 경우 0으로 채움
        while len(frame_similarities) < 75:
            frame_similarities.append(0.0)

        # 정규화: 최소값과 최대값 구하기
        min_sim = min(frame_similarities)
        max_sim = max(frame_similarities)

        # 모든 유사도를 0~1 사이로 정규화
        normalized_similarities = [
            (sim - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 0.0
            for sim in frame_similarities
        ]

        # 결과를 저장 (qid: 숫자 형식으로 저장)
        cosine_similarities[f"qid: {qid}"] = normalized_similarities

        # relevant_clip_ids에서 가장 높은 코사인 유사도와 그 인덱스를 찾기
        relevant_similarities = [(idx, normalized_similarities[idx]) for idx in relevant_clip_ids if idx < len(normalized_similarities)]
        if relevant_similarities:  # 리스트가 비어있지 않은지 확인
            max_index, max_similarity = max(relevant_similarities, key=lambda x: x[1])
            max_relevant_cos_sim[f"qid: {qid}"] = {
                "vid": video_filename,
                "duration": duration,  # duration 추가
                "max_similarity": max_similarity,
                "max_index": max_index
            }
            print(f'QID: {qid}, VID: {video_filename}, Duration: {duration}, Max Relevant Normalized Cosine Similarity: {max_similarity} at Index: {max_index}')
        else:
            print(f"No valid relevant clips found for QID: {qid}")

# 결과를 파일에 저장
with open('max_relevant_cos_sim.json', 'w') as f:
    json.dump(max_relevant_cos_sim, f, indent=4)
