import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import normalize
import umap
import hdbscan
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image

# 설정
BATCH_SIZE = 32
SEED = 42

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 및 프로세서 로드
clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
clip_model.to(device)
clip_processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

# 데이터 경로
train_image_dir = '/home/work/jisoo/train_gt/'
test_image_dir = '/home/work/jisoo/test_input/'
preproc_dir = '/home/work/jisoo/preproc_1/'

# 결과 저장 폴더 생성
os.makedirs(preproc_dir, exist_ok=True)

# 학습 이미지 경로 로드
image_paths = sorted(glob(os.path.join(train_image_dir, '*.png')))
print(f"총 {len(image_paths)}개의 학습 이미지를 로드했습니다.")

# 이미지 임베딩 생성
image_features = []
for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting Features"):
    image_paths_batch = image_paths[i:i + BATCH_SIZE]
    images = [Image.open(image_path) for image_path in image_paths_batch]
    pixel_values = clip_processor.image_processor(images=images, return_tensors='pt')['pixel_values'].to(device)
    with torch.no_grad():
        image_features_row = clip_model.get_image_features(pixel_values).cpu().numpy()
    image_features.append(image_features_row)

# 임베딩 병합 및 정규화
train_embeddings = np.vstack(image_features)
train_embeddings = normalize(train_embeddings, norm="l2")
np.save(os.path.join(preproc_dir, 'train_embeddings.npy'), train_embeddings)
print(f"학습 데이터 임베딩이 {os.path.join(preproc_dir, 'train_embeddings.npy')}에 저장되었습니다.")

# 차원 축소 (UMAP)
clusterable_embedding = umap.UMAP(
    n_neighbors=5,
    min_dist=0.0,
    n_components=2,
    random_state=SEED,
).fit_transform(train_embeddings)

# 시각화 (선택 사항)
plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], s=0.1)
plt.title("UMAP Reduced Embedding")
plt.show()

# 군집화 (HDBSCAN)
labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(clusterable_embedding)

# 군집 결과 시각화
plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=labels, s=0.1, cmap='Spectral')
plt.title("HDBSCAN Clustering")
plt.show()

# 군집 결과 요약
print(f"클러스터 개수: {len(set(labels))} (노이즈 포함)")
print(f"노이즈 데이터 개수: {sum(labels == -1)}")

# 클러스터 내 데이터 통계
counter = Counter([label for label in labels if label != -1])
print(f"최소 클러스터 크기: {min(counter.values())}")
print(f"중앙값 클러스터 크기: {np.median(list(counter.values()))}")
print(f"최대 클러스터 크기: {max(counter.values())}")

# 학습 데이터 DataFrame 생성 및 저장
train_df = pd.DataFrame(columns=['image', 'label'])
train_df['image'] = [os.path.basename(image_path) for image_path in image_paths]
train_df['label'] = labels
train_preproc_path = os.path.join(preproc_dir, 'train_preproc_1.csv')
train_df.to_csv(train_preproc_path, index=False)
print(f"학습 데이터 전처리 결과가 {train_preproc_path}에 저장되었습니다.")

# 테스트 데이터 DataFrame 생성 및 저장
test_image_paths = sorted(glob(os.path.join(test_image_dir, '*.png')))
test_df = pd.DataFrame(columns=['image'])
test_df['image'] = [os.path.basename(image_path) for image_path in test_image_paths]
test_preproc_path = os.path.join(preproc_dir, 'test_preproc_1.csv')
test_df.to_csv(test_preproc_path, index=False)
print(f"테스트 데이터 전처리 결과가 {test_preproc_path}에 저장되었습니다.")
