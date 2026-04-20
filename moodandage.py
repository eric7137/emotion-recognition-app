# === DeepFace 人臉辨識 完整版 ===
# 環境：Google Colab（CPU 即可，GPU 更快）

# Step 1: 安裝
#!pip install deepface -q

# Step 2: 匯入
from deepface import DeepFace
import matplotlib.pyplot as plt
import urllib.request
import cv2
import os

# # Step 3: 下載測試圖片
# os.makedirs("faces", exist_ok=True)
# urls = {
#     "face_a1.jpg": "https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img1.jpg",
#     "face_a2.jpg": "https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img2.jpg",
#     "face_b1.jpg": "https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img3.jpg",
# }
# for fname, url in urls.items():
#     urllib.request.urlretrieve(url, os.path.join("faces", fname))

# Step 4: 人臉驗證
# result = DeepFace.verify("faces/face_a1.jpg", "faces/face_a2.jpg", model_name="Facenet512")
# print(f"同一人: {result['verified']}, 距離: {result['distance']:.4f}")

# Step 5: 屬性分析
analysis = DeepFace.analyze("faces/face_a2.jpg", actions=["age", "gender", "emotion"], detector_backend="mediapipe")
print(f"年齡: {analysis[0]['age']}, 性別: {analysis[0]['dominant_gender']}, 情緒: {analysis[0]['dominant_emotion']}, 情緒分佈: {analysis[0]['emotion']}")

# Step 6: 嵌入向量
embedding = DeepFace.represent("faces/face_a1.jpg", model_name="Facenet512")
print(f"向量維度: {len(embedding[0]['embedding'])}")