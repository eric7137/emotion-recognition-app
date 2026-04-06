import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

print(f"NumPy version: {np.__version__}")        # 應為 1.23.x
print(f"OpenCV version: {cv2.__version__}")      # 應為 4.8.x
print(f"TensorFlow version: {tf.__version__}")  # 應為 2.12.0
print(f"MediaPipe 模組測試: {mp.solutions.face_detection}") # 不應報錯