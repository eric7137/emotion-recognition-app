import cv2
from deepface import DeepFace
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp
import os

# 初始化 MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8)

text_obj = {
    'angry': '生氣', 'disgust': '噁心', 'fear': '害怕',
    'happy': '開心', 'sad': '難過', 'surprise': '驚訝', 'neutral': '正常'
}

def putText(x, y, text, img_in, size=50, color=(255, 255, 0)):
    fontpath = 'NotoSansTC-Bold.ttf'
    # 檢查字體是否存在
    if not os.path.exists(fontpath):
        # 如果沒字體，改用 OpenCV 內建英文，避免程式崩潰
        cv2.putText(img_in, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img_in
    
    font = ImageFont.truetype(fontpath, size)
    imgPil = Image.fromarray(cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(imgPil)
    draw.text((x, y+50), text, fill=color, font=font)
    return cv2.cvtColor(np.array(imgPil), cv2.COLOR_RGB2BGR)

# 讀取圖片
img = cv2.imread('bad.jpg')
if img is None:
    print("找不到圖片！")
    exit()

h_img, w_img, _ = img.shape

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detection.process(img_rgb)

if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        # 轉換為整數座標
        x = int(bboxC.xmin * w_img)
        y = int(bboxC.ymin * h_img)
        w = int(bboxC.width * w_img)
        h = int(bboxC.height * h_img)

        # 安全擴大範圍 (確保不超出圖片邊界)
        x1, y1 = max(0, x-40), max(0, y-40)
        x2, y2 = min(w_img, x+w+40), min(h_img, y+h+40)
        
        # 取得人臉切片
        face = img[y1:y2, x1:x2] 
        
        if face.size > 0:
            try:
                # 使用 skip 或 retinaface，但既然我們已經切好了臉，用 skip 最快
                # 如果要準，detector_backend 可維持 'retinaface'
                emotion_res = DeepFace.analyze(face, actions=['emotion'], 
                                             enforce_detection=False, 
                                             detector_backend='skip', 
                                             align=True)
                
                label = text_obj.get(emotion_res[0]['dominant_emotion'], '未知')
                img = putText(x, y-60, label, img) 
                
            except Exception as e:
                print(f"DeepFace 辨識出錯: {e}")

        # 畫框
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

cv2.namedWindow('oxxostudio', cv2.WINDOW_NORMAL)
cv2.imshow('oxxostudio', img)
cv2.waitKey(0)
cv2.destroyAllWindows()