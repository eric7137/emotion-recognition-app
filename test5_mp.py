import cv2
from deepface import DeepFace
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# 初始化 MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

text_obj = {
    'angry': '生氣', 'disgust': '噁心', 'fear': '害怕',
    'happy': '開心', 'sad': '難過', 'surprise': '驚訝', 'neutral': '正常'
}

def putText(x, y, text, img_in, size=70, color=(255, 255, 255)):
    fontpath = 'NotoSansTC-Bold.ttf'
    font = ImageFont.truetype(fontpath, size)
    imgPil = Image.fromarray(img_in)
    draw = ImageDraw.Draw(imgPil)
    draw.text((x, y), text, fill=color, font=font)
    return np.array(imgPil)

img = cv2.imread('emotions.jpg')
h_img, w_img, _ = img.shape # 取得圖片原始尺寸

# MediaPipe 需要 RGB 格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detection.process(img_rgb)

if results.detections:
    for detection in results.detections:
        # 取得 MediaPipe 的座標 (比例值 0~1)
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * w_img)
        y = int(bboxC.ymin * h_img)
        w = int(bboxC.width * w_img)
        h = int(bboxC.height * h_img)

        # 擴大偵測範圍 (DeepFace 辨識情緒需要多一點周邊資訊)
        x1, y1 = max(0, x-60), max(0, y-20)
        x2, y2 = min(w_img, x+w+60), min(h_img, y+h+60)
        
        face = img[y1:y2, x1:x2] 
        
        try:
            if face.size > 0:
                emotion = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                img = putText(x, y-10, text_obj[emotion[0]['dominant_emotion']], img) 
        except Exception as e:
            print(f"辨識出錯: {e}")

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

cv2.namedWindow('oxxostudio', cv2.WINDOW_NORMAL)
cv2.imshow('oxxostudio', img)
cv2.waitKey(0)
cv2.destroyAllWindows()