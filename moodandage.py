import cv2
from deepface import DeepFace
import numpy as np

img = cv2.imread('mona.jpg')
try:
    emotion = DeepFace.analyze(img, actions=['emotion'])  # 情緒
    age = DeepFace.analyze(img, actions=['age'])          # 年齡
    race = DeepFace.analyze(img, actions=['race'])        # 人種
    gender = DeepFace.analyze(img, actions=['gender'])    # 性別

    print(emotion[0]['dominant_emotion'])
    print(age[0]['age'])
    print(race[0]['dominant_race'])
    print(gender[0]['gender'])
except:
    pass

cv2.imshow('oxxostudio', img)
cv2.waitKey(0)
cv2.destroyAllWindows()