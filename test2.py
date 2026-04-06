import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 只顯示 Fatal 錯誤
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
from deepface import DeepFace

def run_deepface_analysis(img_path):
    # 讀取圖片
    img = cv2.imread(img_path)
    if img is None:
        print("無法讀取圖片")
        return

    try:
        # 執行分析
        # actions 包含年齡與情緒
        # enforce_detection=False 避免沒偵測到臉時報錯
        results = DeepFace.analyze(img, actions=['age', 'emotion'], enforce_detection=False)

        # DeepFace 回傳的是 list (支援多張臉)
        for face in results:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            # 取得資料
            age = int(face['age'])
            emotion = face['dominant_emotion']
            
            # 1. 畫出人臉方框 (綠色)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 2. 在左上角顯示文字
            label = f"{emotion}, {age}y"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 顯示成果
        cv2.imshow("DeepFace Analysis", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"分析失敗: {e}")

if __name__ == "__main__":
    run_deepface_analysis("1234.jpg")