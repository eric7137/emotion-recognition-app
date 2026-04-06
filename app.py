import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from PIL import Image
import pandas as pd

# --- 設定網頁標題 ---
st.set_page_config(page_title="AI 人臉情緒辨識系統", layout="wide")
st.title("😊 AI 人臉情緒辨識分析儀")
st.write("上傳一張照片，AI 將自動偵測人臉並分析情緒，最後生成數據表格。")

# --- 初始化 MediaPipe ---
@st.cache_resource # 使用快取避免重複載入模型
def load_models():
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detector

face_detection = load_models()

# --- 定義情緒中文化 ---
text_obj = {
    'angry': '😡 生氣', 'disgust': '🤢 噁心', 'fear': '😨 害怕',
    'happy': '😄 開心', 'sad': '😢 難過', 'surprise': '😲 驚訝', 'neutral': '😐 正常'
}

# --- 網頁側邊欄：上傳檔案 ---
uploaded_file = st.sidebar.file_uploader("選擇一張照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape

    # 執行 MediaPipe 偵測
    results = face_detection.process(img_rgb)
    
    analysis_data = [] # 用來儲存表格資料

    analysis_data = [] 
    # --- 網頁呈現部分 ---
    col1, col2 = st.columns([2, 1]) # 比例 2:1 分欄
    placeholder_img = col1.empty()   # 為圖片建立佔位
    placeholder_table = col2.empty() # 為表格建立佔位

    if results.detections:
        # 建立一個副本用來畫框
        annotated_img = img_rgb.copy()

        placeholder_img.image(annotated_img, use_container_width=True)
        
        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            x, y = int(bbox.xmin * w_img), int(bbox.ymin * h_img)
            w, h = int(bbox.width * w_img), int(bbox.height * h_img)

            # 裁切人臉供 DeepFace 分析
            x1, y1 = max(0, x-20), max(0, y-20)
            x2, y2 = min(w_img, x+w+20), min(h_img, y+h+20)
            face_crop = img[y1:y2, x1:x2]

            try:
                if face_crop.size > 0:
                    res = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    dominant_emo = res[0]['dominant_emotion']
                    score = res[0]['emotion'][dominant_emo]
                    
                    analysis_data.append({
                        "人臉編號": i + 1, # 這裡的編號要跟圖片一致
                        "主要情緒": text_obj.get(dominant_emo, dominant_emo),
                        "信心值": f"{score:.2f}%"
                    })

                    # 1. 繪製綠色方框
                    cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 5) 
                    
                    # 2. 新增：在方框上方繪製編號背景（黑色小方塊，增加可讀性）
                    cv2.rectangle(annotated_img, (x, y - 45), (x + 50, y), (0, 255, 0), -1)
                    
                    # 3. 新增：寫入人臉編號 (白色文字)
                    cv2.putText(annotated_img, str(i + 1), (x + 5, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            except:
                continue

        with col1:
            st.subheader("📷 辨識結果圖")
            st.image(annotated_img, use_container_width=True)

        with col2:
            st.subheader("📊 數據統計表")
            if analysis_data:
                df = pd.DataFrame(analysis_data)
                # 使用 placeholder 確保表格出現在預定位置，不會擠壓到左側
                placeholder_table.dataframe(df, hide_index=True, use_container_width=True)
                
                # 下載按鈕放在表格下方
                csv = df.to_csv(index=False).encode('utf-8-sig')
                col2.download_button("下載報表 (CSV)", csv, "report.csv", "text/csv")
            else:
                st.warning("偵測到人臉但無法分析情緒。")

        
    else:
        st.error("圖片中未偵測到任何人臉。")

else:
    st.info("請從左側上傳圖片開始分析。")