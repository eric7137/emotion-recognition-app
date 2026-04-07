import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import pandas as pd

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="AI 情緒辨識系統", layout="wide")
st.title("😊 AI 人臉情緒辨識分析儀")
st.write("上傳照片後，系統將自動標記人臉編號並產出情緒分析表格。")

# --- 2. 初始化模型 (使用快取避免重複載入) ---
@st.cache_resource
def load_models():
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detector

face_detection = load_models()

# 情緒對照表
text_obj = {
    'angry': '😡 生氣', 'disgust': '🤢 噁心', 'fear': '😨 害怕',
    'happy': '😄 開心', 'sad': '😢 難過', 'surprise': '😲 驚訝', 'neutral': '😐 正常'
}

# --- 3. 側邊欄上傳介面 ---
uploaded_file = st.sidebar.file_uploader("請選擇一張照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片轉換為 OpenCV 格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape

    # 準備佈局容器：左邊放圖，右邊放表
    col1, col2 = st.columns([2, 1])
    placeholder_img = col1.empty()   # 圖片佔位
    placeholder_table = col2.empty() # 表格佔位
    
    # 執行 MediaPipe 偵測
    results = face_detection.process(img_rgb)
    
    analysis_data = [] # 存放表格數據
    annotated_img = img_rgb.copy() # 用於畫框的副本

    if results.detections:
        # 先顯示原始圖，防止畫面跳動
        placeholder_img.image(annotated_img, use_container_width=True)

        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            x, y = int(bbox.xmin * w_img), int(bbox.ymin * h_img)
            w, h = int(bbox.width * w_img), int(bbox.height * h_img)

            # 裁切人臉
            x1, y1 = max(0, x-20), max(0, y-20)
            x2, y2 = min(w_img, x+w+20), min(h_img, y+h+20)
            face_crop = img[y1:y2, x1:x2]

            try:
                if face_crop.size > 0:
                    # DeepFace 情緒分析
                    res = DeepFace.analyze(face_crop, actions=['emotion'], 
                                         enforce_detection=False, 
                                         detector_backend='skip') # 已有切片，用 skip 最快
                    
                    dominant_emo = res[0]['dominant_emotion']
                    score = res[0]['emotion'][dominant_emo]
                    
                    # 紀錄到表格資料
                    analysis_data.append({
                        "人臉編號": i + 1,
                        "主要情緒": text_obj.get(dominant_emo, dominant_emo),
                        "信心值": f"{score:.2f}%"
                    })

                    # --- 繪製視覺標記 ---
                    # 畫綠色框
                    cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    # 畫編號背景塊
                    cv2.rectangle(annotated_img, (x, y-50), (x+60, y), (0, 255, 0), -1)
                    # 寫上編號 (白色文字)
                    cv2.putText(annotated_img, str(i+1), (x+10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    
                    # 更新圖片顯示 (讓使用者看到處理進度)
                    placeholder_img.image(annotated_img, use_container_width=True)
            except:
                continue

        # --- 最終結果呈現 ---
        with col2:
            st.subheader("📊 數據統計表")
            if analysis_data:
                df = pd.DataFrame(analysis_data)
                # 渲染最終表格
                placeholder_table.dataframe(df, hide_index=True, use_container_width=True)
                
                # 下載按鈕
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 下載 CSV 報表", csv, "emotion_report.csv", "text/csv")
            else:
                st.warning("未能成功分析情緒數據。")
    else:
        st.error("圖片中未偵測到人臉，請換一張試試！")

else:
    st.info("💡 提示：請從左側上傳圖片開始分析。")