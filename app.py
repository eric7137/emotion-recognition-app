import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import pandas as pd
import plotly.express as px

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="AI 情緒辨識分析儀", layout="wide")
st.title("📊 AI 人臉情緒深度分析")
st.write("上傳照片後，透過下拉選單查看特定人臉的詳細情緒分佈。")

# --- 2. 初始化模型 (使用快取) ---
@st.cache_resource
def load_models():
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detector

face_detection = load_models()

# 情緒中文化對照
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

    # 準備佈局容器
    col1, col2 = st.columns([2, 1])
    placeholder_img = col1.empty()   # 圖片佔位
    placeholder_table = col2.empty() # 總表佔位
    
    # 執行 MediaPipe 偵測
    results = face_detection.process(img_rgb)
    
    summary_data = []    
    face_details_dict = {} # 用來存放每個人臉的詳細分數
    annotated_img = img_rgb.copy()

    if results.detections:
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
                    # DeepFace 分析
                    res = DeepFace.analyze(face_crop, actions=['emotion'], 
                                         enforce_detection=False, 
                                         detector_backend='skip')
                    
                    dominant_emo = res[0]['dominant_emotion']
                    raw_scores = res[0]['emotion']
                    
                    # 儲存詳細分數供下拉選單使用
                    face_label = f"人臉 {i+1}"
                    face_details_dict[face_label] = pd.DataFrame({
                        "情緒": [text_obj.get(k, k) for k in raw_scores.keys()],
                        "信心值 (%)": list(raw_scores.values())
                    })
                    
                    summary_data.append({
                        "人臉編號": i + 1,
                        "主要情緒": text_obj.get(dominant_emo, dominant_emo)
                    })

                    # 標記圖片
                    cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    cv2.rectangle(annotated_img, (x, y-50), (x+60, y), (0, 255, 0), -1)
                    cv2.putText(annotated_img, str(i+1), (x+10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    
                    placeholder_img.image(annotated_img, use_container_width=True)
            except:
                continue

        # --- 右側：顯示總體表格 ---
        with col2:
            st.subheader("📊 總體統計表")
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                placeholder_table.dataframe(df_summary, hide_index=True, use_container_width=True)
            else:
                st.warning("未能偵測到有效數據。")

        # --- 下方：下拉選單與長條圖 ---
        if face_details_dict:
            st.divider()
            st.subheader("🔍 單一人臉情緒深度分析")
            
            # 建立下拉選單
            selected_face = st.selectbox("請選擇要查看的人臉編號：", options=list(face_details_dict.keys()))
            
            # 取得該人臉的資料並畫圖
            detail_df = face_details_dict[selected_face]
            
            # 排序讓最高的分數排在前面
            detail_df = detail_df.sort_values(by="信心值 (%)", ascending=False)

            fig = px.bar(detail_df, x="情緒", y="信心值 (%)", 
                         color="情緒", 
                         text_auto='.1f',
                         title=f"{selected_face} 的情緒分佈細節",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            
            fig.update_layout(showlegend=False) # 關閉側邊圖例，因為 X 軸已經有標籤了
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("圖片中未偵測到人臉。")
else:
    st.info("請從左側上傳圖片。")