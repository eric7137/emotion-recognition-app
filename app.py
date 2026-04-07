import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import pandas as pd
import plotly.express as px # 引入 Plotly 畫圖

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="AI 情緒辨識分析儀", layout="wide")
st.title("📊 AI 人臉情緒分佈分析儀")
st.write("上傳照片後，AI 將分析每個人臉的詳細情緒分佈並繪製長條圖。")

# --- 2. 初始化模型 (使用快取) ---
@st.cache_resource
def load_models():
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detector

face_detection = load_models()

# 情緒中文化對照 (與原版一致)
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
    
    summary_data = []    # 存放主表格數據 (簡化版)
    all_emotions_scores = [] # 存放所有人臉的所有情緒分數 (畫圖用)
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
                                         detector_backend='skip')
                    
                    # 1. 取得主情緒
                    dominant_emo = res[0]['dominant_emotion']
                    
                    # 2. **關鍵修改**：取得所有情緒的詳細分數
                    raw_scores = res[0]['emotion']
                    # 將分數轉換成 DataFrame 格式供 Plotly 使用
                    for emo_en, score in raw_scores.items():
                        all_emotions_scores.append({
                            "人臉編號": f"人臉 {i+1}",
                            "情緒名稱": text_obj.get(emo_en, emo_en),
                            "信心值 (%)": score
                        })
                    
                    # 3. 紀錄到主統計表格資料 (維持簡化，只看最顯著)
                    summary_data.append({
                        "人臉編號": i + 1,
                        "主要情緒": text_obj.get(dominant_emo, dominant_emo)
                    })

                    # --- 繪製視覺標記 ---
                    # 畫綠色框
                    cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    # 畫編號背景塊
                    cv2.rectangle(annotated_img, (x, y-50), (x+60, y), (0, 255, 0), -1)
                    # 寫上編號 (白色文字)
                    cv2.putText(annotated_img, str(i+1), (x+10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    
                    # 更新圖片顯示
                    placeholder_img.image(annotated_img, use_container_width=True)
            except:
                continue

        # --- 最終結果呈現 ---
        with col2:
            st.subheader("📊 總體統計表")
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                # 渲染最終表格
                placeholder_table.dataframe(df_summary, hide_index=True, use_container_width=True)
                
                # 下載按鈕
                csv = df_summary.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 下載主要報表 (CSV)", csv, "emotion_summary.csv", "text/csv")
            else:
                st.warning("未能成功分析情緒數據。")
        
        # --- **核心功能**：繪製情緒長條圖 (顯示在圖片下方) ---
        if all_emotions_scores:
            st.divider() # 加一條分割線
            st.subheader("📈 詳細情緒信心值分佈圖")
            df_scores = pd.DataFrame(all_emotions_scores)
            
            # 使用 Plotly 畫堆疊長條圖，按人臉分組
            fig = px.bar(df_scores, x="人臉編號", y="信心值 (%)", color="情緒名稱", 
                         title="各人臉之所有情緒信心值分佈",
                         text_auto='.1f', # 在圖上顯示數字
                         color_discrete_sequence=px.colors.qualitative.Pastel) # 使用柔和配色
            
            # 優化圖表佈局
            fig.update_layout(barmode='group', xaxis_tickangle=0) # 改為並排長條圖
            fig.update_traces(textposition='outside') # 數字顯示在條柱外
            
            st.plotly_chart(fig, use_container_width=True) # 在網頁顯示圖表

    else:
        st.error("圖片中未偵測到人臉，請換一張試試！")

else:
    st.info("💡 提示：請從左側上傳圖片開始分析。")