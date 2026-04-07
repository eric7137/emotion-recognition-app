import streamlit as st
import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# --- 初始化 MediaPipe ---
mp_face_detection = mp.solutions.face_detection

st.set_page_config(layout="wide", page_title="憂鬱程度 AI 檢測系統")

st.title("🧠 臉部表情與 PHQ-9 憂鬱量表整合展示")

# 建立左右兩欄
col1, col2 = st.columns([1, 1.2], gap="large")

# --- 左側：影像上傳與完整問卷 ---
with col1:
    st.subheader("📋 數據輸入區")
    
    # 1. 影像上傳
    uploaded_file = st.file_uploader("上傳面部正面照片", type=['jpg', 'jpeg', 'png'])
    
    st.divider()
    
    # 2. PHQ-9 完整問卷 (不摺疊)
    st.write("**請根據過去兩週的感受回答：**")
    questions = [
        "做事提不起勁或沒有興趣", 
        "感到心情低落、沮喪或絕望", 
        "入睡困難、睡不安穩或睡太多",
        "感到疲倦或缺乏體力", 
        "胃口不好或吃太多", 
        "覺得自己很糟、或覺得自己很失敗",
        "專注困難（如看報紙或看電視時）", 
        "動作或說話速度緩慢/太快到別人都注意到了",
        "有想要傷害自己或自殺的想法"
    ]
    options = {"完全沒有": 0, "幾天": 1, "一半以上": 2, "幾乎天天": 3}
    
    phq9_scores = []
    for i, q in enumerate(questions):
        res = st.radio(f"Q{i+1}: {q}", options.keys(), horizontal=True, key=f"q{i}")
        phq9_scores.append(options[res])
    
    total_phq9 = sum(phq9_scores)
    
    st.divider()
    submit = st.button("🚀 執行 AI 辨識分析", type="primary", use_container_width=True)

# --- 右側：結果展示區 ---
with col2:
    st.subheader("📊 檢測結果對照")
    
    if uploaded_file is not None:
        # 顯示原始上傳圖片
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="您上傳的照片", use_container_width=True)
        
        if submit:
            img_array = np.array(image_pil.convert('RGB'))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape

            with st.spinner('AI 正在分析面部與數據...'):
                # 1. MediaPipe 臉部偵測與裁切
                with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                    results_mp = face_detection.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    
                    if results_mp.detections:
                        det = results_mp.detections[0]
                        bbox = det.location_data.relative_bounding_box
                        x, y, gw, gh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
                        
                        # 增加裁切邊距以便 DeepFace align
                        margin = int(gw * 0.2)
                        face_crop = img_bgr[max(0, y-margin):min(h, y+gh+margin), max(0, x-margin):min(w, x+gw+margin)]
                        
                        try:
                            # 2. DeepFace 情緒辨識 (使用指定參數)
                            results_df = DeepFace.analyze(
                                face_crop, 
                                actions=['emotion'], 
                                enforce_detection=False, 
                                detector_backend='skip', 
                                align=True
                            )
                            
                            emo_data = results_df[0]['emotion']
                            dominant_emotion = results_df[0]['dominant_emotion']
                            sad_prob = emo_data['sad']
                            
                            # 3. 顯示量化分數指標
                            st.success("分析完成！數據對照如下：")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("PHQ-9 總分", f"{total_phq9} / 27")
                            m2.metric("主要辨識情緒", f"{dominant_emotion.capitalize()}")
                            m3.metric("AI 悲傷概率", f"{sad_prob:.1f} %")
                            
                            # 4. 圓餅圖展示
                            fig = go.Figure(data=[go.Pie(
                                labels=list(emo_data.keys()), 
                                values=list(emo_data.values()), 
                                hole=.3,
                                textinfo='label+percent'
                            )])
                            fig.update_layout(title="AI 面部情緒成分分析", margin=dict(t=50, b=0, l=0, r=0))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 5. 判斷提示
                            if total_phq9 >= 10:
                                st.warning(f"⚠️ **提醒：** 您的 PHQ-9 分數為 {total_phq9}，顯示有中度以上憂鬱傾向，建議諮詢專業醫療人員。")
                            
                        except Exception as e:
                            st.error(f"分析失敗：{e}")
                    else:
                        st.error("❌ 無法偵測到臉部，請確保照片中臉部清晰。")
    else:
        st.info("💡 請在左側上傳照片並完成問卷，系統將自動進行對照分析。")