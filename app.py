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

st.title("🧠 臉部表情與 PHQ-9 憂鬱量表整合系統")
st.markdown("本系統結合 **MediaPipe** 精準定位與 **DeepFace** 情緒分析，並與主觀量表進行對照。")

# 建立左右兩欄
col1, col2 = st.columns([1, 1.2], gap="large")

# --- 左側：PHQ-9 問卷與影像上傳 ---
with col1:
    st.subheader("📋 數據輸入")
    uploaded_file = st.file_uploader("1. 上傳面部正面照片", type=['jpg', 'jpeg', 'png'])
    
    st.write("2. 請根據過去兩週的感受回答 PHQ-9 問卷：")
    questions = [
        "做事提不起勁或沒有興趣", "感到心情低落、沮喪或絕望", "入睡困難、睡不安穩或睡太多",
        "感到疲倦或缺乏體力", "胃口不好或吃太多", "覺得自己很糟、或覺得自己很失敗",
        "專注困難（如看報紙或看電視時）", "動作或說話速度緩慢/太快到別人都注意到了",
        "有想要傷害自己或自殺的想法"
    ]
    options = {"完全沒有": 0, "幾天": 1, "一半以上": 2, "幾乎天天": 3}
    
    phq9_scores = []
    # 使用 st.expander 縮小問卷佔用的空間
    with st.expander("展開問卷內容"):
        for i, q in enumerate(questions):
            res = st.radio(f"Q{i+1}: {q}", options.keys(), horizontal=True, key=f"q{i}")
            phq9_scores.append(options[res])
    
    total_phq9 = sum(phq9_scores)
    submit = st.button("🚀 開始深度分析", type="primary", use_container_width=True)

# --- 右側：AI 分析與圓餅圖可視化 ---
with col2:
    st.subheader("📊 AI 客觀辨識結果")
    
    if uploaded_file is not None:
        # 讀取圖片
        image_pil = Image.open(uploaded_file)
        img_array = np.array(image_pil.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        if submit:
            with st.spinner('正在執行多模態分析...'):
                # 1. MediaPipe 臉部偵測
                with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                    results_mp = face_detection.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    
                    if results_mp.detections:
                        # 裁切臉部（包含 margin）
                        det = results_mp.detections[0]
                        bbox = det.location_data.relative_bounding_box
                        x, y, gw, gh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
                        
                        margin = int(gw * 0.15) # 稍微留白
                        face_crop = img_bgr[max(0, y-margin):min(h, y+gh+margin), max(0, x-margin):min(w, x+gw+margin)]
                        
                        try:
                            # 2. DeepFace 分析 (使用你指定的參數)
                            results_df = DeepFace.analyze(
                                face_crop, 
                                actions=['emotion'], 
                                enforce_detection=False, 
                                detector_backend='skip', 
                                align=True
                            )
                            
                            emo_data = results_df[0]['emotion']
                            dominant = results_df[0]['dominant_emotion']
                            
                            # 3. 顯示結果指標
                            res_c1, res_c2 = st.columns(2)
                            res_c1.metric("PHQ-9 總分", f"{total_phq9}/27")
                            res_c2.metric("主要表情", f"{dominant.capitalize()}")

                            # --- 4. 繪製 Plotly 圓餅圖 ---
                            labels = list(emo_data.keys())
                            values = list(emo_data.values())
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=labels, 
                                values=values, 
                                hole=.4, # 鏤空變甜甜圈圖，更有現代感
                                marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692'])
                            )])
                            
                            fig.update_layout(
                                title_text="表情成分比例分佈",
                                annotations=[dict(text='情緒', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=True,
                                margin=dict(t=40, b=0, l=0, r=0)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 5. 結論建議
                            st.divider()
                            if total_phq9 >= 10 or emo_data['sad'] > 30:
                                st.warning(f"💡 **分析建議：** 檢測到較高的情緒特徵值。建議您多關注自身心理狀態，必要時可尋求專業諮商。")
                            else:
                                st.success("💡 **分析建議：** 目前數據顯示狀態尚屬平穩，請繼續保持健康生活。")

                        except Exception as e:
                            st.error(f"分析出錯：{e}")
                    else:
                        st.error("❌ 無法偵測到臉部，請確保照片清晰且光線充足。")
    else:
        st.info("💡 請在左側上傳照片並填寫問卷，然後點擊「開始深度分析」。")