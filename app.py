import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="情緒與 PHQ-9 綜合評估", layout="wide")
st.title("🧠 臉部表情與 PHQ-9 憂鬱量表綜合分析")

# --- 初始化模型 ---
@st.cache_resource
def load_models():
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detector

face_detection = load_models()
text_obj = {'angry': '😡 生氣', 'disgust': '🤢 噁心', 'fear': '😨 害怕', 'happy': '😄 開心', 'sad': '😢 難過', 'surprise': '😲 驚訝', 'neutral': '😐 正常'}

# --- 側邊欄：圖片上傳 ---
uploaded_file = st.sidebar.file_uploader("第一步：請上傳臉部照片", type=["jpg", "jpeg", "png"])

# --- 主要內容區：分為左右兩欄 ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📷 AI 影像情緒分析")
    ai_results = {} # 儲存 AI 分析結果
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)
        
        if results.detections:
            # 簡化處理：取第一個人臉作為主評估對象
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
            face_crop = img[max(0, y-20):min(h, y+bh+20), max(0, x-20):min(w, x+bw+20)]
            
            try:
                res = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, detector_backend='skip')
                ai_results = res[0]['emotion']
                dominant_emo = res[0]['dominant_emotion']
                
                # 畫框與顯示
                cv2.rectangle(img_rgb, (x, y), (x+bw, y+bh), (0, 255, 0), 5)
                st.image(img_rgb, use_container_width=True)
                st.success(f"偵測到主情緒：{text_obj.get(dominant_emo)}")
            except:
                st.error("AI 分析失敗，請確保臉部清晰。")
        else:
            st.warning("未偵測到人臉。")
    else:
        st.info("請先從左側上傳照片，以便進行影像分析。")

with right_col:
    st.subheader("📝 PHQ-9 自我評估表")
    questions = [
        "做事提不起勁或沒有興趣", "感到心情低落、沮喪或絕望", "入睡困難、睡不安穩或睡太多",
        "感到疲倦或沒有活力", "食慾不振或吃太多", "覺得自己很糟、或覺得家人失望",
        "專注困難 (例如看報紙或看電視)", "動作緩慢或坐立不安", "有受傷或自殺的念頭"
    ]
    options = {"完全沒有": 0, "幾天": 1, "一半以上": 2, "幾乎每天": 3}
    
    # 使用 Form 確保一次提交
    with st.form("phq9_form"):
        scores = []
        for i, q in enumerate(questions):
            choice = st.radio(f"{i+1}. {q}", options.keys(), horizontal=True)
            scores.append(options[choice])
        
        submitted = st.form_submit_button("送出綜合評估")

# --- 提交後的綜合分析 ---
if submitted:
    phq9_score = sum(scores)
    st.divider()
    
    final_left, final_right = st.columns(2)
    
    with final_left:
        st.metric("PHQ-9 總分", f"{phq9_score} 分")
        if phq9_score <= 4: st.info("評估結果：無憂鬱傾向")
        elif phq9_score <= 9: st.warning("評估結果：輕微憂鬱")
        elif phq9_score <= 14: st.error("評估結果：中度憂鬱")
        else: st.error("評估結果：重度憂鬱，建議尋求專業諮詢")

    with final_right:
        if ai_results:
            st.write("### AI 觀察與自陳對照")
            # 繪製詳細情緒分佈
            df_emo = pd.DataFrame({"情緒": [text_obj.get(k, k) for k in ai_results.keys()], "比例": list(ai_results.values())})
            fig = px.pie(df_emo, values='比例', names='情緒', title="影像情緒佔比")
            st.plotly_chart(fig, use_container_width=True)
            
            # 綜合判斷邏輯示例
            if ai_results['sad'] > 30 and phq9_score > 10:
                st.error("🚨 警告：AI 偵測到高度憂傷表情，且量表分數偏高，請務必關注身心狀態。")
            elif ai_results['happy'] > 50 and phq9_score > 10:
                st.warning("💡 觀察：影像表情顯得開心，但自陳量表分數較高，可能存在「微笑憂鬱」的風險。")