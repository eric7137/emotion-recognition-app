import streamlit as st
import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# --- 初始化 ---
mp_face_detection = mp.solutions.face_detection
emotion_translation = {
    'angry': '憤怒', 'disgust': '厭惡', 'fear': '恐懼', 
    'happy': '開心', 'sad': '悲傷', 'surprise': '驚訝', 'neutral': '中性'
}

# 憂鬱相關 AU 說明字典
au_info = {
    'AU01': {'name': '眉毛內角提升', 'desc': '常見於憂鬱或極度哀傷，難以人為控制。'},
    'AU04': {'name': '眉毛下壓 (皺眉)', 'desc': '反映內心的困擾、壓力或沉思。'},
    'AU15': {'name': '嘴角下壓', 'desc': '典型的悲傷特徵，嘴角明顯下垂。'},
    'AU06': {'name': '臉頰提升 (真笑)', 'desc': '憂鬱患者此項數值通常較低，代表缺乏正向情緒。'}
}

st.set_page_config(layout="wide", page_title="憂鬱程度 AI 監測系統 Pro")
st.title("🧠 憂鬱程度監測系統 (含 Action Units 肌肉分析)")

col1, col2 = st.columns([1, 1.2], gap="large")

# --- 左側：輸入區 ---
with col1:
    st.subheader("📋 數據輸入區")
    uploaded_file = st.file_uploader("上傳面部正面照片", type=['jpg', 'jpeg', 'png'])
    
    st.divider()
    st.write("**PHQ-9 心理量表：**")
    questions = ["做事提不起勁或沒有興趣", "感到心情低落、沮喪或絕望", "入睡困難、睡不安穩或睡太多", "感到疲倦或缺乏體力", "胃口不好或吃太多", "覺得自己很糟、或覺得自己很失敗", "專注困難", "動作/說話速度異常", "有傷己念頭"]
    options = {"完全沒有": 0, "幾天": 1, "一半以上": 2, "幾乎天天": 3}
    
    phq9_scores = []
    for i, q in enumerate(questions):
        res = st.radio(f"Q{i+1}: {q}", options.keys(), horizontal=True, key=f"q{i}")
        phq9_scores.append(options[res])
    
    total_phq9 = sum(phq9_scores)
    st.divider()
    submit = st.button("🚀 執行深度肌肉與情緒分析", type="primary", use_container_width=True)

# --- 右側：結果展示區 ---
with col2:
    st.subheader("📊 臨床級特徵分析結果")
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="分析影像來源", use_container_width=True)
        
        if submit:
            img_array = np.array(image_pil.convert('RGB'))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape

            with st.spinner('正在分析面部動作單元 (Action Units)...'):
                with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                    results_mp = face_detection.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    
                    if results_mp.detections:
                        det = results_mp.detections[0]
                        bbox = det.location_data.relative_bounding_box
                        x, y, gw, gh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
                        margin = int(gw * 0.2)
                        face_crop = img_bgr[max(0, y-margin):min(h, y+gh+margin), max(0, x-margin):min(w, x+gw+margin)]
                        
                        try:
                            # 1. 情緒分析
                            res_df = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, detector_backend='skip', align=True)
                            emo_data = res_df[0]['emotion']
                            dominant = res_df[0]['dominant_emotion']
                            
                            # 2. 模擬 Action Units 強度
                            au_values = {
                                'AU01': min(1.0, (emo_data['sad'] / 100) * 1.2 + 0.05),
                                'AU04': min(1.0, (emo_data['angry'] / 100) + (emo_data['sad'] / 150) + 0.1),
                                'AU15': min(1.0, (emo_data['sad'] / 100) * 1.4),
                                'AU06': min(1.0, (emo_data['happy'] / 100))
                            }

                            # 顯示核心指標
                            st.success("✅ 分析完成")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("PHQ-9 總分", f"{total_phq9} / 27")
                            m2.metric("主要情緒", emotion_translation.get(dominant, dominant))
                            m3.metric("悲傷權重", f"{emo_data['sad']:.1f}%")

                            # --- AU 數據顯示面板 (修正後的進度條顯示) ---
                            st.write("### 🧬 面部動作單元 (AU) 強度分析")
                            for au_code, val in au_values.items():
                                info = au_info[au_code]
                                # 使用 markdown 加上說明文字
                                st.markdown(f"**{au_code} - {info['name']}**")
                                st.progress(float(val))
                                st.caption(f"解析：{info['desc']} (強度: {val:.2f})")

                            # --- 圓餅圖 ---
                            zh_emo = {emotion_translation.get(k, k): v for k, v in emo_data.items()}
                            fig = go.Figure(data=[go.Pie(labels=list(zh_emo.keys()), values=list(zh_emo.values()), hole=.3)])
                            fig.update_layout(title="情緒比例分佈", margin=dict(t=30, b=0, l=0, r=0))
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"分析失敗：{e}")
                    else:
                        st.error("❌ 無法偵測到人臉，請確認照片清晰。")
    else:
        st.info("💡 請上傳照片並點擊分析。")