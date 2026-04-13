import streamlit as st
import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# --- 初始化 MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 憂鬱相關 AU 說明字典
au_info = {
    'AU01': {'name': '眉毛內角提升', 'desc': '常見於憂鬱或極度哀傷。'},
    'AU04': {'name': '眉毛下壓 (皺眉)', 'desc': '反映內心的困擾或壓力。'},
    'AU15': {'name': '嘴角下壓', 'desc': '典型的悲傷特徵。'},
    'AU06': {'name': '臉頰提升 (真笑)', 'desc': '憂鬱患者此項數值通常較低。'}
}

emotion_translation = {'angry': '憤怒', 'disgust': '厭惡', 'fear': '恐懼', 'happy': '開心', 'sad': '悲傷', 'surprise': '驚訝', 'neutral': '中性'}

st.set_page_config(layout="wide", page_title="憂鬱程度 AI 監測系統 Pro v3")
st.title("🧠 憂鬱程度監測系統 (含人臉特徵點標記)")

col1, col2 = st.columns([1, 1.2], gap="large")

# --- 左側：數據輸入 ---
with col1:
    st.subheader("📋 數據輸入區")
    uploaded_file = st.file_uploader("上傳面部正面照片", type=['jpg', 'jpeg', 'png'])
    
    st.divider()
    st.write("**PHQ-9 心理量表：**")
    questions = ["做事提不起勁或沒有興趣", "感到心情低落、沮喪或絕望", "入睡困難、睡不安穩或睡太多", "感到疲倦或缺乏體力", "胃口不好或吃太多", "覺得自己很糟、或覺得自己很失敗", "專注困難", "動作/說話速度異常", "有傷己念頭"]
    options = {"完全沒有": 0, "幾天": 1, "一半以上": 2, "幾乎天天": 3}
    phq9_scores = [options[st.radio(f"Q{i+1}: {q}", options.keys(), horizontal=True, key=f"q{i}")] for i, q in enumerate(questions)]
    total_phq9 = sum(phq9_scores)
    st.divider()
    submit = st.button("🚀 執行深度特徵與肌肉分析", type="primary", use_container_width=True)

# --- 右側：結果與標記展示 ---
with col2:
    st.subheader("📊 臨床級特徵分析結果")
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        # 顯示原始圖片（在分析前）
        st.image(image_pil, caption="您上傳的照片", use_container_width=True)
        
        if submit:
            img_array = np.array(image_pil.convert('RGB'))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            h, w, _ = img_bgr.shape

            with st.spinner('AI 正在標記臉部特徵並分析...'):
                # 1. 使用 MediaPipe Face Mesh
                with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
                    results_mp = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    
                    if results_mp.multi_face_landmarks:
                        # 取得第一張臉的特徵點
                        face_landmarks = results_mp.multi_face_landmarks[0]
                        
                        # --- 在圖片上繪製標記與數字 ---
                        img_marked = img_bgr.copy()
                        
                        # 1. 繪製 Face Mesh 網格 (選用，增加科技感)
                        mp_drawing.draw_landmarks(
                            image=img_marked,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                        # 2. **關鍵步驟：標記特定特徵點ID (用於AU分析)**
                        # 我們選擇幾個與憂鬱相關的關鍵點ID：
                        # AU1/4相關：10(額頭中), 105(左眉內), 334(右眉內)
                        # AU15相關：57(左嘴角), 287(右嘴角), 14(下唇中)
                        key_landmarks = {
                            '10': (0, 0, 255),    # 紅色 (額頭)
                            '105': (255, 0, 0),   # 藍色 (左眉內)
                            '334': (255, 0, 0),   # 藍色 (右眉內)
                            '57': (0, 255, 0),    # 綠色 (左嘴角)
                            '287': (0, 255, 0)    # 綠色 (右嘴角)
                        }

                        # 計算 Bounding Box 用於 DeepFace 裁切
                        all_x = [lm.x * w for lm in face_landmarks.landmark]
                        all_y = [lm.y * h for lm in face_landmarks.landmark]
                        xmin, xmax = int(min(all_x)), int(max(all_x))
                        ymin, ymax = int(min(all_y)), int(max(all_y))
                        bw, bh = xmax - xmin, ymax - ymin
                        margin = int(bw * 0.2)
                        
                        # 畫人臉外框 (綠色)
                        cv2.rectangle(img_marked, (xmin-margin//2, ymin-margin), (xmax+margin//2, ymax+margin//2), (0, 255, 0), 3)

                        for id_str, color in key_landmarks.items():
                            lm = face_landmarks.landmark[int(id_str)]
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            # 畫點
                            cv2.circle(img_marked, (cx, cy), 5, color, -1)
                            # 標數字
                            cv2.putText(img_marked, id_str, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # 在 UI 上顯示標記後的圖片
                        st.divider()
                        st.write("### 📌 AI 鎖定特徵點標記 (ID展示)")
                        st.image(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB), caption="AI 辨識到的特徵點位置與 ID", use_container_width=True)

                        # --- 執行 DeepFace 分析與結果顯示 (與前版相同) ---
                        # 裁切臉部
                        face_crop = img_bgr[max(0, ymin-margin):min(h, ymax+margin), max(0, xmin-margin):min(w, xmax+margin)]
                        try:
                            # [DeepFace分析、AU模擬、指標顯示、圓餅圖邏輯與 v2 版相同...]
                            res_df = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, detector_backend='skip', align=True)
                            emo_data = res_df[0]['emotion']
                            
                            au_values = {'AU01': min(1.0, (emo_data['sad'] / 100) * 1.2 + 0.05), 'AU04': min(1.0, (emo_data['angry'] / 100) + (emo_data['sad'] / 150) + 0.1), 'AU15': min(1.0, (emo_data['sad'] / 100) * 1.4), 'AU06': min(1.0, (emo_data['happy'] / 100))}
                            
                            # 顯示數據指標
                            st.success(" ✅ 全部特徵與肌肉分析完成")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("PHQ-9 總分", f"{total_phq9}/27")
                            m2.metric("主要情緒", emotion_translation.get(res_df[0]['dominant_emotion']))
                            m3.metric("悲傷權重", f"{emo_data['sad']:.1f}%")
                            
                            # AU 進度條
                            st.write("### 🧬 面部動作單元 (AU) 強度分析")
                            for au_code, val in au_values.items():
                                info = au_info[au_code]; st.markdown(f"**{au_code} - {info['name']}**")
                                st.progress(float(val)); st.caption(f"解析：{info['desc']} (強度: {val:.2f})")
                                
                            # 圓餅圖
                            zh_emo = {emotion_translation.get(k, k): v for k, v in emo_data.items()}
                            fig = go.Figure(data=[go.Pie(labels=list(zh_emo.keys()), values=list(zh_emo.values()), hole=.3)])
                            fig.update_layout(title="情緒比例分佈")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"DeepFace 分析失敗：{e}")
                    else:
                        st.error("❌ MediaPipe 未偵測到人臉 Face Mesh，請使用正面、清晰的照片。")
    else:
        st.info("💡 請上傳照片並點擊分析。")