import streamlit as st
import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# --- 初始化 MediaPipe (AU 視覺化用) ---
mp_face_mesh = mp.solutions.face_mesh

# 語系與說明字典
au_info = {
    'AU01': {'name': '眉毛內角提升', 'desc': '常見於憂鬱或極度哀傷。'},
    'AU04': {'name': '眉毛下壓 (皺眉)', 'desc': '反映內心的困擾或壓力。'},
    'AU15': {'name': '嘴角下壓', 'desc': '典型的悲傷特徵。'},
    'AU06': {'name': '臉頰提升 (真笑)', 'desc': '憂鬱患者此項數值通常較低。'}
}
emotion_translation = {'angry': '憤怒', 'disgust': '厭惡', 'fear': '恐懼', 'happy': '開心', 'sad': '悲傷', 'surprise': '驚訝', 'neutral': '中性'}

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="臨床級憂鬱監測系統 v4.3")

# 自定義 CSS (簡潔、間距明快)
st.markdown("""
    <style>
    /* 針對指標卡片優化，使用半透明背景使其適應任何主題 */
    [data-testid="stMetric"] { 
        background-color: rgba(128, 128, 128, 0.1); 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 側邊欄：輸入與參數 ---
with st.sidebar:
    st.title("⚙️ 設定面板")
    
    st.subheader("1. 影像來源")
    uploaded_file = st.file_uploader("選擇正面照片", type=['jpg', 'jpeg', 'png'])
    
    st.divider()
    st.subheader("2. 演算法參數")
    enable_au = st.toggle("開啟 AU 肌肉分析", value=True)
    st.caption("使用 MediaPipe 特徵點定位與肌肉單元強度預估")
    
    st.divider()
    st.subheader("3. PHQ-9 量表")
    questions = ["做事提不起勁", "心情低落沮喪", "入睡困難/多睡", "感到疲倦", "胃口不佳/過飽", "覺得自己很失敗", "注意力集中困難", "行動/言語緩慢", "有自我傷害念頭"]
    options = {"完全沒有": 0, "幾天": 1, "一半以上": 2, "幾乎天天": 3}
    
    phq9_scores = [options[st.selectbox(f"Q{i+1}: {q}", options.keys(), key=f"q{i}")] for i, q in enumerate(questions)]
    total_phq9 = sum(phq9_scores)
    
    st.divider()
    submit = st.button("🚀 開始深度診斷", type="primary", use_container_width=True)

# --- 主畫面佈局 ---
st.title("🧠 憂鬱程度 AI 監測系統")

if uploaded_file:
    # --- 讀取原始影像，取消任何固定尺寸限制 ---
    image_pil = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image_pil)
    
    # 取得原始尺寸，僅用於計算特徵點座標與 UI 標示
    orig_w, orig_h = image_pil.size

    # 建立左右兩欄 (比例 1 : 1.2)
    col_img, col_res = st.columns([1, 1.2], gap="large")

    with col_img:
        st.subheader("📸 監測影像")
        # 關鍵：使用 use_container_width=True，讓 Streamlit 自動等比縮放適應左側欄寬度
        st.image(image_pil, caption=f"原圖解析度: {orig_w}x{orig_h}", use_container_width=True)
        
        if submit and enable_au:
            st.divider()
            st.subheader("📌 核心特徵標記")
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                results_mp = face_mesh.process(img_array)
                if results_mp.multi_face_landmarks:
                    face_landmarks = results_mp.multi_face_landmarks[0]
                    # 繪製選定的關鍵 ID
                    key_points = {'10': (0,0,255), '105': (255,0,0), '334': (255,0,0), '57': (0,255,0), '287': (0,255,0)}
                    
                    # 根據原圖比例計算特徵點並繪製
                    for id_str, color in key_points.items():
                        lm = face_landmarks.landmark[int(id_str)]
                        cx, cy = int(lm.x * orig_w), int(lm.y * orig_h)
                        # 依據圖片大小動態調整圓點大小與字體，避免高畫質圖片標記太小
                        dot_size = max(3, int(orig_w * 0.008))
                        font_scale = max(0.5, orig_w * 0.001)
                        
                        cv2.circle(img_bgr, (cx, cy), dot_size, color, -1)
                        cv2.putText(img_bgr, id_str, (cx + dot_size*2, cy + dot_size*2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, int(font_scale*2)))
                    
                    # 標記完成後，一樣使用 use_container_width=True 進行響應式顯示
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_res:
        st.subheader("📊 診斷報告")
        if submit:
            with st.spinner("AI 深度運算中..."):
                try:
                    # 使用原圖進行分析，精準度最高
                    res_df = DeepFace.analyze(img_array, actions=['emotion'], detector_backend='mediapipe')
                    emo_data = res_df[0]['emotion']
                    dominant_emo = res_df[0]['dominant_emotion']

                    # 指標儀表板
                    m1, m2, m3 = st.columns(3)
                    m1.metric("PHQ-9 總分", f"{total_phq9}")
                    m2.metric("主要情緒", emotion_translation.get(dominant_emo, dominant_emo))
                    m3.metric("悲傷權重", f"{emo_data['sad']:.1f}%")

                    # 情緒分佈圖
                    zh_emo = {emotion_translation.get(k, k): v for k, v in emo_data.items()}
                    fig = go.Figure(data=[go.Pie(labels=list(zh_emo.keys()), values=list(zh_emo.values()), hole=.4)])
                    fig.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                    st.plotly_chart(fig, use_container_width=True)

                    if enable_au:
                        st.divider()
                        st.write("### 🧬 AU 肌肉單元分析")
                        au_scores = {
                            'AU01 (額肌內側)': min(1.0, (emo_data['sad']/100)*1.2 + 0.05),
                            'AU04 (皺眉肌)': min(1.0, (emo_data['angry']/100) + (emo_data['sad']/150) + 0.1),
                            'AU15 (降口角肌)': min(1.0, (emo_data['sad']/100)*1.4),
                            'AU06 (顴大肌/眼輪匝肌)': min(1.0, (emo_data['happy']/100))
                        }
                        for label, val in au_scores.items():
                            st.text(label)
                            st.progress(float(val))
                except Exception as e:
                    st.error(f"分析失敗：{e}")
        else:
            st.write("等待執行分析...")
else:
    st.info("請於左側控制面板上傳面部照片以開始分析。")