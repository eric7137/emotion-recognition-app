import streamlit as st
import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image
import numpy as np

# --- 初始化 MediaPipe 人臉偵測 ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 設置頁面佈局為寬版
st.set_page_config(layout="wide", page_title="憂鬱程度檢測展示 v2")

st.title("臉部表情偵測 (MediaPipe對齊) 與 PHQ-9 量表對照")
st.write("請填寫問卷，系統將利用 MediaPipe 精準定位臉部後，由 AI 分析情緒。")

# 建立左右兩欄
col1, col2 = st.columns([1, 1])

# --- 左側：影像上傳與 PHQ-9 問卷 ---
with col1:
    st.subheader("📋 用戶輸入區")
    
    # 影像上傳
    uploaded_file = st.file_uploader("上傳一張面部照片", type=['jpg', 'jpeg', 'png'])
    
    st.divider()
    
    # PHQ-9 問卷題目
    questions = ["做事提不起勁或沒有興趣", "感到心情低落、沮喪或絕望", "入睡困難、睡不安穩或睡太多", "感到疲倦或缺乏體力", "胃口不好或吃太多", "覺得自己很糟、或覺得自己很失敗", "專注困難（如看報紙或看電視時）", "動作或說話速度緩慢到別人都注意到了，或正好相反", "有想要傷害自己或自殺的想法"]
    options = {"完全沒有": 0, "幾天": 1, "一半以上": 2, "幾乎天天": 3}
    
    phq9_scores = []
    for i, q in enumerate(questions):
        res = st.radio(f"Q{i+1}: {q}", options.keys(), horizontal=True, key=f"q{i}")
        phq9_scores.append(options[res])
    
    total_phq9 = sum(phq9_scores)
    submit = st.button("提交分析", type="primary", use_container_width=True)

# --- 右側：AI 分析結果 ---
with col2:
    st.subheader("🤖 AI 精準分析結果")
    
    if uploaded_file is not None:
        # 將 PIL Image 轉為 OpenCV BGR 格式
        image_pil = Image.open(uploaded_file)
        img_array = np.array(image_pil.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        with st.spinner('MediaPipe 正在定位並裁切臉部...'):
            # 1. 使用 MediaPipe 偵測人臉
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                results_mp = face_detection.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                
                if results_mp.detections:
                    # 假設只取偵測到的第一張臉
                    detection = results_mp.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    
                    # 計算像素座標
                    xmin = int(bboxC.xmin * w)
                    ymin = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    # 為了強健性，增加一點邊距並確保不超出邊界
                    margin = int(width * 0.1)
                    xmin = max(0, xmin - margin)
                    ymin = max(0, ymin - margin)
                    xmax = min(w, xmin + width + 2*margin)
                    ymax = min(h, ymin + height + 2*margin)
                    
                    # 裁切臉部 (這就是我們要傳給 DeepFace 的 face)
                    face_crop = img_bgr[ymin:ymax, xmin:xmax]
                    
                    # 在原圖上畫框 (用於 Demo 顯示)
                    img_with_box = img_bgr.copy()
                    cv2.rectangle(img_with_box, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3) # 綠色框
                    
                    # 顯示畫了框的圖片
                    st.image(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB), caption="MediaPipe 定位人臉", use_container_width=True)
                else:
                    st.error("MediaPipe 未偵測到人臉，請重新上傳清晰的照片。")
                    submit = False # 阻止後續分析

        # 當用戶按下按鈕，且 MediaPipe 有偵測到臉時
        if submit and results_mp.detections:
            with st.spinner('DeepFace 正在進行情緒辨識 (已加速)...'):
                try:
                    # 2. 執行 DeepFace 分析 (使用你要求的參數)
                    # 我們將 detector_backend 設為 'skip'，因為我們已經手動提供了裁切好的 face
                    results_df = DeepFace.analyze(face_crop, actions=['emotion'], 
                                                 enforce_detection=False, 
                                                 detector_backend='skip', 
                                                 align=True) # align=True 對於裁切後的圖片仍有幫助
                    
                    # 取得分析數據
                    emo = results_df[0]['emotion']
                    sad_score = emo['sad']
                    dominant = results_df[0]['dominant_emotion']
                    
                    # 顯示分析結果卡片
                    st.success("全部分析完成！")
                    res_c1, res_c2 = st.columns(2)
                    res_c1.metric("主觀 PHQ-9 總分", f"{total_phq9} / 27")
                    res_c2.metric("AI 偵測悲傷值 (手動裁切)", f"{sad_score:.2f}%")
                    
                    # 綜合評估
                    severity = "輕微" if total_phq9 < 10 else "中度" if total_phq9 < 15 else "重度"
                    st.info(f"**綜合觀察：** 用戶自評憂鬱程度為「{severity}」( {total_phq9} 分)。AI 利用 MediaPipe 精準裁切後，辨識出的主導情緒為「{dominant}」。")
                    
                    # 顯示裁切後的臉部照片（確認 DeepFace 到底看到了什麼）
                    st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption="DeepFace 分析的裁切人臉", width=200)

                except Exception as e:
                    st.error(f"DeepFace 分析失敗。錯誤訊息：{e}")
    else:
        st.info("等待照片上傳以進行分析...")