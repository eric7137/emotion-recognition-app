import streamlit as st
import cv2
from deepface import DeepFace
from PIL import Image
import numpy as np

# 設置頁面佈局為寬版
st.set_page_config(layout="wide", page_title="憂鬱程度檢測展示")

st.title("臉部表情偵測與 PHQ-9 憂鬱量表對照展示")
st.write("請在上傳照片的同時填寫問卷，系統將對照主觀與客觀數據。")

# 建立左右兩欄
col1, col2 = st.columns([1, 1])

# --- 左側：影像上傳與 PHQ-9 問卷 ---
with col1:
    st.subheader("📋 用戶輸入區")
    
    # 影像上傳
    uploaded_file = st.file_uploader("上傳一張面部照片", type=['jpg', 'jpeg', 'png'])
    
    st.divider()
    
    # PHQ-9 問卷題目
    questions = [
        "做事提不起勁或沒有興趣",
        "感到心情低落、沮喪或絕望",
        "入睡困難、睡不安穩或睡太多",
        "感到疲倦或缺乏體力",
        "胃口不好或吃太多",
        "覺得自己很糟、或覺得自己很失敗",
        "專注困難（如看報紙或看電視時）",
        "動作或說話速度緩慢到別人都注意到了，或正好相反",
        "有想要傷害自己或自殺的想法"
    ]
    
    options = {
        "完全沒有": 0,
        "幾天": 1,
        "一半以上": 2,
        "幾乎天天": 3
    }
    
    phq9_scores = []
    for i, q in enumerate(questions):
        res = st.radio(f"Q{i+1}: {q}", options.keys(), horizontal=True, key=f"q{i}")
        phq9_scores.append(options[res])
    
    total_phq9 = sum(phq9_scores)
    submit = st.button("提交分析", type="primary", use_container_width=True)

# --- 右側：AI 分析結果 ---
with col2:
    st.subheader("🤖 AI 客觀分析結果")
    
    if uploaded_file is not None:
        # 顯示圖片
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳的影像", use_container_width=True)
        
        if submit:
            with st.spinner('AI 正在分析面部特徵...'):
                try:
                    # 將 PIL Image 轉為 OpenCV 格式
                    img_array = np.array(image.convert('RGB'))
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # 執行 DeepFace 分析
                    results = DeepFace.analyze(img_cv, actions=['emotion'], enforce_detection=True)
                    
                    # 取得分析數據
                    emo = results[0]['emotion']
                    sad_score = emo['sad']
                    neutral_score = emo['neutral']
                    dominant = results[0]['dominant_emotion']
                    
                    # 顯示分析結果卡片
                    st.success("分析完成！")
                    res_c1, res_c2 = st.columns(2)
                    res_c1.metric("主觀 PHQ-9 總分", f"{total_phq9} / 27")
                    res_c2.metric("AI 偵測悲傷值", f"{sad_score:.2f}%")
                    
                    # 視覺化圖表
                    st.bar_chart({"情緒權重 (%)": [emo['sad'], emo['neutral'], emo['fear'], emo['happy']]}, 
                                 x_label="情緒類別")
                    
                    # 綜合評估
                    st.info(f"**綜合觀察：** 用戶主觀感受為 {total_phq9} 分，AI 辨識主要情緒為「{dominant}」。")
                    
                except Exception as e:
                    st.error(f"分析失敗：請確保照片中臉部清晰。錯誤訊息：{e}")
    else:
        st.info("等待照片上傳以進行分析...")