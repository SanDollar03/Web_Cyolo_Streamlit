import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# CSSを追加
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        padding: 10px;
        font-family: Arial, sans-serif;
    }
    .container {
        max-width: 1200px;
        margin: auto;
        padding: 0px;
        background-color: #ffffff;
        border-radius: 0px;
        border: 1px solid #dddddd;
    }
    .footer {
        margin-top: 10px;
        text-align: center;
        color: #888888;
        font-size: 12px;
    }
    .stApp {
        padding: 0px;
    }
    .block-container {
        padding: 0px;
    }
    .css-1e5imcs {
        padding: 0px;
    }
    .element-container {
        margin-bottom: 0px;
    }
    .title-container {
        text-align: center;
        padding: 10px 0;
    }
    .right-align {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
    }
    .sidebar-text {
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# サイドバーのタイトル画像の表示
st.sidebar.image("title.jpg", use_column_width=True)

# セッションステートの初期化
if 'camera_id' not in st.session_state:
    st.session_state.camera_id = 0
if 'model_name' not in st.session_state:
    st.session_state.model_name = 'yolov8n.pt'
if 'custom_model_path' not in st.session_state:
    st.session_state.custom_model_path = None
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'camera_resolution' not in st.session_state:
    st.session_state.camera_resolution = (640, 480)  # デフォルトのカメラ解像度

# 接続時にCSVファイルを削除して新たに生成
log_file = os.path.join("csv", 'detection_log.csv')
if not os.path.exists("csv"):
    os.makedirs("csv")
if os.path.exists(log_file):
    os.remove(log_file)

# カメラIDとモデル選択の初期設定
camera_id = st.sidebar.selectbox('Select Camera ID', [0, 1, 2, 3], index=st.session_state.camera_id, key='camera_id')
model_name = st.sidebar.selectbox(
    'Select YOLO Model', 
    ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 'Custom Model'],
    index=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 'Custom Model'].index(st.session_state.model_name),
    key='model_name'
)

# カスタムモデルのアップロード
uploaded_file = st.sidebar.file_uploader("Upload Custom Model", type=['pt'])
if uploaded_file is not None:
    if not os.path.exists("models"):
        os.makedirs("models")
    custom_model_path = os.path.join("models", uploaded_file.name)
    with open(custom_model_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.custom_model_path = custom_model_path
    st.session_state.model_name = 'Custom Model'

# モデルの種類の解説テキスト
st.sidebar.markdown("""
<div class="sidebar-text">
    <h3>YOLO Model Types:</h3>
    <p>- <b>yolov8n.pt</b>: Nano model, fastest but less accurate.</p>
    <p>- <b>yolov8s.pt</b>: Small model, good balance of speed and accuracy.</p>
    <p>- <b>yolov8m.pt</b>: Medium model, moderate speed and accuracy.</p>
    <p>- <b>yolov8l.pt</b>: Large model, slower but more accurate.</p>
    <p>- <b>yolov8x.pt</b>: Extra-large model, slowest but most accurate.</p>
    <p>- <b>Custom Model</b>: Upload your own YOLOv8 model.</p>
</div>
""", unsafe_allow_html=True)

# モデルのロード
if st.session_state.model_name == 'Custom Model' and st.session_state.custom_model_path is not None:
    model = YOLO(st.session_state.custom_model_path)
else:
    model = YOLO(st.session_state.model_name)

# カメラキャプチャの開始
cap = cv2.VideoCapture(st.session_state.camera_id)  # 選択されたカメラIDを使用

# カメラの解像度を取得
st.session_state.camera_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# カメラウィンドウと散布図のコンテナ
st.markdown('<div class="right-align">', unsafe_allow_html=True)
stframe = st.empty()
st.markdown('</div>', unsafe_allow_html=True)
scatter_plot = st.empty()

# ログデータを保存する関数
def save_log_and_update_plot(log_data):
    log_df = pd.DataFrame(log_data, columns=['Class ID', 'Class Name', 'DateTime', 'X', 'Y'])
    log_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    update_scatter_plot()

# 散布図を表示する関数
def update_scatter_plot():
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        plt.rcParams.update({'font.size': 8})  # フォントサイズを小さく設定
        fig, ax = plt.subplots()
        scatter = ax.scatter(log_df['X'], -log_df['Y'], c=log_df['Class ID'], cmap='viridis', alpha=0.6, s=10)  # Y座標を反転し、点のサイズを小さく設定
        ax.set_xlim(0, st.session_state.camera_resolution[0])
        ax.set_ylim(-st.session_state.camera_resolution[1], 0)
        ax.yaxis.tick_right()  # Y軸を右側に配置
        ax.yaxis.set_label_position("right")  # Y軸の名前を右側に配置
        # Create a legend with class names
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
        legend_labels = []
        for label in labels:
            try:
                class_id = int(float(label))  # Fix to handle float conversion
                legend_labels.append(model.names[class_id])
            except ValueError:
                legend_labels.append(label)
        ax.legend(handles, legend_labels, title="Class Name")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Scatter Plot of Detections')
        scatter_plot.pyplot(fig)

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break

    # 画像の検出
    results = model.predict(frame, device='cpu')  # CPUで処理

    # 検出結果の描画とログの作成
    for result in results:
        annotated_frame = result.plot()
        for box in result.boxes:
            x_center = (box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2
            y_center = (box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2
            class_id = box.cls[0].item()
            class_name = model.names[int(class_id)]
            log_entry = [class_id, class_name, datetime.now().strftime("%Y%m%d%H%M%S"), x_center, y_center]
            st.session_state.log_data.append(log_entry)

    # ログデータが10行以上になったら保存
    if len(st.session_state.log_data) >= 10:
        save_log_and_update_plot(st.session_state.log_data)
        st.session_state.log_data = []

    # Streamlitに画像を表示
    stframe.image(annotated_frame, channels="BGR")

    # 散布図の更新
    update_scatter_plot()

    # カメラIDとモデルの選択が変更された場合の処理
    if camera_id != st.session_state.camera_id:
        st.session_state.camera_id = camera_id
        cap.release()
        cap = cv2.VideoCapture(camera_id)
    if model_name != st.session_state.model_name:
        st.session_state.model_name = model_name
        if st.session_state.model_name == 'Custom Model' and st.session_state.custom_model_path is not None:
            model = YOLO(st.session_state.custom_model_path)
        else:
            model = YOLO(st.session_state.model_name)

cap.release()

# 残りのログデータを保存
if st.session_state.log_data:
    save_log_and_update_plot(st.session_state.log_data)

# フッター
st.markdown('<div class="footer"><p>&copy; 2024 Your Company. All rights reserved.</p></div>', unsafe_allow_html=True)
