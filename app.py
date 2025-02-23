import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import cv2
import joblib
import os
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import requests
import json

MLFLOW_SERVER_URL = "http://127.0.0.1:5000"  # Địa chỉ IP của MLFlow server

# Load mô hình đã huấn luyện
DT_MODEL_PATH = "decision_tree_model.pkl"
SVM_MODEL_PATH = "svm_model.pkl"
dt_model = joblib.load(DT_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)

# Giao diện Streamlit
st.title("Nhận diện chữ số viết tay")

option = st.radio("Chọn phương thức nhập ảnh:", ("Vẽ số", "Tải ảnh lên"))
model_choice = st.selectbox("Chọn mô hình dự đoán:", ["Decision Tree", "SVM"])

prediction = None

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) if img.shape[-1] == 4 else img
    gray = cv2.bitwise_not(gray)  # Đảo màu giống MNIST
    
    # Cắt vùng chứa số thực tế
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        gray = gray[y:y+h, x:x+w]
    
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)  # Resize về 28x28
    gray = gray.astype(np.uint8)  # Đảm bảo dữ liệu có dạng số nguyên 0-255
    gray = gray.reshape(1, -1)  # Chuyển thành vector 1x784
    return gray

def log_run_to_mlflow(prediction, option, model_choice, img):
    # Tạo experiment nếu chưa có
    experiment_name = "handwritten_digit_recognition"
    response = requests.post(f"{MLFLOW_SERVER_URL}/api/2.0/mlflow/experiments/create", json={"name": experiment_name})
    experiment_id = response.json()["experiment_id"]

    # Tạo run mới
    response = requests.post(f"{MLFLOW_SERVER_URL}/api/2.0/mlflow/runs/create", json={"experiment_id": experiment_id})
    run_id = response.json()["run"]["info"]["run_id"]

    # Ghi log tham số và metric
    log_metric_url = f"{MLFLOW_SERVER_URL}/api/2.0/mlflow/runs/log-metric"
    data = {
        "run_id": run_id,
        "key": "predicted_number",
        "value": prediction
    }
    requests.post(log_metric_url, json=data)

    log_param_url = f"{MLFLOW_SERVER_URL}/api/2.0/mlflow/runs/log-params"
    params = {
        "input_method": option,
        "model_used": model_choice
    }
    requests.post(log_param_url, json={"run_id": run_id, "params": params})

    # Ghi mô hình
    log_model_url = f"{MLFLOW_SERVER_URL}/api/2.0/mlflow/models/log-model"
    model_data = {
        "run_id": run_id,
        "name": "digit_classification_model",
        "artifact_path": "digit_classification_model",
        "model": dt_model if model_choice == "Decision Tree" else svm_model
    }
    requests.post(log_model_url, json=model_data)

# Xử lý đầu vào từ người dùng
if option == "Vẽ số":
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if canvas_result.image_data is not None:
        img = np.array(canvas_result.image_data, dtype=np.uint8)
        img = preprocess_image(img)
        prediction = dt_model.predict(img)[0] if model_choice == "Decision Tree" else svm_model.predict(img)[0]
        st.image(canvas_result.image_data, caption="Ảnh vẽ", use_container_width=True)

elif option == "Tải ảnh lên":
    uploaded_file = st.file_uploader("Tải ảnh số viết tay lên", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file).convert('L'))  # Chuyển ảnh thành grayscale
        img = preprocess_image(img)
        prediction = dt_model.predict(img)[0] if model_choice == "Decision Tree" else svm_model.predict(img)[0]
        st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)

if prediction is not None:
    st.write(f"### Dự đoán: {prediction}")
    
    # Ghi log vào MLflow qua REST API
    log_run_to_mlflow(prediction, option, model_choice, img)
    
    # Nếu dự đoán sai, cho phép sửa
    correct_label = st.number_input("Nếu sai, hãy nhập lại số đúng:", min_value=0, max_value=9, step=1, value=int(prediction))
    if st.button("Lưu dữ liệu để cải thiện mô hình"):
        save_path = "training_data/"
        os.makedirs(save_path, exist_ok=True)
        label_path = os.path.join(save_path, str(correct_label))
        os.makedirs(label_path, exist_ok=True)
        filename = f"{label_path}/{int(time.time())}.png"
        cv2.imwrite(filename, img.reshape(28, 28))
        st.success(f"Dữ liệu đã được lưu vào {filename}!")
