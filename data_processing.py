import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

def load_and_process_data(data):  # Nhận DataFrame thay vì file path
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Dữ liệu đầu vào phải là pandas DataFrame.")
    
    # Kiểm tra nếu cột 'label' tồn tại
    if 'label' not in data.columns:
        raise ValueError("Dữ liệu không chứa cột 'label'. Kiểm tra lại file đầu vào.")
    
    # Tách đặc trưng và nhãn
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Loại bỏ các cột chứa toàn giá trị NaN
    X = X.dropna(axis=1, how='all')
    
    # Lấy danh sách cột số
    numeric_columns = X.select_dtypes(include=['number']).columns
    
    # Khởi tạo và fit imputer trước khi chia dữ liệu
    imputer = SimpleImputer(strategy="median")
    X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
    
    # Kiểm tra và loại bỏ các dòng trùng lặp
    data = data.drop_duplicates()
    
    # Xử lý các giá trị ngoại lai bằng phương pháp IQR
    Q1 = X[numeric_columns].quantile(0.25)
    Q3 = X[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X[numeric_columns] < (Q1 - 1.5 * IQR)) | (X[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Nếu quá nhiều dữ liệu bị loại bỏ, giữ nguyên dữ liệu gốc
    if mask.sum() / len(X) > 0.1:
        X = X[mask]
        y = y[mask]
    
    # Chuẩn hóa dữ liệu (min-max scaling)
    X[numeric_columns] = (X[numeric_columns] - X[numeric_columns].min()) / (X[numeric_columns].max() - X[numeric_columns].min())
    
    # Điền giá trị NaN còn sót lại bằng 0
    X.fillna(0, inplace=True)
    
    # Kiểm tra giá trị NaN trong y và xử lý
    if y.isna().sum() > 0:
        mode_value = y.mode()
        y.fillna(mode_value[0] if not mode_value.empty else 0, inplace=True)
    
    # Kiểm tra số lượng nhãn trước khi chia dữ liệu
    label_counts = y.value_counts()
    print("Phân phối nhãn:\n", label_counts)
    
    if len(label_counts) < 2:
        print("Cảnh báo: Dữ liệu chỉ có một nhãn. Sẽ chia dữ liệu mà không sử dụng stratify.")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        except ValueError:
            print("Cảnh báo: Không thể sử dụng stratify do số lượng nhãn quá ít. Chia dữ liệu ngẫu nhiên.")
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Đảm bảo không có run nào đang mở trước khi bắt đầu một run mới
    if mlflow.active_run():
        mlflow.end_run()
    
    # Logging dữ liệu với MLflow
    with mlflow.start_run(run_name="Data Processing"):
        mlflow.log_param("Training Size", len(X_train))
        mlflow.log_param("Validation Size", len(X_val))
        mlflow.log_param("Test Size", len(X_test))
        mlflow.log_param("Random State", 42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
