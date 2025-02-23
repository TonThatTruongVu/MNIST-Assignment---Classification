import mlflow
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from data_processing import load_and_process_data

def evaluate_model(y_test, y_pred, model_name):
    """ Đánh giá mô hình với Accuracy, Precision, Recall, F1-score và vẽ Confusion Matrix """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Vẽ Confusion Matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    print(f"\n=== Classification Report - {model_name} ===\n")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1, cm

def train_model(X_train, X_val, X_test, y_train, y_val, y_test, model_type="Decision Tree"):
    """ Huấn luyện mô hình (Decision Tree hoặc SVM) và log kết quả với MLflow """
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    with mlflow.start_run(run_name=model_type):
        if model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=10, random_state=42)
        elif model_type == "SVM":
            model = SVC(kernel='rbf', C=1.0, gamma='scale')
        else:
            raise ValueError("Model type không hợp lệ")
        
        model.fit(X_train, y_train)
        
        # Dự đoán trên tập Validation trước
        y_pred_val = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        # Dự đoán trên tập Test
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1, cm = evaluate_model(y_test, y_pred, model_type)

        # Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_cv_accuracy = cv_scores.mean()

        print(f"{model_type} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, CV Accuracy: {mean_cv_accuracy:.4f}")

        # Lưu mô hình
        model_filename = f"{model_type.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_filename)

        # Logging với MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("Validation Accuracy", val_accuracy)
        mlflow.log_metric("Test Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-score", f1)
        mlflow.log_metric("Cross-Validation Accuracy", mean_cv_accuracy)
        mlflow.log_dict({"Cross-Validation Scores": cv_scores.tolist()}, f"cv_scores_{model_type.lower().replace(' ', '_')}.json")
        mlflow.sklearn.log_model(model, model_type.lower().replace(" ", "_"))
        mlflow.log_artifact(model_filename)
    
    return model

if __name__ == "__main__":
    # Đọc dữ liệu từ file CSV
    file_path = "mnist_test.csv"
    data = pd.read_csv(file_path)
    
    # Xử lý dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_process_data(data)
    
    # Huấn luyện mô hình
    train_model(X_train, X_val, X_test, y_train, y_val, y_test, model_type="Decision Tree")
    train_model(X_train, X_val, X_test, y_train, y_val, y_test, model_type="SVM")
