import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

def main():
    # 1. Tải và chuẩn bị dữ liệu (Giống hệt file overfitting.py)
    data = fetch_california_housing()
    X = data.data
    y = data.target

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Huấn luyện bằng Lasso Regression
    # Cài max_iter cao để đảm bảo thuật toán đủ thời gian ép các biến về 0
    print("Đang huấn luyện mô hình Lasso (Quá trình này có thể mất vài giây)...")
    lasso_model = Lasso(alpha=0.05, max_iter=10000, random_state=42)
    lasso_model.fit(X_train_scaled, y_train)

    # Dự đoán
    train_preds = lasso_model.predict(X_train_scaled)
    test_preds = lasso_model.predict(X_test_scaled)

    # 3. Phân tích tác dụng của Lasso
    zero_weights = np.sum(lasso_model.coef_ == 0)
    total_weights = len(lasso_model.coef_)

    print("\n--- MÔ HÌNH GOOD FIT (Lasso Regression) ---")
    print(f"R2 Score (Tập Train): {r2_score(y_train, train_preds):.4f}")
    # Điểm Test sẽ trở lại mức ổn định và dương
    print(f"R2 Score (Tập Test) : {r2_score(y_test, test_preds):.4f}")
    
    print("\n--- BÁO CÁO DỌN DẸP ---")
    print(f"Tổng số biến đầu vào ban đầu: {total_weights}")
    print(f"Số biến bị Lasso đánh giá là RÁC và ép về 0: {zero_weights}")
    print(f"Số biến thực sự có giá trị được giữ lại: {total_weights - zero_weights}")

if __name__ == "__main__":
    main()
