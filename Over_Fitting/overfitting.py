import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    # 1. Tải dữ liệu
    print("Đang tải dữ liệu...")
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # 2. Cố tình tạo overfitting bằng cách tạo ra 164 đặc trưng đa thức (bậc 3)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # 3. Chuẩn hóa dữ liệu (Bắt buộc khi có nhiều đặc trưng lớn/nhỏ khác nhau)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Huấn luyện bằng Linear Regression thông thường
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Dự đoán
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    # 5. In kết quả
    print("\n--- MÔ HÌNH OVERFITTING (Linear Regression) ---")
    print(f"Số lượng biến đầu vào đang dùng: {X_train_scaled.shape[1]} biến")
    print(f"R2 Score (Tập Train): {r2_score(y_train, train_preds):.4f}")
    
    # Điểm tập Test sẽ cực kỳ tệ (có thể âm) vì mô hình đã học vẹt các biến rác
    print(f"R2 Score (Tập Test) : {r2_score(y_test, test_preds):.4f}") 

if __name__ == "__main__":
    main()
