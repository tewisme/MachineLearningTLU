import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("data.csv")

# 2. Tách Features (X) và Target (y)
X = df[["dien_tich_m2", "so_phong_ngu", "khoang_cach_tt_km", "mat_tien"]]
y = df["gia_ty_vnd"]

# 3. Phân chia tập Train / Test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Khởi tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Dự đoán trên tập Test
y_pred = model.predict(X_test)

# 6. Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"MAE  (Sai số tuyệt đối trung bình) : {mae:.3f} tỷ VNĐ")
print(f"RMSE (Căn bậc hai sai số toàn phương) : {rmse:.3f} tỷ VNĐ")
print(f"R² Score (Độ giải thích biến thiên)  : {r2:.4f}")

# 7. Xem hệ số hồi quy (Trọng số w và bias b)
print("\n=== HỆ SỐ HỒI QUY (WEIGHTS & BIAS) ===")
for col, coef in zip(X.columns, model.coef_):
    print(f"- {col:<20}: {coef:.4f}")
print(f"- Intercept (b)     : {model.intercept_:.4f}")

# 8. Thử nghiệm dự đoán một căn nhà mới
# Ví dụ: Diện tích 75m2, 3 phòng ngủ, cách trung tâm 5km, có mặt tiền (1)
nha_moi = pd.DataFrame(
    [[75, 3, 5.0, 1]],
    columns=["dien_tich_m2", "so_phong_ngu", "khoang_cach_tt_km", "mat_tien"],
)

gia_du_doan = model.predict(nha_moi)[0]
print(f"\n=> Giá dự đoán cho căn nhà mẫu: {gia_du_doan:.2f} tỷ VNĐ")
