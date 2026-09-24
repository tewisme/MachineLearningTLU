import numpy as np
import pandas as pd

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("data.csv")

# Lấy ma trận X và vector y dưới dạng numpy array
features = ["dien_tich_m2", "so_phong_ngu", "khoang_cach_tt_km", "mat_tien"]
X_raw = df[features].values
y_raw = df["gia_ty_vnd"].values.reshape(-1, 1)

# 2. Tự chia Train / Test thủ công (80% train, 20% test)
np.random.seed(42)
n_samples = len(X_raw)
indices = np.random.permutation(n_samples)

train_size = int(0.8 * n_samples)
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train_raw, y_train = X_raw[train_idx], y_raw[train_idx]
X_test_raw, y_test = X_raw[test_idx], y_raw[test_idx]


# Hàm tính các chỉ số đánh giá
def evaluate(y_true, y_pred, model_name="Model"):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    print(f"--- {model_name} ---")
    print(f"MAE : {mae:.4f} tỷ")
    print(f"RMSE: {rmse:.4f} tỷ")
    print(f"R²  : {r2:.4f}\n")


# ==========================================
# CÁCH 1: NORMAL EQUATION (NGHIỆM GIẢI TÍCH)
# w = (X^T * X)^(-1) * X^T * y
# ==========================================

# Thêm cột bias 1 vào ma trận X: [1, x1, x2, ...]
X_train_bias = np.c_[np.ones((X_train_raw.shape[0], 1)), X_train_raw]
X_test_bias = np.c_[np.ones((X_test_raw.shape[0], 1)), X_test_raw]

# Tính trọng số theta (gồm cả bias w0)
theta_ne = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

y_pred_ne = X_test_bias @ theta_ne
evaluate(y_test, y_pred_ne, "CÁCH 1: NORMAL EQUATION")


# ==========================================
# CÁCH 2: BATCH GRADIENT DESCENT (LẶP)
# ==========================================

# Chuẩn hóa Z-score (Feature Scaling) để Gradient Descent hội tụ nhanh
mean = np.mean(X_train_raw, axis=0)
std = np.std(X_train_raw, axis=0)

X_train_scaled = (X_train_raw - mean) / std
X_test_scaled = (X_test_raw - mean) / std

# Thêm bias term sau khi scale
X_train_gd = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_gd = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# Khởi tạo trọng số ngẫu nhiên
m = X_train_gd.shape[0]  # Số lượng mẫu
n = X_train_gd.shape[1]  # Số đặc trưng + bias
theta_gd = np.zeros((n, 1))

# Hyperparameters
learning_rate = 0.05
epochs = 1000

# Vòng lặp huấn luyện
for epoch in range(epochs):
    y_pred = X_train_gd @ theta_gd
    error = y_pred - y_train
    # Gradient của MSE loss: (2/m) * X^T * (X*w - y)
    gradients = (2 / m) * (X_train_gd.T @ error)
    theta_gd -= learning_rate * gradients

y_pred_gd = X_test_gd @ theta_gd
evaluate(y_test, y_pred_gd, "CÁCH 2: GRADIENT DESCENT")


# ==========================================
# DỰ ĐOÁN THỬ NGHIỆM CHO 1 CĂN NHÀ MỚI
# ==========================================
# Căn nhà: diện tích 75m2, 3 phòng ngủ, cách TT 5km, có mặt tiền (1)
nha_moi = np.array([[75, 3, 5.0, 1]])

# 1. Dự đoán bằng Normal Equation
nha_moi_bias = np.c_[np.ones((1, 1)), nha_moi]
gia_ne = (nha_moi_bias @ theta_ne)[0, 0]

# 2. Dự đoán bằng Gradient Descent (phải scale trước)
nha_moi_scaled = (nha_moi - mean) / std
nha_moi_gd = np.c_[np.ones((1, 1)), nha_moi_scaled]
gia_gd = (nha_moi_gd @ theta_gd)[0, 0]

print(f"Giá dự đoán (Normal Equation): {gia_ne:.2f} tỷ")
print(f"Giá dự đoán (Gradient Descent): {gia_gd:.2f} tỷ")
