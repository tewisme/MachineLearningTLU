import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 300

# Tạo các đặc trưng giả lập thực tế
dien_tich = np.random.uniform(30, 150, n_samples).round(1)
so_phong_ngu = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.2, 0.4, 0.3, 0.1])
khoang_cach_tt = np.random.uniform(1.0, 18.0, n_samples).round(2)
mat_tien = np.random.binomial(1, 0.35, n_samples)  # 1: mặt tiền, 0: trong ngõ

# Công thức giá kèm nhiễu (noise) thực tế
gia_ty_vnd = (
    0.045 * dien_tich
    + 0.25 * so_phong_ngu
    - 0.18 * khoang_cach_tt
    + 1.2 * mat_tien
    + np.random.normal(0, 0.4, n_samples)  # noise
).round(2)

# Đảm bảo không có giá âm hoặc phi thực tế
gia_ty_vnd = np.clip(gia_ty_vnd, 0.8, None)

# Gom thành DataFrame và lưu file
df = pd.DataFrame(
    {
        "dien_tich_m2": dien_tich,
        "so_phong_ngu": so_phong_ngu,
        "khoang_cach_tt_km": khoang_cach_tt,
        "mat_tien": mat_tien,
        "gia_ty_vnd": gia_ty_vnd,
    }
)

df.to_csv("data.csv", index=False)
print("Đã tạo file nha_dat_300_rows.csv thành công!")
print(df.head())
