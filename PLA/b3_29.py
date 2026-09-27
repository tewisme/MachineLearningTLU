import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None  # Trọng số (đã bao gồm bias ở w[0])

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Chèn thêm 1 cột toàn số 1 vào đầu ma trận X (để làm hệ số cho w[0] tức là bias)
        # Ví dụ: [-1, 4, 1] sẽ thành [1, -1, 4, 1]
        X = np.insert(X, 0, 1, axis=1)
        
        n_samples, n_features = X.shape
        
        # Khởi tạo trọng số w toàn số 0. Kích thước w giờ là số đặc trưng gốc + 1
        self.w = np.zeros(n_features)
        
        for _ in range(self.epochs):
            for i in range(n_samples):
                # Tính giá trị tuyến tính bằng duy nhất một phép nhân ma trận (không cần + b nữa)
                linear_output = np.dot(X[i], self.w)
                
                if y[i] * linear_output <= 0:
                    # Cập nhật cả trọng số và bias trong cùng một dòng code
                    self.w += self.learning_rate * y[i] * X[i]
                    
    def predict(self, X):
        X = np.array(X)
        # Khi dự đoán dữ liệu mới, cũng phải chèn thêm cột số 1 vào đầu
        X = np.insert(X, 0, 1, axis=1)
        
        linear_output = np.dot(X, self.w)
        return np.where(linear_output >= 0, 1, -1)

# ==========================================
# CHẠY THỬ
# ==========================================
X = [
    [-1,  4,  1],
    [ 2, -1,  0],
    [ 1,  1, -1]
]
y = [1, 1, -1]

model = Perceptron(learning_rate=0.1, epochs=10)
model.fit(X, y)

# w[0] chính là bias (b), w[1:] là trọng số thực sự của các đặc trưng
print("Trọng số w (với w[0] là bias):", model.w)

X_new = [[0, 2, 1], [2, 2, -2]]
predictions = model.predict(X_new)
print("Kết quả dự đoán:", predictions)
