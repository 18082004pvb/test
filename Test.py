# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu từ file CSV
car_data = pd.read_csv('Car Price Prediction.csv')

# Làm sạch dữ liệu
# 1. Xử lý cột Levy
car_data['Levy'] = car_data['Levy'].replace('-', np.nan).astype(float)
car_data['Levy'] = car_data['Levy'].fillna(car_data['Levy'].mean())

# 2. Chuyển đổi Mileage sang kiểu số
car_data['Mileage'] = car_data['Mileage'].str.replace(' km', '').str.replace(',', '').astype(float)

# 3. Tách thông tin Turbo từ Engine volume
car_data['Turbo'] = car_data['Engine volume'].str.contains('Turbo', na=False).astype(int)
car_data['Engine volume'] = car_data['Engine volume'].str.extract(r'([\d.]+)').astype(float)

# 4. Chuẩn hóa số cửa
car_data['Doors'] = car_data['Doors'].str.extract(r'(\d)').astype(float)

# Lựa chọn các cột quan trọng
selected_columns = ['Levy', 'Prod. year', 'Mileage', 'Cylinders', 'Engine volume',
                    'Turbo', 'Fuel type', 'Gear box type', 'Drive wheels', 'Category', 'Leather interior']
data = car_data[selected_columns]

# Xử lý target (Price)
car_data['Price'] = car_data['Price'].replace('-', np.nan).astype(float)
car_data.dropna(subset=['Price'], inplace=True)  # Xóa dòng thiếu giá trị Price
target = car_data['Price']

# Mã hóa các cột phân loại
categorical_features = ['Fuel type', 'Gear box type', 'Drive wheels', 'Category', 'Leather interior']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(data[categorical_features])

# Gộp dữ liệu sau mã hóa với dữ liệu số
numerical_features = ['Levy', 'Prod. year', 'Mileage', 'Cylinders', 'Engine volume', 'Turbo']
numerical_data = data[numerical_features].values

final_data = np.hstack([numerical_data, encoded_data])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
final_data = scaler.fit_transform(final_data)

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(final_data, target, test_size=0.2, random_state=42)

# Lưu trữ kết quả đánh giá của từng biến
results = []

# Lấy tên các đặc trưng
feature_names = numerical_features + list(encoder.get_feature_names_out(categorical_features))

# Lặp qua từng biến
for i, feature in enumerate(feature_names):
    # Lấy cột tương ứng
    X_single = final_data[:, i].reshape(-1, 1)  # Biến đơn lẻ

    # Tách dữ liệu thành tập huấn luyện và kiểm tra
    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single, target, test_size=0.2, random_state=42)

    # Xây dựng mô hình hồi quy tuyến tính
    single_model = LinearRegression()
    single_model.fit(X_train_single, y_train_single)

    # Dự đoán
    y_pred_single = single_model.predict(X_test_single)

    # Đánh giá mô hình
    mse_single = mean_squared_error(y_test_single, y_pred_single)
    r2_single = r2_score(y_test_single, y_pred_single)

    # Lưu kết quả
    results.append({
        'Feature': feature,
        'Mean Squared Error': mse_single,
        'R^2 Score': r2_single,
        'Coefficient': single_model.coef_[0]  # Hệ số hồi quy của biến
    })

# Chuyển kết quả thành DataFrame để hiển thị
results_df = pd.DataFrame(results)

# Sắp xếp theo R^2 Score để xem biến quan trọng nhất
results_df = results_df.sort_values(by='R^2 Score', ascending=False)

# Hiển thị kết quả
print(results_df)
