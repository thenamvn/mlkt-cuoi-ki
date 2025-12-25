Methodology – Fraud Detection with Highly Imbalanced Data
=========================================================

1\. Xử lý mất cân bằng dữ liệu bằng Cost-Sensitive Learning
-----------------------------------------------------------

Trong bài toán phát hiện gian lận thẻ tín dụng, dữ liệu có tính **mất cân bằng nghiêm trọng**, trong đó số lượng giao dịch Fraud chiếm tỷ lệ rất nhỏ so với giao dịch Normal.

Thay vì sử dụng các phương pháp **resampling** như:

*   **SMOTE** (tạo dữ liệu Fraud giả có thể gây nhiễu),
    
*   hoặc **Undersampling** (loại bỏ dữ liệu thật),
    

phương pháp này lựa chọn **Cost-Sensitive Learning**, tức là **điều chỉnh trọng số sai số trực tiếp trong thuật toán học**.

### Cách thực hiện

Trong quá trình huấn luyện:

*   Tỷ lệ mất cân bằng được tính như sau:
    

scale\_pos\_weight\=#Normal#Fraudscale\\\_pos\\\_weight = \\frac{\\#Normal}{\\#Fraud}scale\_pos\_weight\=#Fraud#Normal​

*   Tham số này được sử dụng trong:
    
    *   `scale_pos_weight` đối với **XGBoost** và **LightGBM**
        
    *   `auto_class_weights='Balanced'` đối với **CatBoost**
        

### Nguyên lý hoạt động

Việc gán trọng số làm thay đổi **hàm mất mát (loss function)**:

*   Mô hình sẽ bị **phạt nặng hơn rất nhiều** nếu dự đoán sai một giao dịch Fraud
    
*   So với dự đoán sai một giao dịch Normal
    

Nhờ đó:

*   Mô hình buộc phải chú ý đến lớp thiểu số
    
*   Không làm méo phân phối dữ liệu gốc
    
*   Tránh rủi ro sinh ra các giao dịch Fraud “không tồn tại” như SMOTE
    

* * *

2\. Ensemble Learning – Voting Classifier (Soft Voting)
-------------------------------------------------------

### Động cơ

Một mô hình đơn lẻ dễ gặp các vấn đề:

*   Nhạy với nhiễu
    
*   Overfitting
    
*   Báo động giả (False Positive) ở các trường hợp biên
    

Để khắc phục, hệ thống sử dụng **Ensemble Learning** bằng cách kết hợp ba thuật toán Gradient Boosting mạnh nhất hiện nay:

*   **XGBoost**
    
*   **LightGBM**
    
*   **CatBoost**
    

### Cơ chế Soft Voting

*   Mỗi mô hình dự đoán **xác suất Fraud** cho một giao dịch
    
    *   Ví dụ: XGB = 0.7, LGBM = 0.6, CatBoost = 0.8
        
*   Voting Classifier tính **trung bình xác suất**:
    

(0.7+0.6+0.8)/3\=0.7(0.7 + 0.6 + 0.8) / 3 = 0.7(0.7+0.6+0.8)/3\=0.7

### Tác dụng

*   Giảm **variance (phương sai)** của mô hình
    
*   Nếu một mô hình dự đoán sai (False Positive), các mô hình còn lại có thể điều chỉnh lại quyết định cuối
    
*   Giúp tăng **độ ổn định** và **Precision**, đặc biệt quan trọng trong fraud detection
    

* * *

3\. Feature Engineering dựa trên hành vi người dùng
---------------------------------------------------

### Lý do

Fraud không được xác định bởi giá trị tuyệt đối, mà bởi **mức độ bất thường so với hành vi bình thường của người dùng và ngữ cảnh giao dịch**.  
Do đó, hệ thống tập trung xây dựng các **behavioral features** thay vì chỉ dùng dữ liệu thô.

### Các đặc trưng chính

#### 3.1 `amt_zscore`

*   Đo lường mức độ bất thường của số tiền giao dịch so với **lịch sử chi tiêu của chính chủ thẻ**
    
*   Ví dụ:
    
    *   Người thường chi tiêu ~$50 → giao dịch $500 là bất thường (fraud)
        
    *   Người thường chi tiêu ~$1000 → giao dịch $500 là bình thường
        

Z-score giúp chuẩn hóa hành vi chi tiêu theo từng người dùng, thay vì dùng ngưỡng cố định.

* * *

#### 3.2 `distance_km`

*   Tính khoảng cách địa lý giữa:
    
    *   Vị trí người dùng
        
    *   Vị trí merchant
        
*   Gian lận thường xảy ra:
    
    *   Ở xa vị trí quen thuộc
        
    *   Hoặc có sự di chuyển địa lý bất hợp lý trong thời gian ngắn
        

* * *

#### 3.3 Contextual Aggregation Features

*   So sánh số tiền giao dịch với:
    
    *   Trung bình theo **category**
        
*   Ví dụ:
    
    *   Giao dịch tạp hóa với số tiền rất lớn → bất thường
        

Các đặc trưng này giúp mô hình học **ngữ cảnh tiêu dùng**, không chỉ học con số.

* * *

4\. Tối ưu Threshold (Decision Threshold Optimization)
------------------------------------------------------

### Vấn đề với threshold mặc định

Mặc định, `model.predict()` sử dụng ngưỡng xác suất **0.5**, tuy nhiên:

*   Với dữ liệu mất cân bằng, ngưỡng này **không tối ưu**
    
*   Dễ gây nhiều False Positive hoặc bỏ sót Fraud
    

### Cách thực hiện

*   Hàm `find_optimal_threshold`:
    
    *   Duyệt toàn bộ ngưỡng từ 0 → 1
        
    *   Dựa trên **Precision–Recall Curve**
        
*   Mục tiêu:
    
    *   Tìm ngưỡng tối ưu sao cho **F1-score đạt cao nhất**
        

### Ý nghĩa thực tế

Ví dụ:

*   Threshold tối ưu = 0.8  
    → Mô hình chỉ báo Fraud khi độ chắc chắn > 80%  
    → Giảm đáng kể **False Positive (khóa thẻ nhầm)**
    

* * *

5\. Đánh giá mô hình (Model Evaluation)
---------------------------------------

### Chỉ số sử dụng

*   Precision
    
*   Recall
    
*   F1-score
    
*   **PR-AUC** (quan trọng hơn ROC-AUC trong bài toán imbalance)
    

### Cách đọc Confusion Matrix

*   **False Positive (góc trên bên phải)**  
    → Cần thấp để tránh khóa nhầm thẻ khách hàng
    
*   **True Positive (góc dưới bên phải)**  
    → Cần cao để bắt được gian lận thật
    

PR-AUC > 0.8 với dữ liệu mất cân bằng được xem là **mô hình rất tốt**.

* * *

6\. Data Sanitization (Làm sạch dữ liệu kỹ thuật)
-------------------------------------------------

Trong bước tiền xử lý, tên cột được chuẩn hóa bằng cách loại bỏ ký tự đặc biệt:

python

Copy code

`df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))`

### Mục đích

*   LightGBM lưu cấu trúc cây dưới dạng JSON
    
*   Tên cột chứa ký tự đặc biệt có thể gây lỗi
    
*   Bước này đảm bảo **tính ổn định kỹ thuật** cho pipeline huấn luyện
    

* * *

7\. Tổng kết
------------

Hệ thống sử dụng một kiến trúc **Hybrid** kết hợp:

> \*\*Cost-Sensitive Learning

*   Boosting Ensemble (XGB, LGBM, CatBoost)
    
*   Behavioral Feature Engineering
    
*   Threshold Optimization\*\*
    

Phương pháp này:

*   Không sinh dữ liệu giả
    
*   Phản ánh đúng hành vi thực tế
    
*   Giảm False Positive
    
*   Phù hợp triển khai trong môi trường production
    

👉 Đây là **phương pháp chính của dự án**, không phải thử nghiệm phụ.”