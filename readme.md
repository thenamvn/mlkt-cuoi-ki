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
    


# Dựa vào biểu đồ Feature Importance và logic nghiệp vụ phát hiện gian lận (Fraud Detection), việc 6 đặc trưng này đứng đầu là **hoàn toàn hợp lý**. Chúng phản ánh chính xác tâm lý và hành vi của kẻ gian lận.

Dưới đây là giải thích chi tiết tại sao chúng lại quan trọng đến vậy:

### 1. `amt` (Số tiền) & `amt_log` (Log số tiền) - Vị trí Top 1 & 2
*   **Lý do:** Đây là động cơ chính của gian lận. Kẻ gian thường có 2 xu hướng:
    *   **Rút cạn hạn mức:** Thực hiện các giao dịch giá trị rất lớn (mua đồ điện tử, trang sức) để tẩu tán tiền nhanh nhất có thể trước khi thẻ bị khóa.
    *   **Test thẻ:** Thực hiện giao dịch rất nhỏ để xem thẻ còn sống không.
*   **Tại sao quan trọng:** `amt` là tín hiệu trực tiếp nhất. `amt_log` giúp mô hình xử lý tốt hơn sự chênh lệch quá lớn giữa giao dịch 1$ và 10.000$ (giảm độ lệch - skewness), giúp thuật toán hội tụ nhanh hơn.

### 2. `category` (Loại hình kinh doanh) - Vị trí Top 3
*   **Lý do:** Gian lận không xảy ra ngẫu nhiên. Kẻ gian thường nhắm vào các loại hình dễ thanh khoản (bán lại lấy tiền mặt) hoặc khó truy vết.
*   **Ví dụ:**
    *   **Rủi ro cao:** Mua sắm trực tuyến (online shopping), đồ điện tử, trang sức, thẻ quà tặng.
    *   **Rủi ro thấp:** Thanh toán tiền điện nước, đi siêu thị mua rau, đổ xăng (tùy ngữ cảnh).
*   **Tại sao quan trọng:** Mô hình học được rằng "Nếu giao dịch thuộc nhóm `misc_net` hoặc `shopping_net`, xác suất lừa đảo cao hơn hẳn so với `grocery_pos`".

### 3. `category_mean_amt` (Số tiền trung bình của loại hình đó) - Vị trí Top 4
*   **Lý do:** Đây là **ngữ cảnh (Context)**. Nó cho mô hình biết "bình thường người ta tiêu bao nhiêu ở chỗ này".
*   **Ví dụ:** Trung bình một lần đi `gas_transport` (đổ xăng) là 50$. Trung bình mua `grocery` (tạp hóa) là 100$.
*   **Tại sao quan trọng:** Nó làm nền tảng để so sánh cho feature tiếp theo.

### 4. `amt_vs_category_mean` (Tỷ lệ số tiền / Trung bình loại hình) - Vị trí Top 5
*   **Lý do:** Đây là feature **phát hiện bất thường (Anomaly Detection)** mạnh nhất.
*   **Ví dụ:**
    *   Bạn mua cà phê (`food_dining`), trung bình mọi người tiêu 5$.
    *   Đột nhiên có một giao dịch 500$ tại quán cà phê đó.
    *   => `amt` (500) / `category_mean` (5) = **100 lần**.
*   **Tại sao quan trọng:** Con số 500$ nếu mua Tivi thì bình thường, nhưng mua cà phê là lừa đảo. Feature này giúp mô hình hiểu được sự **vô lý** của giao dịch trong ngữ cảnh cụ thể.

### 5. `hour` (Giờ giao dịch) - Vị trí Top 6
*   **Lý do:** Thói quen sinh hoạt của con người và kẻ gian khác nhau.
*   **Hành vi:**
    *   Người thật: Thường ngủ từ 11h đêm đến 6h sáng. Giao dịch chủ yếu giờ hành chính hoặc buổi tối.
    *   Kẻ gian (hoặc Hacker quốc tế): Thường hoạt động vào khung giờ "chết" (2h - 4h sáng) khi nạn nhân đang ngủ để không nhận được thông báo biến động số dư ngay lập tức, hoặc do lệch múi giờ.
*   **Tại sao quan trọng:** Một giao dịch mua hàng hiệu lúc 3 giờ sáng là tín hiệu đỏ cực lớn.

### Tóm lại
Mô hình của bạn đang hoạt động rất "thông minh". Nó không chỉ nhìn vào số tiền (`amt`), mà nó đang so sánh số tiền đó với ngữ cảnh (`category`, `amt_vs_category_mean`) và thời gian (`hour`). Đây chính là lý do tại sao độ chính xác (Precision) của bạn đạt tới 93%.

# Giải thích chi tiết cơ chế tìm threshold và lý do tại sao threshold có thể cao như vậy:

### 1. Cơ chế tìm Threshold của hàm `find_optimal_threshold`

Hàm này hoạt động như sau:
1.  Nó thử **tất cả** các ngưỡng có thể từ 0 đến 1.
2.  Tại mỗi ngưỡng, nó tính **F1-Score** (trung bình điều hòa giữa Precision và Recall).
3.  Nó chọn ngưỡng nào làm cho F1-Score cao nhất.

**Công thức:** $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

### 2. Tại sao Threshold lại cao đến 96%?

Lý do nằm ở chất lượng mô hình của bạn: **Mô hình quá tự tin và phân loại quá tốt.**

*   **Phân tách rõ ràng:** Mô hình đã học được cách phân biệt Normal và Fraud cực kỳ rạch ròi.
    *   Với giao dịch bình thường (Normal), mô hình dự đoán xác suất lừa đảo cực thấp (ví dụ: 0.001, 0.05).
    *   Với giao dịch lừa đảo (Fraud), mô hình dự đoán xác suất cực cao (ví dụ: 0.98, 0.99).
*   **Khoảng trống ở giữa:** Có rất ít các điểm dữ liệu "lập lờ" nằm ở khoảng giữa (ví dụ 0.4 - 0.6).

Khi mô hình đẩy xác suất của Fraud lên rất cao (gần 1.0), thì điểm cắt tối ưu (nơi F1 cao nhất) sẽ tự động trượt lên cao theo để loại bỏ tối đa các ca nghi ngờ sai (False Positives).

### 3. Minh họa dễ hiểu

Hãy tưởng tượng thang điểm từ 0 đến 100:
*   **Mô hình yếu:** Kẻ gian được chấm 60 điểm, người thường 40 điểm. Bạn phải đặt ngưỡng ở 50 để phân loại.
*   **Mô hình của bạn (Mạnh):** Kẻ gian được chấm 99 điểm, người thường chỉ 1-2 điểm.
    *   Lúc này, bạn đặt ngưỡng ở 50 cũng được, 80 cũng được, hay **96** cũng được.
    *   Tuy nhiên, thuật toán chọn 96 vì có thể có một vài giao dịch bình thường "hơi lạ" bị chấm 90 điểm. Đặt ngưỡng 96 giúp loại bỏ những ca "hơi lạ" này (tăng Precision) mà vẫn giữ được hầu hết kẻ gian (Recall vẫn cao).

### 4. Có cách nào khác để chọn Threshold không?

Nếu bạn không muốn dùng F1-Score (cân bằng), bạn có thể chọn Threshold theo mục tiêu kinh doanh (Business Metric):

**Cách 1: Ưu tiên bắt hết lừa đảo (Maximize Recall)**
Nếu ngân hàng chấp nhận gọi điện làm phiền khách hàng (False Positive) miễn là không để lọt kẻ gian.
*   *Code:* Chọn threshold sao cho Recall >= 0.95.

**Cách 2: Ưu tiên độ chính xác (Maximize Precision)**
Nếu ngân hàng sợ khách hàng giận vì khóa thẻ nhầm.
*   *Code:* Chọn threshold sao cho Precision >= 0.99.

**Code thay thế để bạn thử nghiệm (Thêm vào file notebook):**

```python
def find_threshold_by_recall(y_true, y_proba, target_recall=0.90):
    """
    Tìm ngưỡng để đạt được Recall mong muốn (ví dụ bắt được 90% kẻ gian)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Tìm vị trí mà Recall vừa đủ >= target_recall
    # Recalls thường được sắp xếp giảm dần, nên ta tìm điểm gần nhất
    idx = np.argmin(np.abs(recalls - target_recall))
    
    selected_threshold = thresholds[idx]
    print(f"\n>>> Threshold for {target_recall*100}% Recall: {selected_threshold:.4f}")
    print(f"    Corresponding Precision: {precisions[idx]:.4f}")
    
    return selected_threshold

# Gọi hàm thử
# threshold_recall = find_threshold_by_recall(y_test, y_pred_proba, target_recall=0.95)
```

**Kết luận:** Threshold 96% chứng tỏ Feature Engineering của bạn (đặc biệt là các feature so sánh hành vi) cực kỳ hiệu quả, khiến mô hình rất tự tin khi bắt gian lận. Bạn có thể yên tâm sử dụng.


# Lí giải tối ưu tốc độ tính toán và xử lí imbalance
Dưới đây là 2 "thủ phạm" chính khiến đoạn code mẫu kia chậm hơn code của bạn hàng chục, thậm chí hàng trăm lần:

### 1. "Thủ phạm" lớn nhất: SMOTE (Imbalanced-learn) vs. Class Weights
*   **Code mẫu (Chậm):** Sử dụng `SMOTETomek`.
    ```python
    # Dòng này là "tử huyệt" về tốc độ trên dữ liệu lớn
    smote = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    ```
    *   **Cơ chế:** Nó phải tính toán khoảng cách giữa các điểm dữ liệu (K-Nearest Neighbors) để sinh ra các điểm dữ liệu giả (fake data) nhằm cân bằng số lượng Fraud và Normal.
    *   **Hậu quả:** Với dữ liệu hàng trăm ngàn dòng như `fraudTrain.csv`, việc tính toán khoảng cách này cực kỳ tốn kém tài nguyên và thời gian. Nó còn làm tăng kích thước dữ liệu đầu vào cho model.

*   **Code của bạn (Nhanh):** Sử dụng `scale_pos_weight`.
    ```python
    clf_xgb = xgb.XGBClassifier(..., scale_pos_weight=scale_weight, ...)
    ```
    *   **Cơ chế:** Không sinh ra dữ liệu mới. Nó chỉ đơn giản là nhân một hệ số phạt vào công thức toán học khi tính lỗi (Loss Function).
    *   **Hậu quả:** Tốc độ xử lý **nhanh như không xử lý imbalance**, tốn 0 giây để chuẩn bị.

### 2. Sklearn GradientBoosting vs. LightGBM/XGBoost
*   **Code mẫu (Chậm):** Sử dụng `sklearn.ensemble.GradientBoostingClassifier`.
    *   Đây là thư viện cũ của Sklearn, chủ yếu chạy đơn luồng (hoặc tối ưu kém hơn) và không sử dụng kỹ thuật Histogram-based để tăng tốc.
*   **Code của bạn (Nhanh):** Sử dụng `LightGBM` và `XGBoost`.
    *   Hai thư viện này được viết bằng C++, tối ưu hóa đến tận "chân răng" cho phần cứng, sử dụng đa luồng (`n_jobs=-1`) và thuật toán Histogram để xử lý dữ liệu cực nhanh.

### Tóm lại
Cách tiếp cận này **chuẩn công nghiệp (Industry Standard)** hiện nay:
1.  Dùng **LightGBM/XGBoost** (Nhanh & Mạnh).
2.  Dùng **Class Weights** thay vì SMOTE (Hiệu quả & Không tốn tài nguyên).
3.  Feature Engineering tập trung (Vectorized) thay vì tính toán thống kê rườm rà.
