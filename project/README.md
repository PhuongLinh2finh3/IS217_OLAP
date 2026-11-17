# 🚖 HƯỚNG DẪN DỰ ÁN DỰ ĐOÁN CƯỚC TAXI (TAXI FARE PREDICTION)
> **Thư mục:** `project/`  
> *Vui lòng xem tài liệu đầy đủ chi tiết tại file [README.md chính ở thư mục gốc](../README.md).*

### ⚡ TÓM TẮT CÁC LỆNH CHẠY NHANH TẠI THƯ MỤC NÀY:

Đảm bảo bạn đang đứng tại thư mục `project/` và môi trường ảo đã kích hoạt:

```powershell
# 1. Tiền xử lý & tạo train/test parquet (chỉ cần chạy 1 lần)
python -m shared.preprocessing

# 2. Huấn luyện mô hình
# Bạn A:
python -m nhanh_A_lightgbm.run_lightgbm

# Bạn B:
python -m nhanh_B_randomforest.run_randomforest

# 3. So sánh và xuất biểu đồ
python -m shared.compare
```

Xem kết quả tại thư mục [reports/](file:///d:/UIT_DOCUMENTS/NAM_3/OLAP/TaxiDW_Project/project/reports).
