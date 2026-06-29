# HƯỚNG DẪN SỬ DỤNG VÀ CHẠY DEMO CHƯƠNG TRÌNH
*(Đề tài: Phân đoạn khối u não với mô hình TransUNet tùy biến)*

Tài liệu này hướng dẫn chi tiết cách cài đặt môi trường, chuẩn bị dữ liệu và chạy các tính năng chính của chương trình (Giao diện Demo và Huấn luyện mô hình).

---

## 1. CÀI ĐẶT MÔI TRƯỜNG (ENVIRONMENT SETUP)

Chương trình yêu cầu **Python 3.9** trở lên. Việc cài đặt môi trường ảo (Virtual Environment) được đặc biệt khuyến nghị để tránh xung đột thư viện với các dự án khác trên máy của bạn.

### Bước 1: Mở Terminal / Command Prompt
Di chuyển đến thư mục chứa mã nguồn:
```bash
cd đường_dẫn_tới_thư_mục/DeTaiTotNghiep/src
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Tùy chọn)
- **Trên Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
- **Trên macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib pillow scikit-learn
pip install customtkinter
```
*(Lưu ý: Lệnh cài đặt `torch` ở trên dành cho các máy tính Windows có card đồ họa NVIDIA (CUDA 11.8). Nếu máy bạn không có card đồ họa rời, chỉ cần chạy lệnh `pip install torch torchvision` mặc định).*

---

## 2. CHUẨN BỊ DỮ LIỆU VÀ TRỌNG SỐ (WEIGHTS)

Để ứng dụng có thể dự đoán được ảnh, bạn cần cung cấp bộ trọng số mà nhóm đã huấn luyện sẵn, cũng như một số ảnh MRI để thử nghiệm.

1. **Trọng số mô hình (Pretrained Weights):**
   - File `unet_best_model.pth` phải được đặt trong thư mục `src/Unet/`
   - File `transunet_best_model.pth` phải được đặt trong thư mục `src/Transunet/`
2. **Dữ liệu hình ảnh (Dataset):**
   - Đảm bảo các bức ảnh dùng để chạy Demo được đặt tại đường dẫn: `src/archive/demo_favorites/` (Ứng dụng sẽ tự động quét thư mục này để lấy ảnh đưa lên giao diện).

---

## 3. CHẠY GIAO DIỆN DEMO CHUYÊN GIA

Giao diện Demo (GUI) được xây dựng bằng `customtkinter`, cho phép người dùng quan sát trực quan sự khác biệt giữa Ảnh gốc, Nhãn chuyên gia (Ground Truth), Kết quả phân đoạn của Baseline U-Net và Kết quả của TransUNet.

Từ thư mục `src/`, chạy lệnh sau:
```bash
python demo_app.py
```

**Cách sử dụng giao diện:**
- Ứng dụng sẽ tự động nạp ảnh đầu tiên trong thư mục `demo_favorites`.
- Nó sẽ tự động tính toán Điểm số Dice (Độ chính xác) và số lượng Pixel khối u để hiển thị lên màn hình.
- Nửa bên trái là **Ảnh MRI Gốc** và **Ground Truth**.
- Nửa bên phải hiển thị biểu đồ nhiệt (Heatmap) chỉ ra mức độ "tự tin" của từng mô hình, kèm theo ảnh nhị phân phân đoạn cắt lớp.
- Nhấn nút **"Đổi ảnh bệnh nhân"** (màu xanh ở góc phải) để chuyển sang xem phân đoạn của một ca bệnh khác.

---

## 4. HƯỚNG DẪN HUẤN LUYỆN LẠI MÔ HÌNH (TRAINING)

Nếu bạn cần huấn luyện lại mô hình từ đầu với thông số mới hoặc trên một tập dữ liệu khác, bạn có thể thực hiện thông qua script `train.py`.

### Bước 1: Cấu hình tham số
Mở file `src/configs/config.py` và điều chỉnh các thông số:
- `epochs`: Số vòng lặp huấn luyện (Mặc định: 150).
- `batch_size`: Kích thước lô (Mặc định: 16).
- `lr` (Learning Rate): Tốc độ học (Mặc định: 1e-4).
- `pos_weight`: Trọng số phạt cho lỗi dự đoán sai khối u (Mặc định: 15.0).
- `dataset_path_local`: Trỏ đến thư mục chứa Dataset Kaggle 3M gốc (Mặc định: `src/archive/kaggle_3m`).

### Bước 2: Chạy lệnh huấn luyện
```bash
python train.py
```
Quá trình huấn luyện sẽ mất từ vài tiếng đến vài ngày tùy thuộc vào cấu hình máy tính của bạn. Kết quả huấn luyện (File Loss, File Weights mới nhất) sẽ được lưu đè vào thư mục của mô hình tương ứng.

---
**Chúc bạn bảo vệ đồ án thành công!** 🎓
