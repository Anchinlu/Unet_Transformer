# 🧠 ĐỒ ÁN TỐT NGHIỆP: PHÂN ĐOẠN KHỐI U NÃO VỚI MÔ HÌNH TRANSUNET TÙY BIẾN

Dự án này là kết quả nghiên cứu và thực nghiệm nhằm giải quyết bài toán phân đoạn khối u não (Brain Tumor Segmentation) trên ảnh cộng hưởng từ (MRI). Dự án tập trung khắc phục những nhược điểm cố hữu của mạng U-Net truyền thống bằng cách kết hợp sức mạnh của Vision Transformer và cơ chế Attention.

---

## 1. MỤC TIÊU ĐỒ ÁN
- **Bài toán:** Phân đoạn ảnh y tế (Medical Image Segmentation). Mục tiêu là tự động khoanh vùng chính xác vị trí và ranh giới của khối u não trên các lát cắt MRI.
- **Giá trị thực tiễn:** Hệ thống đóng vai trò như một trợ lý chẩn đoán thứ hai (Second Opinion), hỗ trợ các bác sĩ chẩn đoán hình ảnh khoanh vùng nhanh chóng các khu vực nghi ngờ, tiết kiệm thời gian phân tích thủ công và giảm thiểu rủi ro bỏ sót các khối u ở giai đoạn mầm mống.

## 2. KIẾN TRÚC MÔ HÌNH & TẬP DỮ LIỆU
- **Bộ dữ liệu:** Kaggle 3M Lower Grade Glioma (LGG) Segmentation Dataset (110 bệnh nhân, 3.929 cặp ảnh MRI đa kênh).
- **Mô hình nền tảng (Baseline):** U-Net (2015).
- **Kiến trúc đề xuất (TransUNet V2):** 
  - Khắc phục điểm mù cục bộ bằng cách thay thế toàn bộ khối CNN ở đáy mạng (Bottleneck) bằng kiến trúc **Vision Transformer (ViT)** để lấy thông tin toàn cục (Global Context).
  - Khắc phục nhiễu truyền dẫn bằng cách tích hợp **Cơ chế Cổng chú ý (Attention Gate)** vào các luồng kết nối tắt (Skip Connections).
  - Xử lý mất cân bằng lớp (Class Imbalance) bằng hàm mất mát **DiceBCELoss** kết hợp trọng số `pos_weight`.

## 3. CÁC PHẦN MỀM CẦN THIẾT ĐỂ TRIỂN KHAI (PREREQUISITES)
Để chạy được mã nguồn của đồ án, máy tính cần cài đặt sẵn các phần mềm và thư viện sau:
- **Python:** Phiên bản 3.9 hoặc mới hơn.
- **Git:** Để clone mã nguồn về máy.
- **PyTorch:** Framework Deep Learning chính (Khuyến nghị cài đặt bản có hỗ trợ CUDA/GPU để tăng tốc độ xử lý).
- **Các thư viện Python phụ trợ:** `numpy`, `matplotlib`, `pillow`, `customtkinter` (cho giao diện UI), `scikit-learn`.

*(Có thể sử dụng Anaconda hoặc Virtualenv để tạo môi trường ảo độc lập cho dự án)*

## 4. CÁCH THỨC CHẠY CHƯƠNG TRÌNH

### Bước 1: Tải mã nguồn về máy
```bash
git clone https://github.com/Anchinlu/Unet_Transformer.git
cd Unet_Transformer/src
```

### Bước 2: Cài đặt thư viện
```bash
pip install torch torchvision
pip install numpy matplotlib pillow customtkinter scikit-learn
```

### Bước 3: Chuẩn bị dữ liệu và Trọng số (Weights)
- Giải nén bộ dữ liệu Kaggle LGG vào thư mục `src/archive/`.
- Đảm bảo các file trọng số của mô hình (`unet_best_model.pth` và `transunet_best_model.pth`) đã được đặt đúng vào các thư mục tương ứng là `src/Unet/` và `src/Transunet/`.

### Bước 4: Chạy Giao diện Demo (UI)
Để mở giao diện người dùng trực quan, so sánh kết quả trực tiếp giữa U-Net và mô hình cải tiến:
```bash
python demo_app.py
```
*(Giao diện sẽ tự động tải các ảnh kiểm thử từ thư mục `src/archive/demo_favorites` và hiển thị kết quả phân đoạn, điểm số Dice/IoU cũng như Heatmap)*

### Bước 5: Huấn luyện lại mô hình (Training) - Tùy chọn
Nếu muốn tự huấn luyện lại mô hình từ đầu, sử dụng lệnh:
```bash
python train.py
```
*(Cấu hình thông số huấn luyện như số epochs, batch size, learning rate có thể được điều chỉnh trực tiếp bên trong mã nguồn hoặc thư mục `configs`)*

---
*Dự án hoàn thành tháng 06/2026.*
