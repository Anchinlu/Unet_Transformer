# Kế hoạch thực hiện Đồ án: Phân đoạn khối u não với U-Net và Transformer

Tài liệu này ghi lại chi tiết các bước mà chúng ta sẽ thực hiện để hoàn thành đồ án, theo đúng định hướng "Tự code từ đầu (From Scratch)" bằng PyTorch.

## Giai đoạn 1: Chuẩn bị và Tiền xử lý dữ liệu (Data Pipeline)
**Mục tiêu:** Xây dựng bộ nạp dữ liệu (DataLoader) chuẩn bị cho việc huấn luyện trên Kaggle.
*   [x] **Dataset:** Tìm kiếm và tải tập dữ liệu MRI (ưu tiên BraTS hoặc bộ dữ liệu LGG Segmentation trên Kaggle).
*   **Thực hiện:**
    *   [x] Viết class `BrainTumorDataset` kế thừa từ `torch.utils.data.Dataset`.
    *   [x] Đọc file ảnh gốc và file mask.
    *   [ ] Tiền xử lý: Resize ảnh về chung kích thước (ví dụ 224x224 hoặc 256x256), chuẩn hóa pixel về dải [0, 1].
    *   [ ] Data Augmentation (nếu cần): Random rotation, flip để tăng tính đa dạng cho dữ liệu.
    *   [ ] Chia tập train, validation và test.

## Giai đoạn 2: Xây dựng Baseline Model (U-Net gốc)
**Mục tiêu:** Code mạng U-Net truyền thống để làm mốc so sánh (Baseline).
*   [x] **Thực hiện:**
    *   [x] Tự viết hàm cho khối `DoubleConv` (chứa 2 lớp Conv2d, BatchNorm2d và ReLU).
    *   [x] Xây dựng nhánh Mã hóa (Encoder) với các lớp MaxPool2d để giảm kích thước.
    *   [x] Xây dựng nhánh Giải mã (Decoder) với ConvTranspose2d để tăng kích thước.
    *   [x] Viết logic cho các kết nối tắt (Skip Connections) cắt và ghép tensor.

## Giai đoạn 3: Xây dựng Mô hình Cải tiến (TransUNet)
**Mục tiêu:** Tích hợp Vision Transformer vào kiến trúc U-Net.
*   **Thực hiện:**
    *   [x] **Patch Embedding:** Viết code chia ảnh/feature map thành các mảnh nhỏ (patches) và nhúng vào vector.
    *   [x] **Transformer Block:** Tự code hoặc sử dụng chuẩn lớp `nn.MultiheadAttention` của PyTorch để tạo khối Transformer Encoder. 
    *   [x] **Ghép nối:** Đưa đầu ra của nhánh CNN vào Transformer, sau đó reshape kết quả từ Transformer trả về nhánh Decoder của U-Net.

## Giai đoạn 4: Hàm mất mát (Loss Function) và Chỉ số đánh giá (Metrics)
**Mục tiêu:** Định nghĩa các tiêu chuẩn để đánh giá mô hình y tế.
*   **Thực hiện:**
    *   [x] Viết hàm `Dice Loss` và kết hợp với `BCE Loss` (Binary Cross Entropy).
    *   [x] Viết hàm tính toán `Dice Coefficient` (F1-score) và `IoU` (Intersection over Union).

## Giai đoạn 5: Vòng lặp Huấn luyện (Training Loop)
**Mục tiêu:** Viết code để huấn luyện mô hình trên Kaggle.
*   **Thực hiện:**
    *   [x] Khởi tạo DataLoader, Model, Optimizer (AdamW), Loss function.
    *   [x] Viết vòng lặp cho từng Epoch (Feedforward, Backpropagation, Optimizer step).
    *   [x] Tính toán loss và metrics trên tập Validation sau mỗi Epoch.
    *   [x] Lưu lại trọng số mô hình tốt nhất (`best_model.pth`).

## Giai đoạn 6: Đánh giá và Trực quan hóa (Evaluation & Visualization)
**Mục tiêu:** Vẽ biểu đồ để so sánh và đưa vào báo cáo.
*   **Thực hiện:**
    *   [ ] Vẽ biểu đồ Loss và Dice Score qua từng Epoch của cả 2 mô hình.
    *   [ ] Dùng `matplotlib` để in ra 3 cột: Ảnh MRI Gốc | Mask Thực tế (Ground Truth) | Mask Dự đoán của mô hình.
