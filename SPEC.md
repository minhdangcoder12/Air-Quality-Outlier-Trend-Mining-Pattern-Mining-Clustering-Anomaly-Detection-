# Spec — Air Quality Outlier & Trend Mining

Ghi lại các quyết định thiết kế và lý do. Số liệu lấy từ lần chạy notebook ngày 13/11/2026 (`reports/figures/summary.json`).

## Mục tiêu

Đề 17: tiền xử lý chuỗi thời gian (nội suy giá trị thiếu, khử nhiễu); phát hiện đỉnh ô nhiễm bằng phát hiện bất thường chuỗi thời gian; khai phá mẫu tuần tự ô nhiễm theo mùa/giờ; phân cụm để nhóm các đơn vị có hồ sơ ô nhiễm giống nhau.

## Ánh xạ yêu cầu đề bài → dataset

| Đề bài | Dataset | Ghi chú |
|---|---|---|
| Chuỗi ô nhiễm | `CO(GT)`, `NOx(GT)`, `NO2(GT)`, `C6H6(GT)` theo giờ | Máy phân tích chuẩn; 9.357 giờ, 10/03/2004 – 04/04/2005 |
| Giá trị thiếu | `-200` và mốc giờ không có | Đổi thành NaN; lưới giờ liên tục bằng `asfreq("h")` |
| Biến ngữ cảnh | `T`, `RH`, `AH` | Chỉ dùng để mô tả cụm, không đưa vào phân cụm |
| Trạm quan trắc | **Không có** — chỉ 1 thiết bị | Thay bằng **ngày** làm “trạm ảo” (hồ sơ 24h × 4 chất); ghi rõ là hạn chế |
| `NMHC(GT)` | thiếu 90,2% | Loại |
| Cảm biến `PT08.*` | rẻ tiền, trôi số | Chỉ dùng trong EDA (tương quan) |

## Quyết định và căn cứ

| Quyết định | Căn cứ (số liệu) |
|---|---|
| Nội suy theo thời gian chỉ khoảng trống ≤ 6h | 86–94% khoảng trống của CO/NOx/NO2 ≤ 6h; lấp phần đầu của khoảng trống dài (`limit=`) tạo dữ liệu giả nên không dùng |
| Khoảng trống dài để NaN; chỉ lấp tạm bằng trung vị (tháng, thứ, giờ) cho MSTL | CO còn 16,0% giờ thiếu sau nội suy; các điểm lấp tạm không bao giờ được gắn cờ hay đưa vào khai phá |
| Trung vị trượt 3h để khử nhiễu | Cắt gai 1 điểm, giữ đỉnh ≥ 2h (hình `03_preprocess.png`) |
| MSTL chu kỳ 24h + 168h trên `log1p` | Chu kỳ tuần rõ (Chủ nhật CO = 62% ngày thường); dao động mùa đông lớn hơn → log đưa phần dư về sai lệch tương đối |
| Robust z > 3,5 trên phần dư (trung vị/MAD chỉ tính trên giờ có số đo thật), chỉ chiều dương | Ngưỡng Iglewicz–Hoaglin; cấy đỉnh giả: bắt lại 83% đỉnh +5σ, 100% đỉnh +8σ |
| Thêm điều kiện “trên trung vị của chất” và bỏ 7 ngày đầu/cuối | Không có điều kiện này: đỉnh giả lúc 0–6h khi nồng độ rất thấp, và đỉnh “mạnh nhất” rơi vào ngày cuối chuỗi (biên MSTL) |
| Baseline đối chứng: z toàn cục > 3σ trên chuỗi đã làm sạch | Dồn 88% cờ vào tháng 10–2; chỉ ~17% giờ trùng với phương pháp phần dư |
| Isolation Forest, `contamination = 0,02`, cùng tập giờ với bộ đơn biến | Bổ sung bất thường đa biến; 56% cờ là giờ thấp bất thường mà bộ đơn biến bỏ qua |
| PrefixSpan, 5 khung giờ, mức H/M/L theo tam phân vị, bỏ M | Giữ thông tin thời gian trong phần tử; min support 15% → 21 mẫu closed trên 279 ngày |
| Phân cụm ngày: 96 chiều z-score → PCA 90% (8 chiều) | Giữ cả mức và hình dạng; Hopkins 0,724 (ngẫu nhiên 0,499) |
| k = 4 | Silhouette K-Means cao nhất với k ≥ 3 (0,283); k = 2 (0,306) chỉ tách sạch/bẩn |
| K-Means++ làm kết quả chính | Silhouette 0,283 > Hybrid 0,274 > Ward 0,240; ổn định ARI trung bình 0,932 |
| SEED = 42 | Tái lập |

## Phương pháp

1. Đọc CSV (`;`, thập phân `,`), `-200` → NaN, lưới giờ liên tục.
2. Nội suy khoảng trống ≤ 6h, trung vị trượt 3h.
3. MSTL (24h, 168h, robust) trên `log1p` → phần dư → robust z → đỉnh bất thường + sự kiện; so với baseline 3σ; Isolation Forest đa biến.
4. Heatmap giờ × thứ, giờ × tháng; PrefixSpan trên chuỗi khung giờ H/L; support theo mùa và ngày thường/cuối tuần.
5. Hồ sơ ngày 24h × 4 chất → PCA → Hopkins → chọn k → K-Means / Ward / Hybrid → ổn định 30 lần lấy mẫu con → mô tả cụm.

## Kết quả (k = 4, K-Means)

- 4 cụm ngày: Ngày nghỉ sạch (34%), Ngày làm việc ấm (34%), Ngày lạnh NOx–NO2 cao (22%), Ngày ô nhiễm nặng mùa đông (10%, 3,31 giờ đỉnh bất thường/ngày).
- 398 giờ đỉnh bất thường (158 sự kiện) trên 4 chất; CO nhiều nhất (180 giờ, 66 sự kiện).
- Mẫu mạnh nhất: `10–15h:H → 16–20h:H` confidence 92,9%; `06–09h:H → 16–20h:H` gặp ở 43% ngày thường, 2% ngày cuối tuần.

## Đầu ra

- `notebooks/air_quality_mining.ipynb` sinh bởi `tools/build_notebook.py`, chạy được trên máy và Colab (~4 phút CPU).
- `reports/REPORT.md`.
- `reports/figures/`: 16 biểu đồ, 6 bảng CSV, `summary.json`.

## Ngoài phạm vi

Dashboard/app; dự báo nồng độ; phân cụm trạm thật (cần dataset nhiều trạm); hiệu chỉnh cảm biến PT08; DBSCAN (không thử).
