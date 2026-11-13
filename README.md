# Air Quality Outlier & Trend Mining

> **Trạng thái:** notebook chạy trọn từ đầu đến cuối ngày 13/11/2026 (~4 phút CPU). Mọi số liệu dưới đây lấy từ `reports/figures/summary.json` của lần chạy đó.

## Mục tiêu

Đề 17 — *Pattern Mining + Clustering + Anomaly Detection* trên chuỗi chất lượng không khí:

- Tiền xử lý chuỗi thời gian: nội suy giá trị thiếu, khử nhiễu.
- Phát hiện **đỉnh ô nhiễm bất thường** bằng phát hiện bất thường chuỗi thời gian.
- Khai phá **mẫu tuần tự** ô nhiễm theo giờ và theo mùa.
- **Phân cụm** để nhóm các đơn vị có hồ sơ ô nhiễm giống nhau.

## Dataset

[UCI Machine Learning Repository — Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) (De Vito et al., 2008).

| Thuộc tính | Giá trị |
|---|---|
| Số bản ghi | 9.357 giờ (10/03/2004 – 04/04/2005) |
| Thiết bị | 1 thiết bị cảm biến đa kênh, đặt bên đường tại một thành phố Ý |
| Biến dùng | CO, NOx, NO2, C6H6 (máy phân tích chuẩn) + T, RH, AH |
| Mã thiếu | `-200` |
| Định dạng | CSV, phân tách `;`, thập phân `,` |

Notebook tự tải file zip từ UCI về `data/` nếu chưa có `data/AirQualityUCI.csv`.

## Đặc trưng và quy tắc xử lý

| Chỉ số đề bài | Đặc trưng sử dụng | Công thức / ghi chú |
|---|---|---|
| Nồng độ ô nhiễm | `CO(GT)`, `NOx(GT)`, `NO2(GT)`, `C6H6(GT)` | Đơn vị gốc; z-score theo chất khi gộp |
| Chỉ số ô nhiễm tổng hợp | trung bình z của 4 chất | Dùng cho khai phá tuần tự |
| Hồ sơ ô nhiễm | vector 24 giờ × 4 chất = 96 chiều mỗi ngày | Dùng cho phân cụm |
| Thời tiết | `T`, `RH`, `AH` | Chỉ để mô tả cụm |

Quy tắc:
- **Bỏ `NMHC(GT)`:** thiếu 90,2%.
- **Không dùng cảm biến `PT08.*` để phân tích:** cảm biến rẻ tiền, trôi số; chỉ xem trong EDA.
- **Chỉ nội suy khoảng trống ≤ 6 giờ**, lấp cả khoảng; khoảng dài để NaN (CO còn 16,0% giờ thiếu).
- **Không gắn cờ bất thường trên giá trị lấp tạm** và trong 7 ngày đầu/cuối chuỗi (biên MSTL).
- **Phân cụm ngày thay cho trạm:** dataset chỉ có 1 thiết bị (xem *Hạn chế*).

Chi tiết từng quyết định và số liệu làm căn cứ: [SPEC.md](SPEC.md).

## Phương pháp

| Bước | Thuật toán | Vai trò |
|---|---|---|
| 1 | Nội suy theo thời gian (≤ 6h) + trung vị trượt 3h | Làm sạch chuỗi |
| 2 | MSTL (chu kỳ 24h + 168h, robust) trên `log1p` | Tách xu hướng, mùa vụ ngày/tuần, phần dư |
| 3 | Robust z-score (MAD) > 3,5 trên phần dư | Phát hiện đỉnh ô nhiễm |
| 3' | Isolation Forest trên phần dư 4 chất | Bất thường đa biến |
| 4 | PrefixSpan (closed, min support 15%) | Mẫu tuần tự theo khung giờ, so theo mùa / ngày thường–cuối tuần |
| 5 | PCA → K-Means++ / Agglomerative (Ward) / Hybrid | Phân cụm hồ sơ ô nhiễm theo ngày |
| Đối chứng | z-score toàn cục > 3σ; dữ liệu ngẫu nhiên đều (Hopkins) | So sánh |

## Kết quả

**Đỉnh ô nhiễm.** Tìm được 466 giờ bất thường (186 sự kiện); CO nhiều nhất với 211 giờ. Bộ phát hiện bắt lại 100% đỉnh cấy giả +8σ (83% với +5σ). Chỉ khoảng 15% số giờ trùng với baseline 3σ: baseline dồn 88% cờ vào tháng 10–2, còn phần dư MSTL bắt cả đỉnh giữa trưa và đêm khuya.

**Mẫu tuần tự (279 ngày, 21 mẫu closed):**

| Mẫu | Confidence | Ngày thường | Cuối tuần | Hè | Đông |
|---|---|---|---|---|---|
| 10–15h:H → 16–20h:H | 92,9% | 47% | 16% | 12% | 52% |
| 06–09h:H → 16–20h:H | 86,7% | 43% | 2% | 20% | 31% |
| 16–20h:H → 21–23h:H | 40,7% | 31% | 13% | 5% | 45% |

**Phân cụm (k = 4):**

| Phương pháp (k = 4) | Silhouette ↑ | Davies–Bouldin ↓ |
|---|---|---|
| **K-Means++ (n_init = 20)** | **0,283** | 1,325 |
| Agglomerative (Ward) | 0,240 | 1,348 |
| Hybrid (Ward → K-Means) | 0,274 | 1,261 |

Hopkins = 0,724 (ngẫu nhiên 0,499). Độ ổn định qua 30 lần lấy mẫu con 80%: ARI trung bình 0,932.

| Cụm | Tỉ trọng | Cuối tuần | Mùa chính | CO (mg/m³) | NOx (ppb) | Giờ đỉnh/ngày | Gợi ý |
|---|---|---|---|---|---|---|---|
| 1. Ngày nghỉ sạch | 34% | 65% | Xuân/Hè | 1,37 | 129 | 0,29 | Mốc nền tham chiếu |
| 2. Ngày làm việc ấm | 34% | 6% | Xuân–Thu | 2,32 | 200 | 0,29 | Ô nhiễm giao thông giờ cao điểm |
| 3. Ngày lạnh NOx–NO2 cao | 22% | 28% | Đông | 2,17 | 343 | 1,39 | Ô nhiễm nền do thời tiết |
| 4. Ngày ô nhiễm nặng mùa đông | 10% | 7% | Đông/Thu | 3,40 | 533 | 3,79 | Ưu tiên cảnh báo |

Phân tích đầy đủ: [reports/REPORT.md](reports/REPORT.md).

## Công nghệ

| Thư viện | Công dụng |
|---|---|
| pandas, numpy | Đọc dữ liệu, chuỗi thời gian |
| statsmodels | MSTL |
| scikit-learn | Isolation Forest, PCA, K-Means, Agglomerative, chỉ số đánh giá |
| scipy | Linkage, dendrogram |
| prefixspan | Khai phá mẫu tuần tự |
| matplotlib, seaborn | Biểu đồ |
| nbformat, jupyter | Sinh và chạy notebook |

## Cấu trúc dự án

```text
.
├── README.md
├── SPEC.md                         # Quyết định thiết kế và căn cứ
├── requirements.txt
├── data/                           # AirQualityUCI.csv (tự tải, không commit)
├── notebooks/
│   └── air_quality_mining.ipynb    # Notebook chính (đã chạy, có output)
├── tools/
│   └── build_notebook.py           # Sinh notebook từ mã nguồn
└── reports/
    ├── REPORT.md                   # Báo cáo đầy đủ
    └── figures/
        ├── 01_missing.png          # Tỉ lệ thiếu từng biến
        ├── 02_corr.png             # Ma trận tương quan
        ├── 03_preprocess.png       # Trước/sau nội suy và khử nhiễu
        ├── 04_stl_co.png           # Phân rã MSTL của CO
        ├── 05_monthly_trend.png    # Xu hướng theo tháng
        ├── 06_spikes_example.png   # Ví dụ đỉnh bất thường
        ├── 07_spike_timing.png     # Đỉnh theo giờ/tháng so với baseline
        ├── 08_iforest.png          # Isolation Forest
        ├── 09_heatmaps.png         # Giờ × thứ, giờ × tháng
        ├── 10_sequence_patterns.png# Mẫu tuần tự theo nhóm
        ├── 11_choose_k.png         # Chọn số cụm
        ├── 12_dendrogram.png       # Cây phân cấp Ward
        ├── 13_stability.png        # Độ ổn định cụm
        ├── 14_pca_clusters.png     # Cụm trên mặt phẳng PCA
        ├── 15_cluster_profiles.png # Hồ sơ 24h của từng cụm
        ├── 16_cluster_composition.png # Mùa và cuối tuần theo cụm
        ├── anomalies.csv           # Danh sách sự kiện bất thường
        ├── anomaly_summary.csv     # Thống kê bất thường theo chất
        ├── patterns.csv            # Top 15 mẫu tuần tự
        ├── evaluation.csv          # So sánh phương pháp phân cụm
        ├── stability.csv           # ARI 30 lần chạy
        ├── cluster_profile.csv     # Hồ sơ cụm
        └── summary.json            # Số liệu chính (nguồn cho README/REPORT)
```

## Cài đặt và chạy

### Cách 1: Google Colab (không cần cài đặt)

1. Tải `notebooks/air_quality_mining.ipynb` lên Colab.
2. Chạy ô đầu tiên sau khi thêm dòng `!pip install prefixspan` (Colab đã có các thư viện còn lại).
3. Chọn *Runtime → Run all*. Notebook tự tải dữ liệu từ UCI; biểu đồ được lưu vào `reports/figures/` trong phiên Colab.

### Cách 2: Máy cá nhân

Yêu cầu Python 3.11+.

**1.** Cài thư viện:

```bash
pip install -r requirements.txt
```

**2.** (Tùy chọn) Sinh lại notebook sau khi sửa `tools/build_notebook.py`:

```bash
python tools/build_notebook.py
```

**3.** Chạy toàn bộ notebook và lưu output:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/air_quality_mining.ipynb
```

**4.** Hoặc mở bằng Jupyter / VS Code, chọn kernel Python và *Run All*.

Lưu ý: sửa trực tiếp trong `.ipynb` rồi chạy lại bước 2 sẽ ghi đè thay đổi — hãy sửa trong `tools/build_notebook.py`.

## Hạn chế

- **Chỉ có một thiết bị đo**, nên “phân cụm trạm” được thay bằng phân cụm **ngày** (mỗi ngày là một hồ sơ 24h × 4 chất). Về phương pháp hai bài toán tương đương, nhưng kết quả mô tả kiểu ngày chứ không phải kiểu trạm.
- Chỉ ~13 tháng dữ liệu: chưa tách được xu hướng dài hạn khỏi mùa vụ.
- Không có nhãn sự kiện ô nhiễm thật; bộ phát hiện chỉ được kiểm tra bằng đỉnh cấy giả và so với baseline.
- Cấu trúc cụm ở mức vừa (Silhouette 0,28); ranh giới cụm 2–3 không sắc nét.

## Thành viên

| Họ và tên | MSSV | Công việc |
|---|---|---|
| _(điền)_ | _(điền)_ | _(điền)_ |

## Tài liệu tham khảo

- De Vito, S. et al. (2008). *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario.* Sensors and Actuators B, 129(2), 750–757.
- Bandara, K., Hyndman, R. J., Bergmeir, C. (2021). *MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns.* arXiv:2107.13462.
- Iglewicz, B., Hoaglin, D. (1993). *How to Detect and Handle Outliers.* ASQC Quality Press.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). *Isolation Forest.* ICDM.
- Pei, J. et al. (2001). *PrefixSpan: Mining Sequential Patterns Efficiently by Prefix-Projected Pattern Growth.* ICDE.
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.* J. Comput. Appl. Math., 20, 53–65.
- Davies, D. L., Bouldin, D. W. (1979). *A Cluster Separation Measure.* IEEE TPAMI, 1(2), 224–227.
- [statsmodels MSTL](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.MSTL.html), [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html).
