# Báo cáo: Air Quality Outlier & Trend Mining

## Tóm tắt

Dữ liệu là **9.357 giờ** đo chất lượng không khí từ một thiết bị đặt bên đường ở một thành phố Ý (10/03/2004 – 04/04/2005). Sau khi nội suy các khoảng trống ngắn và khử nhiễu, chuỗi được phân rã bằng MSTL. Phần dư được dùng để tìm đỉnh ô nhiễm: tìm được **466 giờ bất thường (186 sự kiện)**, và bộ phát hiện bắt lại **100%** các đỉnh cấy giả +8σ. Khai phá tuần tự (PrefixSpan) cho thấy luật **“trưa cao → chiều cao” đúng 92,9%** và cao điểm sáng gần như biến mất vào cuối tuần (43% → 2%). Phân cụm 279 ngày đủ số liệu ra **4 kiểu ngày ô nhiễm**. Cụm ô nhiễm nặng nhất chỉ chiếm 10% số ngày nhưng có **3,79 giờ đỉnh bất thường/ngày**, gấp 13 lần ngày nghỉ sạch. Các cụm ổn định (ARI trung bình **0,93**), nhưng cấu trúc cụm chỉ ở mức vừa phải (Silhouette **0,28**).

## 1. Đề bài và dữ liệu

Đề 17 yêu cầu: tiền xử lý chuỗi thời gian, phát hiện đỉnh ô nhiễm, khai phá mẫu tuần tự theo mùa/giờ, và phân cụm các đơn vị có hồ sơ ô nhiễm giống nhau.

| Đề bài | Dữ liệu dùng | Ghi chú |
|---|---|---|
| Chuỗi ô nhiễm | CO (mg/m³), NOx (ppb), NO2 (µg/m³), C6H6 (µg/m³) | Máy phân tích chuẩn, theo giờ |
| Giá trị thiếu | `-200` | 18,0% (CO), 17,5% (NOx, NO2), 3,9% (C6H6) |
| Bối cảnh | T, RH, AH | Chỉ dùng mô tả cụm |
| Trạm quan trắc | **Không có** | Chỉ 1 thiết bị → phân cụm **ngày** thay cho trạm (mục 7) |

`NMHC(GT)` thiếu 90,2% nên bị loại (hình `01_missing.png`).

## 2. Khám phá dữ liệu (EDA)

Các chất có nguồn chung từ giao thông nên tương quan mạnh với nhau (hình `02_corr.png`).

- CO có hai đỉnh mỗi ngày, cao nhất lúc **19h**, đỉnh phụ lúc 8–9h (hình `04_stl_co.png`).
- Chủ nhật, CO trung bình chỉ bằng **62%** ngày thường (hình `09_heatmaps.png`).
- Theo mùa (chỉ số, trung bình cả kỳ = 100; hình `05_monthly_trend.png`):

| Chất | Hè (6–8) | Đông (11–2) |
|---|---|---|
| NOx | 44 | **155** |
| NO2 | 81 | 123 |
| CO | 77 | 112 |
| C6H6 | 93 | 96 |

NOx tăng khoảng 3,5 lần vào mùa đông, còn C6H6 gần như không đổi. Mức tăng của NOx lớn hơn nhiều so với các chất cùng nguồn giao thông. Điều này gợi ý có thêm nguồn thải mùa đông (sưởi) hoặc lớp nghịch nhiệt, nhưng cũng có thể do máy đo NOx trôi số; dữ liệu không đủ để phân biệt hai khả năng này.

## 3. Tiền xử lý

| Chất | % thiếu gốc | % nội suy ngắn (≤ 6h) | % còn thiếu (khoảng dài) |
|---|---|---|---|
| CO | 18,0 | 2,0 | 16,0 |
| NOx | 17,5 | 3,5 | 14,0 |
| NO2 | 17,5 | 3,5 | 14,1 |
| C6H6 | 3,9 | 0,2 | 3,8 |

- **Chỉ nội suy khoảng trống ≤ 6h**, và lấp **cả khoảng**. Tham số `limit=` của pandas sẽ lấp 6 giờ đầu của một khoảng trống dài, tạo ra dữ liệu giả, nên không dùng.
- **Khử nhiễu** bằng trung vị trượt 3h: cắt gai 1 điểm, giữ đỉnh kéo dài ≥ 2h (hình `03_preprocess.png`).
- **Khoảng trống dài** để NaN. MSTL cần chuỗi liên tục nên chỉ lấp tạm bằng trung vị (tháng, thứ, giờ); các điểm lấp tạm không bao giờ được gắn cờ, khai phá mẫu hay phân cụm.

## 4. Phát hiện đỉnh ô nhiễm

**Phương pháp.** Áp dụng MSTL (chu kỳ 24h và 168h, robust) trên `log1p`. Một giờ được tính là đỉnh khi thỏa cả ba điều kiện:
1. Robust z của phần dư > 3,5.
2. Nồng độ cao hơn trung vị của chất đó.
3. Không nằm trong 7 ngày đầu hoặc cuối chuỗi.

Cần thang log vì mùa đông dao động mạnh hơn: với ngưỡng tuyệt đối, mọi đỉnh sẽ dồn vào mùa đông. Hai điều kiện sau được thêm sau lần chạy đầu. Lần đó cho ra đỉnh giả lúc 0–6h (nồng độ rất thấp, chỉ tăng tương đối) và đỉnh “mạnh nhất” rơi vào ngày cuối chuỗi, đúng vùng biên MSTL ước lượng kém.

| Chất | Giờ bất thường | % giờ có số đo | Sự kiện | Baseline 3σ (giờ) | Trùng baseline |
|---|---|---|---|---|---|
| CO | 211 | 2,68% | 76 | 101 | 13% |
| NOx | 121 | 1,50% | 45 | 142 | 23% |
| C6H6 | 85 | 0,94% | 34 | 101 | 12% |
| NO2 | 49 | 0,61% | 31 | 44 | 12% |

- **Tự kiểm tra:** cấy đỉnh giả vào CO. Bộ phát hiện bắt lại **83%** đỉnh +5σ và **100%** đỉnh +8σ. Đỉnh +5σ bị sót khi rơi đúng vào chỗ phần dư vốn âm.
- **So với baseline** (z toàn cục > 3σ): baseline chỉ tìm “giờ ô nhiễm cao”, dồn vào cao điểm 17–20h và **88%** cờ rơi vào tháng 10–2. Phương pháp phần dư tìm giờ “cao hơn kỳ vọng của chính giờ/ngày/mùa đó”, nên bắt được cả đỉnh đêm khuya và giữa trưa. Hai phương pháp chỉ trùng khoảng 15% số giờ (hình `07_spike_timing.png`).
- **Sự kiện mạnh nhất:** CO ngày 31/01/2005 từ 14h đến 21h, đạt 6,8 mg/m³, z = 8,6 (hình `06_spikes_example.png`). Danh sách đầy đủ ở `anomalies.csv`.

**Đa biến.** Isolation Forest chạy trên vector phần dư của 4 chất và gắn cờ 149 giờ. Trong đó **35%** trùng đỉnh đơn biến, còn **58%** là giờ **thấp** bất thường mà bộ đơn biến (chỉ xét chiều dương) bỏ qua; Jaccard giữa hai bộ là 0,14 (hình `08_iforest.png`). Isolation Forest vì vậy bổ sung cho bộ đơn biến chứ không thay thế được nó: nếu chỉ cần tìm *đỉnh* ô nhiễm thì bộ đơn biến phù hợp hơn.

## 5. Mẫu theo giờ và mùa

Heatmap giờ × thứ và giờ × tháng (hình `09_heatmaps.png`):
- CO tập trung vào hai cao điểm 7–9h và 18–20h các ngày T2–T6. Hai cao điểm này mờ đi vào T7 và gần như biến mất vào CN.
- NO2 kéo dài sang buổi tối và đậm nhất vào tháng 11–2, vì là chất thứ cấp hình thành chậm hơn CO.

## 6. Mẫu tuần tự (PrefixSpan)

Mỗi ngày đủ số đo được chia thành 5 khung giờ (00–05h, 06–09h, 10–15h, 16–20h, 21–23h). Mỗi khung được gán mức H/M/L theo tam phân vị của chỉ số ô nhiễm tổng hợp (trung bình z của 4 chất), sau đó bỏ M. Thuật toán chạy trên 279 ngày, min support 15%, và tìm được 21 mẫu closed (`patterns.csv`, hình `10_sequence_patterns.png`).

| Mẫu | Support | Confidence | Ngày thường | Cuối tuần | Hè | Đông |
|---|---|---|---|---|---|---|
| 00–05h:L → 16–20h:H | 47,3% | 58,7% | 63% | 12% | 42% | 42% |
| 10–15h:H → 16–20h:H | 37,3% | **92,9%** | 47% | 16% | 12% | 52% |
| 06–09h:H → 16–20h:H | 30,5% | 86,7% | **43%** | **2%** | 20% | 31% |
| 00–05h:L → 06–09h:L | 25,4% | 31,6% | 12% | **55%** | 39% | 18% |
| 16–20h:H → 21–23h:H | 25,1% | 40,7% | 31% | 13% | **5%** | **45%** |

- **Ô nhiễm có quán tính:** khi khung trưa đã cao thì chiều cao với xác suất 93%. Khi sáng cao thì chiều cao với xác suất 87%.
- **Cuối tuần:** chuỗi điển hình là “đêm thấp → sáng thấp”, xuất hiện ở 55% ngày cuối tuần nhưng chỉ 12% ngày thường.
- **Mùa đông:** ô nhiễm chiều kéo dài sang tối (16–20h:H → 21–23h:H) ở 45% ngày đông, so với 5% ngày hè. Điều này phù hợp với lớp nghịch nhiệt buổi tối mùa đông giữ chất ô nhiễm lại gần mặt đất.

## 7. Phân cụm hồ sơ ô nhiễm theo ngày

**Vì sao phân cụm ngày.** Dataset chỉ có một thiết bị nên không thể nhóm các trạm. Mỗi ngày được coi là một “trạm ảo” với hồ sơ 24h × 4 chất = 96 chiều. Giá trị dùng z-score theo từng chất, nên hồ sơ giữ cả **mức** lẫn **hình dạng** đường cong. Sau đó PCA giữ 90% phương sai, còn 8 chiều.

**Dữ liệu có cấu trúc cụm không?** Hopkins = **0,724**, trong khi dữ liệu ngẫu nhiên đều cho 0,499. Dữ liệu có xu hướng cụm ở mức vừa, không phải các cụm tách biệt rõ.

**Chọn k** (hình `11_choose_k.png`):

| k | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Silhouette (K-Means) ↑ | 0,306 | 0,260 | **0,283** | 0,268 | 0,249 | 0,235 | 0,199 |
| Davies–Bouldin (K-Means) ↓ | 1,265 | 1,342 | **1,325** | 1,321 | 1,395 | 1,464 | 1,443 |

Chọn **k = 4**. k = 2 có Silhouette cao hơn nhưng chỉ tách được ngày “sạch” và ngày “bẩn”, không đủ để mô tả hồ sơ. Với k ≥ 3, k = 4 tốt nhất theo Silhouette. Dendrogram Ward cũng cho 4 nhánh lớn (hình `12_dendrogram.png`).

**So sánh phương pháp** (`evaluation.csv`):

| Phương pháp (k = 4) | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | ARI so với K-Means |
|---|---|---|---|---|
| **K-Means++ (n_init = 20)** | **0,283** | 1,325 | **124,1** | 1,00 |
| Agglomerative (Ward) | 0,240 | 1,348 | 108,4 | 0,53 |
| Hybrid (tâm Ward → K-Means) | 0,274 | **1,261** | 121,4 | 0,73 |

K-Means++ được chọn làm kết quả chính vì tốt nhất trên 2/3 chỉ số. Hybrid có Davies–Bouldin thấp nhất, nhưng cách biệt nhỏ.

### Độ ổn định (hình `13_stability.png`)

Chạy K-Means 30 lần, mỗi lần trên 80% số ngày lấy ngẫu nhiên và với seed khác. ARI so với kết quả gốc trung bình **0,932**, thấp nhất 0,697 (`stability.csv`). Các cụm vì vậy không phụ thuộc vào cách lấy mẫu.

### Hồ sơ cụm (`cluster_profile.csv`, hình `14_pca_clusters.png`, `15_cluster_profiles.png`, `16_cluster_composition.png`)

| Cụm | Tên đề xuất | Số ngày | Cuối tuần | Mùa chính | CO | NOx | NO2 | C6H6 | T (°C) | Giờ đỉnh/ngày |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Ngày nghỉ sạch | 94 (34%) | **65%** | Xuân/Hè | 1,37 | 129 | 85 | 6,0 | 19,0 | 0,29 |
| 2 | Ngày làm việc ấm | 95 (34%) | 6% | Xuân/Hè/Thu | 2,32 | 200 | 105 | 12,4 | 22,2 | 0,29 |
| 3 | Ngày lạnh NOx–NO2 cao | 61 (22%) | 28% | Đông | 2,17 | 343 | 143 | 9,3 | 10,5 | 1,39 |
| 4 | Ngày ô nhiễm nặng mùa đông | 29 (10%) | 7% | Đông/Thu | **3,40** | **533** | **148** | **15,9** | 11,6 | **3,79** |

- **Cụm 1 và 2** tách nhau chủ yếu theo **lịch làm việc**: 65% so với 6% ngày cuối tuần. Hai cụm có nhiệt độ gần nhau, nên khác biệt này đến từ lưu lượng giao thông.
- **Cụm 3** tách theo **thời tiết**: lạnh, NOx/NO2 cao suốt ngày kể cả đêm, và tỉ lệ cuối tuần 28% gần với mức ngẫu nhiên 29%. Cụm này là ô nhiễm nền mùa đông, không gắn với lịch đi làm.
- **Cụm 4** là tổ hợp tệ nhất: ngày làm việc trong mùa lạnh, độ ẩm cao (RH 62%), cả hai cao điểm đều rất cao. Cụm này chỉ chiếm 10% số ngày nhưng chứa phần lớn các giờ đỉnh bất thường (3,79 giờ/ngày, gấp 13 lần cụm 1 và 2).

## 8. Kết luận

1. Ô nhiễm tại điểm đo bị chi phối bởi **giao thông**: hai cao điểm 8–9h và 19h, Chủ nhật CO chỉ bằng 62% ngày thường. Mùa đông khuếch đại ô nhiễm, mạnh nhất với NOx (chỉ số 155 so với 44 mùa hè).
2. Tìm đỉnh bất thường bằng **phần dư MSTL + robust z** tìm được những giờ mà ngưỡng tuyệt đối bỏ sót: chỉ khoảng 15% số giờ trùng với baseline 3σ. Bộ phát hiện bắt lại 100% đỉnh cấy giả +8σ.
3. Isolation Forest bổ sung bất thường **kết hợp** và giờ **thấp** bất thường (58% số cờ). Để tìm đỉnh ô nhiễm, bộ đơn biến phù hợp hơn.
4. Luật tuần tự **“trưa cao → chiều cao” (confidence 92,9%)** cho thấy ô nhiễm có quán tính trong ngày. Có thể dùng luật này như tín hiệu cảnh báo sớm cho buổi chiều.
5. Cuối tuần và mùa đông có chuỗi mẫu riêng: cuối tuần là “đêm thấp → sáng thấp” (55%); mùa đông là ô nhiễm chiều kéo dài sang tối (45% so với 5% mùa hè).
6. Bốn kiểu ngày tách theo hai trục, **lịch làm việc** và **thời tiết lạnh**. Cụm 4 (10% số ngày) là đối tượng ưu tiên cho cảnh báo ô nhiễm.

## 9. Hạn chế và hướng phát triển

**Hạn chế:**
- Chỉ có **một thiết bị**, nên không phân cụm được trạm thật. Phân cụm ngày là cách thay thế tương đương về phương pháp, nhưng khác về ý nghĩa.
- Dữ liệu chỉ có khoảng 13 tháng. Xu hướng dài hạn và sự khác biệt giữa các năm chưa thể tách khỏi mùa vụ.
- Không có nhãn sự kiện ô nhiễm thật. Bộ phát hiện bất thường chỉ được kiểm tra bằng đỉnh cấy giả và so với baseline.
- Mức tăng NOx mùa đông có thể một phần do máy đo trôi số; dữ liệu không đủ để phân biệt.
- Cấu trúc cụm vừa phải (Silhouette 0,28; Hopkins 0,72). Ranh giới giữa cụm 2 và cụm 3 không sắc nét.

**Hướng phát triển:**
- Dùng dataset nhiều trạm (ví dụ UCI Beijing Multi-Site Air Quality) để phân cụm trạm thật bằng cùng pipeline.
- Ghép dữ liệu gió và lớp nghịch nhiệt để giải thích cụm 3 và 4.
- Dự báo đỉnh ô nhiễm buổi chiều từ mẫu buổi sáng/trưa.

## 10. Tái lập kết quả

```bash
pip install -r requirements.txt
python tools/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/air_quality_mining.ipynb
```

Hoặc mở notebook trên Google Colab và chọn *Run all*. Notebook tự tải dữ liệu nếu `data/AirQualityUCI.csv` chưa có. Thời gian chạy khoảng 4 phút trên CPU. Đầu ra trong `reports/figures/` gồm 16 biểu đồ, 6 bảng CSV và `summary.json`.
