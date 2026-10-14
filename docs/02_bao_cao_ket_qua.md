# Báo cáo kết quả: Mô hình ra quyết định & phân loại ML/LLM cho Trust & Safety trong CS e-commerce

> Tổng hợp từ notebook 01–04 và phụ lục A1–A2. Mọi số liệu đều tái lập được bằng `scripts/` và `notebooks/`.
> Phần LLM (notebook 03) và Jev (notebook 05) đã sẵn sàng nhưng **cần chạy trên Colab** (GPU / API key): container nghiên cứu không truy cập được Hugging Face và API của TypeSafe, nên chưa có số liệu LLM/Jev trong báo cáo này.

## 1. Tóm tắt điều hành

| Câu hỏi | Trả lời ngắn | Nguồn |
|---|---|---|
| Mô hình nào phát hiện tin nhắn độc hại/spam tốt nhất? | **TF-IDF (từ + ký tự) + LogReg/LinearSVM**: PR-AUC ngang Stacking (0.734–0.737 so với 0.740), nhanh, nhẹ, giải thích được. Ensemble cây đứng sau | nb 01 |
| Cây quyết định có dùng được không? | Cây đơn kém nhất ở mọi bài toán, nhưng **cắt tỉa cost-complexity** cải thiện rõ (PR-AUC +0.11 trên văn bản, AUC +0.039 trên bảng; số lá giảm khoảng 5 lần) và cho **bộ luật đọc được** | nb 01, 02, A1 |
| Người dùng né bộ lọc thì sao? | Mất 2–6% recall; **chuẩn hoá ngược** lấy lại 64–73%; đặc trưng ký tự bền hơn đặc trưng từ | nb 01 |
| Mô hình công khai dùng cho chat CS được không? | **Rủi ro lớn**: đổi nguồn dữ liệu làm PR-AUC giảm 20–35%, nên **phải gán nhãn một mẫu chat nội bộ** | nb 01 |
| Gian lận thẻ / chargeback / bùng hàng? | Ensemble cây dẫn đầu (RF-balanced PR-AUC 0.821). Nhưng **chọn theo chi phí** thì CatBoost giảm nhiều nhất (−67.5%). Boosting mặc định **không ổn định** khi lớp dương cực hiếm, phải regularize | nb 02 |
| Chọn mô hình khi nhiều tiêu chí xung đột? | MCDM (AHP + TOPSIS/VIKOR/PROMETHEE/EDAS + SMAA): kết quả **phụ thuộc mạnh vào chuẩn hoá**; phải kiểm tra độ nhạy. Với kiểm duyệt CS, LogReg đứng đầu ổn định (P(top1) ≈ 1.0) | nb 04, A2 |
| Xử lý từng ca thế nào? | **Bayes chi phí kỳ vọng + phân bổ review theo lợi ích** thắng luật cố định 3.9–5.2% khi năng lực review ≥ 5%. Nhưng lợi thế **mất hết nếu chi phí bị ước lượng sai**, nên cần định lượng chi phí trước | nb 04 |
| Tự động hoá được bao nhiêu? | 84% tin nhắn với ≤ 1% lỗi; 94% với ≤ 2% lỗi (selective classification) | nb 04 |

## 2. Kết quả chi tiết

### 2.1 Kiểm duyệt văn bản (6 task, 11 mô hình)
| Mô hình | PR-AUC TB | recall@1%FPR TB | Ghi chú |
|---|---|---|---|
| Stacking (LR + NB + LightGBM) | 0.740 | 0.410 | huấn luyện chậm khoảng 40 lần |
| LinearSVM | 0.737 | 0.387 | |
| LogReg | 0.734 | 0.389 | Pareto: chất lượng / tốc độ / giải thích |
| LightGBM | 0.721 | 0.403 | |
| XGBoost | 0.708 | 0.368 | |
| RF / ExtraTrees | 0.700 | 0.36 | ExtraTrees tốt nhất trên OffensEval (0.749) |
| DT-pruned | 0.621 | 0.270 | DT-CART chỉ 0.510 / 0.145 |

* Hate speech là phần khó nhất: Davidson khoảng 0.43, HatEval khoảng 0.55. Spam gần như đã giải quyết xong (0.985).
* Mô hình tuyến tính thắng trên văn bản thưa nhiều chiều, **ngược với dữ liệu bảng** (A1: ensemble cây thắng). Kết luận "cây tốt nhất" **phụ thuộc loại dữ liệu**.

### 2.2 Gian lận / chargeback / bùng hàng
* ULB (thật, chia theo thời gian): RF-balanced PR-AUC 0.821, hơn các mô hình khác có ý nghĩa thống kê. Theo % giảm chi phí: CatBoost 67.5%, RF-balanced 64.6%, HistGB-reg 59.9%.
* LightGBM mặc định 0.17–0.44 và không tái lập được, so với 0.80 sau regularize (`min_child_samples=100`, `reg_lambda=5`).
* DT-pruned xếp hạng tốt (0.72) nhưng chỉ giảm 2.4% chi phí; IsolationForest làm **tăng** 32% chi phí. Đây là bằng chứng rằng *xếp hạng tốt chưa chắc quyết định tốt*.
* Class weighting làm Brier tăng khoảng 70 lần, nên phải hiệu chỉnh lại trước khi dùng xác suất cho quyết định.
* Mô phỏng bùng hàng/chargeback: giảm khoảng 38% và 82–84% chi phí. Các mô hình tốt không khác nhau có ý nghĩa (do cơ chế mô phỏng gần tuyến tính).

### 2.3 MCDM
* **Phụ lục A2**, chọn mô hình trên 14 dataset UCI: các phương pháp đồng thuận trung bình (Kendall τ 0.48–0.53). **Chuẩn hoá vector so với min-max cho τ = −0.07**: chuẩn hoá vector nén AUC (hệ số biến thiên 4.6%) trong khi thời gian (hệ số biến thiên 173%) lấn át, dẫn tới xếp DT-CART hạng 2 một cách vô lý.
* **Notebook 04**: chiều ngược lại, min-max *phóng đại* tiêu chí gần như hằng số (Robustness, hệ số biến thiên 1%). → Quy trình đề xuất: (1) kiểm tra hệ số biến thiên của từng tiêu chí; (2) chạy ≥ 3 phương pháp và ≥ 2 kiểu chuẩn hoá; (3) báo cáo **SMAA** (xác suất đạt hạng 1) thay vì một xếp hạng duy nhất.
* **SMAA ngẫu nhiên trên fold CV** (đề xuất cải tiến): lan truyền đồng thời bất định trọng số và bất định đo lường.

### 2.4 Chính sách triage (chi phí trên 1000 ca)
| Năng lực review | Luật 0.5/0.9 | Một ngưỡng tối ưu | Scorecard MCDM | Bayes thô | Bayes hiệu chỉnh |
|---|---|---|---|---|---|
| 0% | 1262.5 | **1195.9** | 1285.5 | 1256.0 | 1230.3 |
| 2% | 1219.3 | **1145.8** | 1234.9 | 1162.8 | 1165.1 |
| 5% | 1140.6 | 1099.5 | 1168.6 | **1097.9** | 1102.0 |
| 20% | 1065.6 | 1055.5 | 1098.4 | **1009.7** | 1021.4 |

* Chi phí bị ước lượng sai (12 kịch bản chi phí thật): regret trung bình là Bayes-biết-chi-phí-thật 1170.8 < luật 1240.9 ≈ Bayes-giả-định 1246.6 < ngưỡng 1260.8 < scorecard MCDM 1370.6.

## 3. Đóng góp / hướng cải tiến đã hiện thực

| # | Cải tiến | Kết quả |
|---|---|---|
| C1 | Cắt tỉa cost-complexity có CV + quy tắc gần 1-SE | +0.039 AUC (bảng), +0.11 PR-AUC (văn bản), số lá giảm khoảng 5 lần |
| C2 | Chuẩn hoá ngược chống né bộ lọc | lấy lại 64–73% recall bị mất |
| C3 | Phân tích độ nhạy theo chuẩn hoá cho MCDM | phát hiện đảo xếp hạng hoàn toàn (τ = −0.07) |
| C4 | SMAA ngẫu nhiên trên fold CV | báo cáo được mức chắc chắn của khuyến nghị |
| C5 | Triage Bayes + phân bổ năng lực review theo lợi ích kỳ vọng | −3.9% đến −5.2% chi phí so với luật cố định (năng lực ≥ 5%) |
| C6 | Đánh giá theo chi phí phụ thuộc giá trị đơn | đổi mô hình được chọn (CatBoost thay vì RF-balanced) |
| C7 | Regularize boosting khi lớp cực hiếm | PR-AUC từ 0.17–0.44 lên 0.80 |
| C8 | Ensemble LLM + TF-IDF; LLM làm chuyên gia AHP (đo CR, ẩn danh tên) | **chờ chạy Colab** (nb 03) |
| C9 | Bộ mô phỏng bùng hàng + schema dữ liệu; pipeline hoá đơn giả | nền tảng để cắm dữ liệu nội bộ |
| C10 | **Jev (decision model zero-shot)**: client có cache, adapter Jev-like mở, 7 thí nghiệm (kiểm duyệt, hiệu chỉnh, obfuscation, ensemble, cold-start bùng hàng, triage Bayes, AHP) | **chờ chạy với API key** (nb 05, docs/04) |

## 4. Hạn chế
* Chi phí trong `CostModel`/`COSTS` là giả định; bùng hàng/chargeback dùng dữ liệu mô phỏng.
* Dữ liệu kiểm duyệt là Wikipedia/Twitter/SMS, **không phải chat CS**; siêu tham số mặc định; thời gian đo trên CPU dùng chung.
* Chưa có kết quả LLM/guard model/transformer (cần GPU).

## 5. Lộ trình tiếp theo
1. **Chạy notebook 05 (Jev)** với `TYPESAFE_API_KEY` (khoảng 8 nghìn lần gọi, dưới 0.2 USD theo giá công bố) và commit `results/jev/cache.jsonl` để tái lập miễn phí.
2. **Chạy notebook 03 trên Colab** (Qwen2.5/Qwen3, Phi-4-mini, Gemma-3, Llama-3.2, Llama Guard 3, Qwen3Guard, ShieldGemma) rồi chạy lại notebook 04 để đưa LLM vào MCDM.
3. **Dữ liệu nội bộ:** gán nhãn khoảng 2–5k tin nhắn chat CS (toxic/abusive/threat/spam-scam) và lịch sử đơn bùng/chargeback; chạy lại toàn bộ bằng `load_internal_csv` và `load_internal_orders`.
4. **Định lượng chi phí** với CS lead, finance và legal (AHP/BWM để ra tỉ lệ chi phí), rồi cập nhật `CostModel`.
5. Fine-tune transformer nhỏ (DistilRoBERTa) hoặc LoRA cho LLM trên dữ liệu nội bộ; active learning với LLM làm bộ gán nhãn sơ bộ.
6. Uplift modeling cho can thiệp bùng hàng (gọi xác minh hay yêu cầu cọc); chính sách robust theo bất định chi phí (minimax regret).
7. Hiện thực pipeline hoá đơn giả (docs/03) và cắm `p_doc_forged` vào triage.
