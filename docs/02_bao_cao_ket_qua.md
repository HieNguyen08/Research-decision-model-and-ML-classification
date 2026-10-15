# Báo cáo kết quả: Mô hình ra quyết định & phân loại ML/LLM cho Trust & Safety trong CS e-commerce

> Tổng hợp từ notebook 01–05 và phụ lục A1–A2. Mọi số liệu đều tái lập được bằng `scripts/` và `notebooks/`.
> Notebook 03 (LLM, guard model, classifier chuyên dụng) và 05 (decision model zero-shot WaterSheep) đã chạy trên **Colab GPU (A100)**. Số liệu được trích từ output đã lưu (`scripts/extract_notebook_results.py` → `results/llm`, `results/jev`).
> **Jev thật (TypeSafe AI) chưa được chạy** vì cần credit trả phí; WaterSheep là bản mở cùng giao thức.

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
| LLM nhỏ (1–4B) thay được TF-IDF không? | **Chưa**: tốt nhất là llama-3.2-3b 8-shot với PR-AUC 0.726, so với 0.756. Nhưng **ghép LLM zero-shot + TF-IDF** đạt **0.793 (+4.9%)**, cao nhất nghiên cứu. LLM thắng lớn khi dữ liệu lệch phân phối (TweetEval-hate 0.761 so với 0.511) | nb 03 |
| Guard model (Llama Guard, Qwen3Guard, ShieldGemma)? | **Không hợp nếu dùng nguyên trạng** (PR-AUC 0.45–0.60): danh mục vi phạm không có lăng mạ/spam; chậm 85–570 ms/tin | nb 03 |
| Có đạt PR-AUC ≥ 0.90 được không? | **Có ở Jigsaw, HateXplain, spam** (nhờ fine-tune transformer, nb 06). **Không thể ở Davidson-hate** (trần nhãn ≈ 0.45–0.56: chính người gán nhãn chỉ đồng ý F1 0.28) **và HatEval** (test lệch phân phối: 0.83 khi kiểm định chéo so với 0.54 trên test). Đo Davidson theo định nghĩa "lăng mạ" sát nhu cầu CS thì đạt 0.996. ANN huấn luyện từ đầu (CNN/BiGRU/fastText/MLP) kém TF-IDF | docs/05, nb 06 |
| Decision model zero-shot (họ Jev, bản mở WaterSheep)? | Với tin nhắn: đạt 91% PR-AUC của TF-IDF mà không cần nhãn, bền nhất trước né bộ lọc, nhưng **quá tự tin** nên cần khoảng 50 nhãn để hiệu chỉnh. Với đơn hàng: **không dùng được** (AUC < 0.5) | nb 05 |

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
* **Trọng số do mô hình đề xuất (nb 04 B7):** mọi nguồn (3 LLM, WaterSheep) đều giữ LogReg ở hạng 1.
  * LLM không bao giờ cho ma trận AHP nhất quán (CR < 0.1: 0/15 lần).
  * WaterSheep nhất quán (CR ≈ 0.01) nhưng trọng số gần đều và gần như không đổi giữa hai kịch bản: τ = 0.89 so với AHP live-chat, nhưng chỉ 0.42 so với offline-audit.
  * → **CR thấp không chứng minh phán đoán tốt.** Trọng số phải do người chịu trách nhiệm quyết định.

### 2.4 Chính sách triage (chi phí trên 1000 ca)
| Năng lực review | Luật 0.5/0.9 | Một ngưỡng tối ưu | Scorecard MCDM | Bayes thô | Bayes hiệu chỉnh |
|---|---|---|---|---|---|
| 0% | 1262.5 | **1195.9** | 1285.5 | 1256.0 | 1230.3 |
| 2% | 1219.3 | **1145.8** | 1234.9 | 1162.8 | 1165.1 |
| 5% | 1140.6 | 1099.5 | 1168.6 | **1097.9** | 1102.0 |
| 20% | 1065.6 | 1055.5 | 1098.4 | **1009.7** | 1021.4 |

* Chi phí bị ước lượng sai (12 kịch bản chi phí thật): regret trung bình là Bayes-biết-chi-phí-thật 1170.8 < luật 1240.9 ≈ Bayes-giả-định 1246.6 < ngưỡng 1260.8 < scorecard MCDM 1370.6.

### 2.5 LLM, guard model, classifier chuyên dụng (nb 03, Colab A100, 300 tin/task)
PR-AUC trung bình 6 task (cấu hình tốt nhất của mỗi mô hình, cùng tập con test):

| Mô hình | PR-AUC TB | ROC-AUC TB | recall@5%FPR | ms/tin (A100) |
|---|---|---|---|---|
| **qwen3-4b zero-shot ⊕ TF-IDF LogReg** (rank-average, w = 0.3) | **0.793** | – | – | khoảng 30 |
| TF-IDF Stacking / LogReg | 0.761 / 0.756 | 0.857 / 0.853 | 0.596 / 0.594 | < 1 (CPU) |
| llama-3.2-3b (8-shot) | 0.726 | **0.877** | 0.546 | khoảng 22 |
| qwen3-4b* / phi-4-mini / qwen2.5-3b (8-shot) | 0.712 / 0.703 / 0.697 | 0.861 / 0.864 / 0.849 | 0.531 / 0.478 / 0.483 | 20–30 |
| qwen2.5-1.5b / qwen3-1.7b / gemma-3-1b (8-shot) | 0.683 / 0.666 / 0.560 | 0.847 / 0.831 / 0.754 | | 15–20 |
| ShieldGemma-2B / Qwen3Guard-4B / -0.6B / Llama-Guard-3-1B | 0.598 / 0.551 / 0.515 / 0.452 | 0.80 / 0.82 / 0.77 / 0.69 | 0.26–0.37 | 85–570 |

\* qwen3-4b bị đánh giá thấp do lỗi bão hoà float32 (đã sửa trong `drm.llm`; cần chạy lại riêng mô hình này).

* **LLM xếp hạng tổng thể tốt hơn nhưng vùng điểm cao nhiễu hơn:** llama-3.2-3b có ROC-AUC cao hơn TF-IDF, nhưng PR-AUC và recall@5%FPR thấp hơn. Với kiểm duyệt (cần FPR thấp), PR-AUC và recall@FPR là thước đo đúng.
* **Theo task:**
  * LLM thắng rõ trên **TweetEval-hate** (0.761 so với 0.511; tập test lệch phân phối so với train).
  * LLM thua rõ trên **Davidson** (≤ 0.31 so với 0.55; ranh giới hate/offensive là quy ước của người gán nhãn) và **HateXplain** (0.84 so với 0.91).
  * Spam: qwen3-4b 0.980, ngang TF-IDF.
* **Few-shot** giúp +0.01 đến +0.13 (mô hình yếu được lợi nhiều nhất).
* **Contextual calibration** không đổi PR-AUC với bài toán nhị phân, vì là phép tịnh tiến log-odds.
* **Ghép LLM + TF-IDF** (w = 0.3): +0.024 đến +0.037 với các LLM 3–4B; +0.007 đến +0.013 với LLM 1.5–1.7B.
* **Guard model:** hiệu năng thấp chủ yếu do **danh mục vi phạm không khớp** với nhu cầu CS:
  * Llama Guard 3 không coi lăng mạ là vi phạm (Jigsaw 0.27);
  * Qwen3Guard không có nhóm spam (SMS 0.28–0.39);
  * ShieldGemma nhận chính sách viết bằng lời nên tốt nhất trong nhóm.
* **Classifier fine-tune sẵn:**
  * Nhiều điểm cao là do **rò rỉ dữ liệu**, được đánh dấu ở cột `leakage`: toxic-bert trên Jigsaw 0.982, bert-tiny trên SMS 0.970, roberta-hate trên TweetEval-hate 0.894 (nghi rò rỉ).
  * Khi bỏ các ô đó, chúng thắng TF-IDF trên TweetEval-offensive (0.76–0.78 so với 0.70), thua trên HateXplain và Jigsaw. Rất nhanh: 0.5–2 ms/tin.
* **LLM làm chuyên gia AHP:** JSON hợp lệ 15/15 lần, nhưng CR < 0.1 ở 0/15 lần. LLM dồn 38–60% trọng số vào Quality dù kịch bản nhấn mạnh độ trễ và ngân sách GPU.

### 2.6 Decision model zero-shot: WaterSheep (bản mở họ Jev) và Jev-like Qwen (nb 05)
| | WaterSheep | Jev-like Qwen2.5-1.5B | TF-IDF LogReg (đã huấn luyện) |
|---|---|---|---|
| PR-AUC TB tin nhắn (6 task) | 0.688 (0.698 với câu hỏi theo task) | 0.560 | 0.756 |
| ECE thô → sau Platt với 50 nhãn/task | 0.269 → **0.053** | 0.230 → 0.070 | 0.095 |
| PR-AUC giảm khi obfuscation 30% | **−0.9%** | −5.0% | −4.5% |
| Ghép với TF-IDF (w = 0.25) | 0.773 (+2.2%) | | |
| ROC-AUC zero-shot: bùng hàng / chargeback (mô phỏng) | **0.411 / 0.251** (ngược) | 0.710 / 0.732 | 0.787 / 0.895 (huấn luyện đủ dữ liệu) |
| Triage Bayes (chi phí / 1000 ca, review 5%) | 1746 | | 1090 (TF-IDF + GBM) |
| AHP: CR / trọng số | 0.01 / gần đều, không đổi theo kịch bản | | |
| Độ trễ p50 / chi phí | 39 ms / 0 USD (GPU Colab) | 47 ms | |

* **Tin nhắn:** WaterSheep chỉ khoảng 150M tham số nhưng đạt 91% PR-AUC của TF-IDF mà không cần nhãn.
  * Ngang hoặc hơn TF-IDF ở Jigsaw (0.913), offensive (0.780 so với 0.662) và spam (1.000).
  * Rất kém ở Davidson-hate (0.129).
  * Ngang các LLM 3–4B zero-shot dù nhỏ hơn khoảng 20 lần.
* **Quá tự tin:** dự đoán 0.8–0.97 nhưng tần suất thật chỉ 0.35–0.45. Vì vậy Bayes dùng xác suất thô của WaterSheep (1746) còn **kém luật ngưỡng** (1641); phải hiệu chỉnh trước.
* **Đơn hàng (JSON dạng bảng):** câu hỏi Yes/No của WaterSheep cho tín hiệu *ngược*, và prior của nó không giúp cold-start (≤ +0.004 AUC).
  Jev-like Qwen (LLM) đọc JSON tốt hơn nhiều: bùng hàng 0.71, ngang LogReg huấn luyện với khoảng 128 đơn có nhãn.
* **`action` / `severity`** chưa dùng để tự động hoá được: `block_and_report` cho 53% tin bình thường của TweetEval-hate; `action` trực tiếp có chi phí cao nhất (1891).
* **Chi phí Jev thật cho cùng khối lượng** (8402 lần gọi, 4.94M token) theo giá công bố: khoảng **0.21 USD**.
* Đối chiếu từng giả thuyết H0–H7: docs/04 mục 6.

### 2.7 Hướng tới ≥ 0.90: mạng nơ-ron và mức trần dữ liệu (docs/05)
* **ANN huấn luyện từ đầu trên CPU** (`results/nn`, toàn bộ test): PR-AUC trung bình là BiGRU 0.700, MLP 0.692, fastText-style 0.692, TextCNN 0.690, đều **thấp hơn TF-IDF + LogReg (0.734)**. Ghép với TF-IDF cũng không tăng trung bình. Đòn bẩy nằm ở **tiền huấn luyện** (fine-tune transformer, nb 06).
* **Mức trần** (`results/ceiling`):
  * Davidson-hate: PR-AUC trần khoảng 0.45–0.56, và TF-IDF đã đạt 0.43.
  * HateXplain: trần 0.94–0.98.
  * HatEval: lệch phân phối mạnh; 8% tin test trùng train nhưng chỉ 61% cùng nhãn.
  * Tập con 300 tin quá nhỏ (Davidson có khoảng tin cậy [0.33, 0.77]), nên **từ nb 06 trở đi chỉ đánh giá trên toàn bộ test**.
  * Trên toàn bộ test, mức xuất phát thực của TF-IDF là 0.734 chứ không phải 0.79.
* **Mục tiêu đề xuất:** đạt **≥ 90% mức trần** của từng benchmark, báo cáo kèm tỉ lệ dương, ROC-AUC và khoảng tin cậy. Đặt "PR-AUC trung bình ≥ 0.90" thì kết quả phụ thuộc vào việc chọn benchmark nào.

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
| C8 | Ensemble LLM zero-shot + TF-IDF (rank-average); LLM làm chuyên gia AHP (đo CR) | PR-AUC 0.756 → **0.793 (+4.9%)**, cao nhất nghiên cứu; AHP của LLM không nhất quán (0/15) |
| C9 | Bộ mô phỏng bùng hàng + schema dữ liệu; pipeline hoá đơn giả | nền tảng để cắm dữ liệu nội bộ |
| C10 | **Decision model zero-shot họ Jev**: client có cache, adapter WaterSheep (mở) và Jev-like, 7 thí nghiệm | WaterSheep: 91% PR-AUC của TF-IDF không cần nhãn; bền nhất trước obfuscation (−0.9%); 50 nhãn đưa ECE từ 0.27 xuống 0.05; ghép với TF-IDF +2.2%; **thất bại với đơn hàng**. Jev thật chờ credit (khoảng 0.21 USD) |
| C11 | Phát hiện & sửa lỗi bão hoà float32 khi chấm LLM (điểm hoà nhau ở p = 1) | xếp hạng theo log-odds; có test hồi quy; ảnh hưởng tới số của qwen3-4b |
| C12 | Phân tích mức trần dữ liệu (đồng thuận người gán nhãn, gán nhãn lại, lệch phân phối, tin trùng, khoảng tin cậy của tập test) | chỉ ra mục tiêu ≥ 0.90 khả thi ở đâu; Davidson-hate trần khoảng 0.45–0.56 |
| C13 | ANN (CPU) + pipeline fine-tune transformer / LoRA / embedding + ghép tham lam (nb 06) | ANN không tiền huấn luyện kém TF-IDF (0.69–0.70 so với 0.734); nb 06 **chờ chạy Colab** |

## 4. Hạn chế
* Chi phí trong `CostModel`/`COSTS` là giả định; bùng hàng/chargeback dùng dữ liệu mô phỏng.
* Dữ liệu kiểm duyệt là Wikipedia/Twitter/SMS, **không phải chat CS**; siêu tham số mặc định; thời gian đo trên CPU dùng chung.
* LLM/WaterSheep chỉ chấm 300 tin/task. Chênh lệch < 0.03 PR-AUC cần paired bootstrap trước khi kết luận. Số được trích từ output notebook đã làm tròn 3 chữ số; chưa có file điểm `.npz` của lần chạy Colab.
* qwen3-4b bị đánh giá thấp do lỗi bão hoà (đã sửa, cần chạy lại). Chưa đo độ bền trước obfuscation của LLM ở nb 03. Chưa chạy Jev thật.

## 5. Lộ trình tiếp theo
1. ~~Chạy notebook 03, 05 trên Colab~~ (xong). Việc còn lại:
   * chạy lại riêng qwen3-4b sau bản sửa bão hoà;
   * chạy lại E5 với prior của Jev-like Qwen (LLM đọc JSON đơn hàng tốt hơn WaterSheep);
   * nếu có credit, thêm `TYPESAFE_API_KEY` để so Jev thật (khoảng 0.21 USD);
   * nhớ đặt `PUSH_RESULTS = True` hoặc tải thư mục `results/` để giữ file điểm.
2. **Chạy notebook 06 trên Colab A100** (fine-tune transformer + ghép mô hình; nhóm `CORE` khoảng 2–3 giờ). Đặt `PUSH_RESULTS = True`, hoặc để mặc định để tải file zip kết quả về.
3. **Kiến trúc đề xuất cho chat CS** (từ nb 03–05):
   * TF-IDF huấn luyện trên nhãn nội bộ, ghép (rank-average) với một LLM 3–4B hoặc WaterSheep đã hiệu chỉnh bằng khoảng 50 nhãn;
   * đưa xác suất vào chính sách Bayes chi phí (nb 04);
   * LLM sinh giải thích cho nhân viên;
   * không dùng guard model nguyên trạng, và không để mô hình zero-shot tự chọn hành động.
4. **Dữ liệu nội bộ:** gán nhãn khoảng 2–5k tin nhắn chat CS (toxic/abusive/threat/spam-scam) và lịch sử đơn bùng/chargeback; chạy lại toàn bộ bằng `load_internal_csv` và `load_internal_orders`.
5. **Định lượng chi phí** với CS lead, finance và legal (AHP/BWM để ra tỉ lệ chi phí), rồi cập nhật `CostModel`.
6. Fine-tune transformer nhỏ (DistilRoBERTa) hoặc LoRA cho LLM trên dữ liệu nội bộ; active learning với LLM làm bộ gán nhãn sơ bộ.
7. Uplift modeling cho can thiệp bùng hàng (gọi xác minh hay yêu cầu cọc); chính sách robust theo bất định chi phí (minimax regret).
8. Hiện thực pipeline hoá đơn giả (docs/03) và cắm `p_doc_forged` vào triage.
