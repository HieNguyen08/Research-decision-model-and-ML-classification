# Tổng quan tài liệu: Mô hình ra quyết định & phân loại ML/LLM cho Trust & Safety trong CS e-commerce

> Phạm vi: đội Customer Service (CS) của sàn thương mại điện tử cần **phát hiện** (phân loại) và **ra quyết định xử lý**
> các tình huống: ngôn từ độc hại/công kích/thù ghét trong chat, spam & lừa đảo, gian lận thanh toán & **chargeback**
> (hoàn tiền gian lận), **bùng hàng COD**, và **hoá đơn/bằng chứng giả do AI tạo**. Dữ liệu: tiếng Anh.

Tài liệu chia hai tầng, tương ứng hai câu hỏi khác nhau:

| Tầng | Câu hỏi | Công cụ |
|---|---|---|
| **Dự báo** | Ca này có rủi ro không, với xác suất bao nhiêu? | Cây quyết định & ensemble, mô hình tuyến tính, transformer, LLM |
| **Quyết định** | Nên làm gì (cho qua, chuyển người xem, chặn), với nguồn lực có hạn và nhiều mục tiêu xung đột? | Lý thuyết quyết định Bayes, học nhạy chi phí, human-in-the-loop, MCDM |

---

## 1. Lý thuyết mô hình ra quyết định

### 1.1 Quyết định Bayes và học nhạy chi phí
Với xác suất dự báo $p = P(y=1 \mid x)$ và ma trận chi phí $c_{FN}, c_{FP}$, hành động tối ưu là gắn cờ khi
$p \cdot c_{FN} > (1-p)\, c_{FP}$, tức ngưỡng tối ưu $p^* = c_{FP}/(c_{FP}+c_{FN})$ **chứ không phải 0.5** (Elkan, 2001).
Trong gian lận thanh toán, chi phí phụ thuộc **từng giao dịch** (giá trị đơn). Đây là học nhạy chi phí phụ thuộc mẫu (*example-dependent cost-sensitive*), ví dụ cây quyết định nhạy chi phí của Bahnsen et al. (2015).
Điều kiện cần để áp dụng là **xác suất phải được hiệu chỉnh** (calibration): Platt scaling (Platt, 1999), isotonic regression (Zadrozny & Elkan, 2002).
Niculescu-Mizil & Caruana (2005) cho thấy boosting và SVM thường lệch xác suất, còn logistic regression gần như hiệu chỉnh sẵn.
Khi lấy mẫu lại để cân bằng lớp, cần hiệu chỉnh ngược xác suất (Dal Pozzolo et al., 2015a).

### 1.2 Quyết định có con người trong vòng lặp
* **Reject option** (Chow, 1970): từ chối tự động hoá những ca mà rủi ro kỳ vọng vượt chi phí chuyển cho người.
* **Selective classification** (Geifman & El-Yaniv, 2017): đường *risk–coverage*, tức tự động xử lý x% ca tự tin nhất với sai số y%.
* **Learning to defer** (Madras et al., 2018; Mozannar & Sontag, 2020): học đồng thời bộ phân loại và quy tắc chuyển cho chuyên gia.
* **Thực tế CS:** năng lực review **có hạn** (K ca/ngày). Vì vậy bài toán trở thành *phân bổ năng lực*: chọn K ca có **lợi ích kỳ vọng của việc review** lớn nhất. Đây là dạng knapsack, giải tham lam được.

### 1.3 Ra quyết định đa tiêu chí (MCDM)
Khi các mục tiêu **không quy đổi được về một đơn vị tiền** (trải nghiệm khách, sức khoẻ tinh thần nhân viên, rủi ro pháp lý, chi phí GPU...), dùng MCDM:

| Nhóm | Phương pháp | Đặc điểm chính |
|---|---|---|
| Trọng số chủ quan | **AHP** (Saaty, 1980), **BWM** (Rezaei, 2015) | So sánh cặp; AHP có chỉ số nhất quán CR < 0.1 |
| Trọng số khách quan | **Entropy**, **CRITIC** (Diakoulaki et al., 1995) | Dựa trên độ phân tán / tương quan dữ liệu |
| Giá trị cộng | **SAW/WSM** | Bù trừ hoàn toàn, đơn giản |
| Điểm tham chiếu | **TOPSIS** (Hwang & Yoon, 1981), **VIKOR** (Opricovic, 1998), **EDAS** (Keshavarz Ghorabaee et al., 2015), **MOORA** (Brauers & Zavadskas, 2006) | Khoảng cách tới nghiệm lý tưởng / thoả hiệp / trung bình |
| Quan hệ trội | **PROMETHEE II** (Brans & Vincke, 1985), **ELECTRE** (Roy, 1968) | Ít bù trừ, cần hàm ưu tiên/ngưỡng |
| Bất định | **SMAA** (Lahdelma et al., 1998) | Xác suất mỗi phương án đạt hạng r trên không gian trọng số |

Những vấn đề đã được biết:
* **Các phương pháp cho kết quả khác nhau** trên cùng dữ liệu (Zanakis et al., 1998; Opricovic & Tzeng, 2004 so sánh VIKOR và TOPSIS).
* **Rank reversal**: thêm hoặc bớt một phương án làm đảo thứ tự các phương án còn lại (Belton & Gear, 1983 với AHP; Wang & Triantaphyllou, 2008 với ELECTRE).
* **MCDM để chọn mô hình ML** đã được dùng, phổ biến là AHP kết hợp TOPSIS để xếp hạng classifier theo độ chính xác, thời gian và độ ổn định.
* **Kết quả của nghiên cứu này** (phụ lục A2): **cách chuẩn hoá** (vector hay min-max) làm đảo xếp hạng TOPSIS hoàn toàn (Kendall τ = −0.07) khi các tiêu chí có hệ số biến thiên chênh lệch lớn. Đây là trường hợp rất điển hình của bài toán chọn mô hình ML (AUC dao động ±5%, còn thời gian chạy chênh nhau 100 lần).

### 1.4 Cây quyết định: vừa là mô hình dự báo, vừa là "luật quyết định" diễn giải được
* **Họ cây kinh điển:** ID3 (Quinlan, 1986), C4.5 (Quinlan, 1993), CART với cắt tỉa cost-complexity (Breiman et al., 1984). Cây tối ưu toàn cục: OCT (Bertsimas & Dunn, 2017), GOSDT (Lin et al., 2020). Cây xiên: OC1 (Murthy et al., 1994). Cây mềm/khả vi: Frosst & Hinton (2017), NODE (Popov et al., 2020).
* **Ensemble:** Random Forest (Breiman, 2001), Extra-Trees (Geurts et al., 2006), Rotation Forest (Rodríguez et al., 2006). Gradient boosting (Friedman, 2001) với các cài đặt XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017), CatBoost (Prokhorenkova et al., 2018).
* **Bằng chứng benchmark trên dữ liệu bảng:**
  * Fernández-Delgado et al. (2014): Random Forest đứng đầu trên 121 dataset.
  * Grinsztajn et al. (2022): cây vẫn vượt deep learning trên dữ liệu cỡ vừa (~10k mẫu), do có inductive bias phù hợp (bền với đặc trưng nhiễu, học được hàm không trơn).
  * McElfresh et al. (2023, 176 dataset): khác biệt GBDT so với NN thường nhỏ hơn tác động của việc tuning; GBDT tốt hơn khi phân phối lệch hoặc có đuôi dài.
  * Tabular foundation model là xu hướng mới mạnh trên dữ liệu nhỏ: TabPFN (Hollmann et al., *Nature* 2025), TabPFN-2.5, TabICLv2.

## 2. Phát hiện ngôn từ độc hại, thù ghét & spam

* **Dữ liệu chuẩn:**
  * Davidson et al. (2017): phân biệt *hate speech* với *offensive*. Đây là bài toán khó vì từ ngữ chồng lấn.
  * Jigsaw Toxic Comment (Perspective API).
  * HateXplain (Mathew et al., 2021): có rationale và cộng đồng bị nhắm tới.
  * TweetEval (Barbieri et al., 2020), gồm OffensEval/OLID (Zampieri et al., 2019) và HatEval (Basile et al., 2019).
  * SMS Spam Collection (Almeida et al., 2011).
* **Phương pháp:** TF-IDF n-gram từ và ký tự với mô hình tuyến tính là baseline rất mạnh, nhanh và dễ giải thích. Transformer fine-tune (BERT/RoBERTa) cho chất lượng cao hơn nhưng tốn tài nguyên hơn.
* **Né bộ lọc (adversarial):** Hosseini et al. (2017) cho thấy chỉ cần viết sai chính tả hoặc chèn dấu câu (`idiot` thành `idiiot`) là hạ điểm toxicity của Perspective API từ 84% xuống 20%. Đặc trưng **ký tự** và bước **chuẩn hoá ngược** là các biện pháp rẻ để chống lại.
* **Thiên lệch:** mô hình hate speech gắn cờ quá mức phương ngữ của người Mỹ gốc Phi (Sap et al., 2019). Khi hành động "chặn" ảnh hưởng tới khách hàng, cần kiểm toán công bằng.
* **Lệch miền:** mô hình huấn luyện trên một nguồn tổng quát hoá kém sang nguồn khác (Arango et al., 2019; Yin & Zubiaga, 2021). Chat CS là một miền **khác hẳn** với Wikipedia hay Twitter, nên bắt buộc phải có dữ liệu nội bộ để đánh giá.

## 3. LLM cho kiểm duyệt và cho ra quyết định

* **LLM tổng quát làm moderator:** Kumar, AbuHashem & Durumeric (ICWSM 2024) cho thấy GPT-3.5/4, Gemini và Llama 2 **vượt các toxicity classifier phổ biến**. Tuy nhiên tăng kích thước chỉ cải thiện thêm rất ít, và khi kiểm duyệt theo luật riêng của từng cộng đồng thì độ chính xác trung vị chỉ khoảng 64%.
* **Guard model chuyên dụng:** Llama Guard (Inan et al., 2023), Llama Guard 3 (1B/8B, taxonomy 14 nhóm MLCommons), ShieldGemma (2B/9B/27B, 4 nhóm hại), Qwen3Guard (0.6B/4B/8B, phán quyết 3 mức Safe/Controversial/Unsafe), Granite Guardian (IBM, 2024).
  Taxonomy của các mô hình khác nhau và phần lớn số liệu là do chính tác giả công bố, nên **cần đánh giá trên chính chính sách của doanh nghiệp**.
* **Hệ thống công nghiệp:** OpenAI moderation (Markov et al., AAAI 2023) mô tả quy trình đầy đủ gồm taxonomy, gán nhãn, active learning và xử lý lệch phân phối.
* **Mô hình mở cỡ nhỏ chạy được trên GPU T4:** Qwen2.5 (0.5B–7B), Qwen3 (0.6B–8B dense, có chế độ suy nghĩ), Phi-4-mini (3.8B), Gemma 3 (1B–27B), Llama 3.2 (1B/3B).
* **Kỹ thuật:**
  * Chấm điểm bằng log-likelihood của nhãn: ổn định hơn sinh văn bản rồi parse, và cho xác suất để tính AUC.
  * *Contextual calibration* (Zhao et al., 2021): trừ đi thiên lệch của prompt rỗng.
  * Few-shot in-context.
* **LLM với dữ liệu bảng:** TabLLM (Hegselmann et al., AISTATS 2023) và TabuLa-8B (Gardner et al., NeurIPS 2024). Khảo sát của Fang et al. (TMLR 2024).
  **Cảnh báo:** LLM đã "học thuộc" nhiều dataset phổ biến (Bordt et al., COLM 2024), nên đánh giá trên dataset công khai có thể bị thổi phồng.
* **LLM trong MCDM:**
  * LLM đóng vai chuyên gia ảo để sinh ma trận so sánh cặp AHP (ví dụ tích hợp AHP với GPT-4, arXiv 2402.07404).
  * Khung LLM-based MCDM tổng quát (arXiv 2502.15778).
  * Câu hỏi mở: **tính nhất quán** (CR) và **thiên lệch theo tên phương án** của LLM. Notebook 03 kiểm tra cả hai.

### 3.1 Decision model chuyên dụng: Jev và họ "System One"
* **Jev** (TypeSafe AI, 09/2026) là mô hình **phân biệt**. Nó nhận *state* (văn bản/JSON) cùng nhiều câu hỏi có kiểu: Noul (có/không), Choice (chọn nhãn), Score (thang bậc). Kết quả là **xác suất đã hiệu chỉnh** cho mọi câu hỏi trong một lần gọi; mô hình được huấn luyện bằng **RLCD** (Reinforcement Learning for Calibrated Decisions).
* **Bằng chứng:**
  * Guo et al. (2026): AUROC trung vị 0.886 **zero-shot** trên 44 benchmark lỗi alignment; thắng baseline có giám sát ở 25/31 benchmark; rẻ hơn LLM-judge khoảng 63 lần.
  * Cisco (2026): mô hình huấn luyện theo chính sách nội bộ vẫn thắng ở FPR nghiêm ngặt.
  * Cộng đồng: quá tự tin khi người gán nhãn bất đồng; dễ bị tấn công đối kháng.
* **Liên hệ với đề tài:** Jev vừa là bộ dự báo zero-shot (giải quyết cold-start cho chat CS và bùng hàng), vừa là đầu vào đã hiệu chỉnh cho tầng quyết định Bayes. Chi tiết, rủi ro và giả thuyết kiểm định ở [docs/04_jev_system_one.md](04_jev_system_one.md); thí nghiệm ở notebook 05.

## 4. Gian lận thanh toán, chargeback & bùng hàng COD

* **Gian lận thẻ:** dataset ULB (Dal Pozzolo et al., 2015a) với 0.17% gian lận. Các thách thức chính:
  * Mất cân bằng lớp nặng.
  * *Concept drift* và độ trễ xác minh nhãn (Dal Pozzolo et al., 2018).
  * Chi phí phụ thuộc giá trị giao dịch (Bahnsen et al., 2015, 2016).
  * Phải đánh giá bằng **chia theo thời gian**, không chia ngẫu nhiên.
* **Chargeback / friendly fraud:** khách nhận hàng rồi khiếu nại với ngân hàng để đòi tiền. Nhãn chargeback đến **muộn** (vài tuần), và các ngân hàng thường xếp loại tranh chấp này vào nhóm "fraud".
  IEEE-CIS Fraud Detection (Vesta, Kaggle 2019) gồm khoảng 590k giao dịch e-commerce. Nhãn `isFraud` được ban tổ chức mô tả là gắn với chargeback được báo cáo (theo thảo luận trên trang cuộc thi; chưa kiểm chứng được nguồn chính thức).
* **Bùng hàng COD / RTO (Return-to-Origin):** **gần như không có nghiên cứu học thuật** hay dataset công khai về từ chối nhận hàng COD.
  Các giải pháp hiện có là công cụ thương mại chấm điểm rủi ro RTO (đưa ra hành động như tắt COD, gọi xác nhận, yêu cầu cọc) và nghiên cứu về dự đoán *huỷ đơn* nói chung.
  **Đây là khoảng trống nghiên cứu rõ ràng.** Hướng tiếp cận đề xuất:
  * Định nghĩa schema dữ liệu và bộ mô phỏng (notebook 02).
  * Mô hình hoá **can thiệp** bằng uplift/causal: gọi xác minh hay yêu cầu cọc *làm thay đổi* hành vi khách, nên dự báo rủi ro thuần tuý là chưa đủ.

## 5. Hoá đơn / bằng chứng giả do AI tạo

* **Xu hướng:**
  * SAP Concur báo cáo phát hiện số hoá đơn nghi do AI tạo **tăng 18 lần** sau khi triển khai bộ kiểm tra, với khoảng 1% hoá đơn bị gắn cờ (2025).
  * AppZen ghi nhận hoá đơn AI chiếm khoảng 14% tài liệu gian lận vào tháng 9/2025, trong khi năm 2024 là 0% (theo FT/ICAEW).
  * Trong CS e-commerce, hình thức tương ứng là ảnh sản phẩm hỏng giả, hoá đơn chuyển khoản giả và biên nhận hoàn tiền giả gửi kèm yêu cầu hoàn tiền.
* **Hướng phát hiện:**
  1. **Nhất quán nội dung**: OCR rồi kiểm tra tổng tiền, thuế, ngày và mã đơn so với cơ sở dữ liệu đơn hàng.
  2. **Pháp y tài liệu**: phát hiện copy-move/splicing, font không đồng nhất, dữ liệu forgery cho biên nhận.
  3. **Phát hiện ảnh do mô hình sinh**: Wang et al. (CVPR 2020) chứng minh ảnh do CNN sinh có dấu vết dễ phát hiện; Ojha et al. (CVPR 2023) tổng quát hoá sang các bộ sinh chưa từng gặp.
  4. **Nguồn gốc**: metadata/EXIF và chuẩn content credentials C2PA.
  5. **Đưa điểm rủi ro này vào cùng mô hình ra quyết định** như các tín hiệu khác.
  Thiết kế pipeline chi tiết ở `docs/03_pipeline_hoa_don_gia.md`.

## 6. Khoảng trống & hướng cải tiến đề xuất (được hiện thực trong repo)

| # | Khoảng trống | Đề xuất | Ở đâu |
|---|---|---|---|
| G1 | MCDM chọn mô hình ML bỏ qua **chuẩn hoá** và **bất định đo lường** | Phân tích độ nhạy theo chuẩn hoá; **SMAA ngẫu nhiên trên fold CV** | A2, nb 04 |
| G2 | Quyết định kiểm duyệt thường dùng **ngưỡng cố định 0.5 / luật**, bỏ qua chi phí và năng lực review | **Bayes expected-cost + phân bổ năng lực review** theo lợi ích kỳ vọng; so với scorecard MCDM | nb 04 |
| G3 | Bộ lọc dễ bị **né** | Đặc trưng ký tự + **chuẩn hoá ngược** (deobfuscation) | nb 01 |
| G4 | **Lệch miền** sang chat CS | LLM zero-shot (ít phụ thuộc miền) + **ensemble LLM + TF-IDF**; active learning trên dữ liệu nội bộ | nb 01, 03 |
| G5 | LLM làm "chuyên gia" MCDM: nhất quán? thiên lệch? | Đo CR qua nhiều lần lấy mẫu; ẩn danh tên phương án | nb 03 |
| G6 | Xác suất **chưa hiệu chỉnh** làm hỏng quyết định chi phí | Isotonic/Platt; đo Brier; so sánh quyết định trước và sau hiệu chỉnh | nb 02, 04 |
| G7 | Không có dữ liệu bùng hàng công khai | Schema + bộ mô phỏng có cơ chế công khai; hook dữ liệu nội bộ; hướng uplift modeling | nb 02 |
| G8 | Hoá đơn giả bằng AI | Pipeline nhiều lớp + tích hợp điểm vào triage | docs/03 |
| G9 | Cây đơn kém chính xác nhưng dễ diễn giải | Cắt tỉa cost-complexity (+0.039 AUC, số lá giảm 5 lần), cây xiên, luật cho nhân viên | A1, nb 01–02 |
| G10 | Thiếu dữ liệu gán nhãn (cold-start) cho chat CS / bùng hàng | **Decision model zero-shot (Jev)** + hiệu chỉnh bằng ít nhãn + tầng Bayes chi phí; so với LLM mở không RLCD | nb 05, docs/04 |

---

## Tài liệu tham khảo

* Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering. *ACM DocEng*.
* Arango, A., Pérez, J., & Poblete, B. (2019). Hate speech detection is not as easy as you may think. *SIGIR*.
* Bahnsen, A. C., Aouada, D., & Ottersten, B. (2015). Example-dependent cost-sensitive decision trees. *Expert Systems with Applications*, 42(19).
* Bahnsen, A. C., Aouada, D., Stojanovic, A., & Ottersten, B. (2016). Feature engineering strategies for credit card fraud detection. *Expert Systems with Applications*, 51.
* Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *Findings of EMNLP*.
* Basile, V. et al. (2019). SemEval-2019 Task 5: Multilingual detection of hate speech against immigrants and women in Twitter. *SemEval*.
* Belton, V., & Gear, T. (1983). On a short-coming of Saaty's method of analytic hierarchies. *Omega*, 11(3).
* Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. *Machine Learning*, 106.
* Bordt, S., Nori, H., Rodrigues, V., Nushi, B., & Caruana, R. (2024). Elephants never forget: Memorization and learning of tabular data in large language models. *COLM*. arXiv:2404.06209.
* Brans, J. P., & Vincke, P. (1985). A preference ranking organisation method (PROMETHEE). *Management Science*, 31(6).
* Brauers, W. K. M., & Zavadskas, E. K. (2006). The MOORA method and its application to privatization in a transition economy. *Control and Cybernetics*, 35(2).
* Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
* Breiman, L. (2001). Random forests. *Machine Learning*, 45.
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
* Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Trans. Information Theory*, 16(1).
* Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015a). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*.
* Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018). Credit card fraud detection: A realistic modeling and a novel learning strategy. *IEEE TNNLS*, 29(8).
* Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. *ICWSM*.
* Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7.
* Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7).
* Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*.
* Fang, X. et al. (2024). Large language models (LLMs) on tabular data: Prediction, generation, and understanding – A survey. *TMLR*. arXiv:2402.17944.
* Fernández-Delgado, M., Cernadas, E., Barro, S., & Amorim, D. (2014). Do we need hundreds of classifiers to solve real world classification problems? *JMLR*, 15.
* Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5).
* Frosst, N., & Hinton, G. (2017). Distilling a neural network into a soft decision tree. arXiv:1711.09784.
* Guo, R. et al. (2026). Just Ask Jev: Reinforcement learning for calibrated decisions as a zero-shot detector of AI alignment failures. arXiv:2609.29429.
* Gardner, J., Perdomo, J. C., & Schmidt, L. (2024). Large scale transfer learning for tabular data via language modeling. *NeurIPS*. arXiv:2406.12031.
* Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
* Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63.
* Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets & Benchmarks*. arXiv:2207.08815.
* Hegselmann, S. et al. (2023). TabLLM: Few-shot classification of tabular data with large language models. *AISTATS*. arXiv:2210.10723.
* Hollmann, N. et al. (2025). Accurate predictions on small data with a tabular foundation model. *Nature*, 637, 319–326.
* Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017). Deceiving Google's Perspective API built for detecting toxic comments. arXiv:1702.08138.
* Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.
* Inan, H. et al. (2023). Llama Guard: LLM-based input-output safeguard for human-AI conversations. arXiv:2312.06674.
* Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
* Keshavarz Ghorabaee, M., Zavadskas, E. K., Olfat, L., & Turskis, Z. (2015). Multi-criteria inventory classification using a new method of evaluation based on distance from average solution (EDAS). *Informatica*, 26(3).
* Kumar, D., AbuHashem, Y., & Durumeric, Z. (2024). Watch your language: Investigating content moderation with large language models. *ICWSM*, 18, 865–878.
* Lahdelma, R., Hokkanen, J., & Salminen, P. (1998). SMAA – Stochastic multiobjective acceptability analysis. *European Journal of Operational Research*, 106(1).
* Ling, G., Xue, M., & Ye, Z. (2026). Jev in the wild: A data-driven analysis of the Jev model's functionality, applications and ecosystem. arXiv:2609.30216.
* Lin, J., Zhong, C., Hu, D., Rudin, C., & Seltzer, M. (2020). Generalized and scalable optimal sparse decision trees. *ICML*.
* Madras, D., Pitassi, T., & Zemel, R. (2018). Predict responsibly: Improving fairness and accuracy by learning to defer. *NeurIPS*.
* Markov, T. et al. (2023). A holistic approach to undesired content detection in the real world. *AAAI*.
* Mathew, B. et al. (2021). HateXplain: A benchmark dataset for explainable hate speech detection. *AAAI*.
* McElfresh, D. et al. (2023). When do neural nets outperform boosted trees on tabular data? *NeurIPS Datasets & Benchmarks*. arXiv:2305.02997.
* Mozannar, H., & Sontag, D. (2020). Consistent estimators for learning to defer to an expert. *ICML*.
* Murthy, S. K., Kasif, S., & Salzberg, S. (1994). A system for induction of oblique decision trees. *JAIR*, 2.
* Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
* Ojha, U., Li, Y., & Lee, Y. J. (2023). Towards universal fake image detectors that generalize across generative models. *CVPR*.
* Opricovic, S. (1998). *Multicriteria Optimization of Civil Engineering Systems* (VIKOR). Faculty of Civil Engineering, Belgrade.
* Opricovic, S., & Tzeng, G.-H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2).
* Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*.
* Popov, S., Morozov, S., & Babenko, A. (2020). Neural oblivious decision ensembles for deep learning on tabular data. *ICLR*.
* Prokhorenkova, L. et al. (2018). CatBoost: Unbiased boosting with categorical features. *NeurIPS*.
* Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1. — (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
* Rezaei, J. (2015). Best-worst multi-criteria decision-making method. *Omega*, 53.
* Rodríguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006). Rotation forest: A new classifier ensemble method. *IEEE TPAMI*, 28(10).
* Roy, B. (1968). Classement et choix en présence de points de vue multiples (la méthode ELECTRE). *RIRO*, 2(8).
* Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
* Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. *ACL*.
* Wang, S.-Y., Wang, O., Zhang, R., Owens, A., & Efros, A. A. (2020). CNN-generated images are surprisingly easy to spot... for now. *CVPR*.
* Wang, X., & Triantaphyllou, E. (2008). Ranking irregularities when evaluating alternatives by using some ELECTRE methods. *Omega*, 36(1).
* Yin, W., & Zubiaga, A. (2021). Towards generalisable hate speech detection: A review on obstacles and solutions. *PeerJ Computer Science*, 7.
* Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD*.
* Zampieri, M. et al. (2019). Predicting the type and target of offensive posts in social media (OLID). *NAACL*.
* Zanakis, S. H., Solomon, A., Wishart, N., & Dublish, S. (1998). Multi-attribute decision making: A simulation comparison of select methods. *European Journal of Operational Research*, 107(3).
* Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate before use: Improving few-shot performance of language models. *ICML*.

Nguồn web về xu hướng & mô hình (truy cập 10/2026):
[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) ·
[Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) ·
[ShieldGemma model card](https://ai.google.dev/gemma/docs/shieldgemma/model_card) ·
[Llama Guard 3 1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B) ·
[Granite Guardian](https://arxiv.org/abs/2412.07724) ·
[TabPFN-2.5](https://arxiv.org/abs/2511.08667) ·
[TabICLv2](https://arxiv.org/abs/2602.11139) ·
[AHP + GPT-4](https://arxiv.org/abs/2402.07404) ·
[LLM-based MCDM framework](https://arxiv.org/abs/2502.15778) ·
[ICAEW: AI-generated receipts](https://www.icaew.com/insights/viewpoints-on-the-news/2025/nov-2025/expenses-fraud-how-to-spot-an-ai-generated-receipt) ·
[Skift: AI expense fraud](https://skift.com/2025/11/26/business-travel-faces-a-deluge-of-ai-generated-expense-fraud/) ·
[TweetEval data](https://github.com/cardiffnlp/tweeteval) ·
[HateXplain data](https://github.com/hate-alert/HateXplain) ·
[Davidson data](https://github.com/t-davidson/hate-speech-and-offensive-language)
