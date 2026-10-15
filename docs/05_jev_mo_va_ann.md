# Vượt 0.90 bắt đầu từ nhãn, không từ mô hình

> Tổng hợp từ bốn nhánh nghiên cứu (bản Jev mở và bộ phân loại zero-shot; SOTA nơ-ron trên từng benchmark; LLM fine-tune, ít nhãn và ghép mô hình; mức trần và nhiễu nhãn), cộng các phép đo trong repo (`results/ceiling`, `results/nn`).
> Nhiều nguồn chỉ đọc được qua trích đoạn tìm kiếm vì arXiv/ACL bị chặn từ môi trường nghiên cứu; các con số đó cần đối chiếu lại với bản gốc trước khi trích dẫn chính thức.

Giữ nguyên định nghĩa nhãn hiện tại thì không mô hình nào đưa được PR-AUC trung bình của sáu benchmark lên trên 0.90. Điều này đúng với bản Jev mở, encoder fine-tune, LLM LoRA và cả ensemble. Hai benchmark có trần thấp do chính nhãn của chúng. Nhãn "hate so với phần còn lại" của Davidson nhiễu đến mức một oracle cũng chỉ đạt khoảng **0.45–0.56 PR-AUC**. Tập test HatEval lệch phân phối và gán nhãn không nhất quán. Vì vậy, kể cả khi bốn benchmark còn lại đạt tuyệt đối 1.0, trung bình vẫn chỉ khoảng **0.89**.

Hướng khả thi có ba bước. Bước một là sửa thước đo. Đánh giá trên toàn bộ tập test thay vì tập con 300 tin, và đo Davidson theo định nghĩa "lăng mạ (hate + offensive) so với sạch". Định nghĩa này sát nhu cầu CS hơn và có trần khoảng 0.999. Bước hai là lấy encoder fine-tune theo miền làm xương sống: DeBERTa-v3-large và RoBERTa-large cho Jigsaw và HateXplain, các mô hình tiền huấn luyện trên tweet cho OLID, HatEval và Davidson. Bước ba là thêm một LLM 7–8B LoRA cùng các thành viên rẻ và đa dạng: TF-IDF ký tự, embedding Qwen3, LLM zero-shot chấm bằng log-odds và WaterSheep. Các thành viên này được ghép bằng rank-average trên tập validation. Làm như vậy, tôi ước tính trung bình đạt **0.88–0.92**, tức là sát ngưỡng chứ chưa chắc vượt. Nếu tách HatEval ra làm bài kiểm tra độ bền, năm benchmark còn lại đạt trung bình khoảng 0.93.

Hiện có hơn 50 bản tái hiện Jev mở, nhưng chưa bản nào công bố kết quả trên sáu benchmark này. Bản mở mạnh nhất vẫn kém Jev khoảng 5 điểm trên một bộ đánh giá độc lập. Giá trị thật của họ mô hình này nằm ở ba điểm: xác suất hiệu chỉnh được bằng vài chục nhãn, độ bền trước né bộ lọc và khả năng khởi động lạnh. Nó không nằm ở PR-AUC. Toàn bộ công thức chạy trong khoảng 6–12 GPU-giờ trên một A100 40 GB của Colab (ước tính).

Cuối cùng, mọi điểm benchmark chỉ là bằng chứng sàng lọc. Chuyển sang dữ liệu khác nguồn thường mất 20–35% PR-AUC. Vì vậy, mốc 0.90 cho chat CS phải được kiểm chứng trên một mẫu chat nội bộ có gán nhãn.

## Trung bình 0.90 bất khả thi khi Davidson-hate kẹt ở trần 0.45–0.56

### Con số 0.79 hiện tại mềm hơn vẻ ngoài

Con số 0.79 là PR-AUC trung bình của **qwen3-4b zero-shot ghép rank-average với TF-IDF LogReg (w = 0.3)**, đo trên **300 tin mỗi task** ([báo cáo kết quả nội bộ](../docs/02_bao_cao_ket_qua.md)). Lưới quét trọng số w được báo cáo ngay trên chính các tập con này ([ens_llm_tfidf_mean.csv](../results/llm/ens_llm_tfidf_mean.csv)). Như vậy, w = 0.3 nhiều khả năng đã được chọn trên dữ liệu đánh giá.

Tập con 300 tin cũng quá nhỏ để so sánh mô hình. Davidson chỉ còn **17 tin dương**, và khoảng tin cậy 95% của PR-AUC trải từ **0.33 đến 0.77**. Jigsaw đạt 0.928 trên tập con nhưng chỉ **0.862** trên toàn bộ 6000 tin test của repo ([subset_ci.csv](../results/ceiling/subset_ci.csv)).

Trên toàn bộ tập test, TF-IDF LogReg đạt Jigsaw 0.862, OLID 0.701, HatEval 0.538, Davidson 0.430, HateXplain 0.889 và SMS 0.983. Trung bình là khoảng **0.734** ([shift.csv](../results/ceiling/shift.csv)). Thử một phép chọn lạc quan: lấy mô hình tốt nhất cho từng task trên tập con, sau khi bỏ các ô bị rò rỉ dữ liệu. Trung bình vẫn chỉ khoảng 0.83 ([pr_auc_best_config_by_task.csv](../results/llm/pr_auc_best_config_by_task.csv); [e4_ensemble.csv](../results/jev/e4_ensemble.csv)). Điểm xuất phát trung thực vì vậy là 0.73–0.79, chưa phải 0.79 chắc chắn.

Phép tính trung bình mới là rào cản thật. Muốn trung bình vượt 0.90, tổng sáu PR-AUC phải vượt 5.4. Giả sử Davidson-hate dừng ở trần 0.56 và HatEval đạt mức lạc quan 0.80. Khi đó, **kể cả nếu Jigsaw, OLID, HateXplain và SMS đều đạt tuyệt đối 1.0, trung bình cũng chỉ là 0.893**. Giới hạn này nằm trong nhãn, không nằm trong mô hình, nên không kiến trúc nào vượt qua được.

### Davidson: nhãn hate phụ thuộc một phiếu dao động

Davidson là nút thắt lớn nhất. Tính lại từ phiếu gốc, nhãn "hate so với phần còn lại" chỉ có **Fleiss κ ≈ 0.25**. Chỉ **18.4%** tweet mang nhãn hate có đồng thuận tuyệt đối, còn **70.6%** có 2 phiếu hate và 1 phiếu offensive. Nghĩa là nhãn của phần lớn lớp dương phụ thuộc vào một phiếu dao động ([labeled_data.csv](https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv)).

Hai ước lượng trần cho kết quả gần nhau. Mô hình beta-binomial giả định người gán nhãn hoán đổi được. Theo mô hình này, một oracle biết đúng xác suất của từng tweet chỉ đạt **PR-AUC ≈ 0.56** và **F1 lớp hate ≈ 0.55**, dù ROC-AUC vẫn khoảng 0.95 ([labeled_data.csv](https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv)). Phân tích "gán nhãn lại" của chính repo cho trần còn chặt hơn: **0.449** ([human_agreement.csv](../results/ceiling/human_agreement.csv)). TF-IDF trên toàn bộ test đã đạt 0.430, tức khoảng 96% trần gán lại, hoặc 77% trần oracle.

Văn liệu khớp với kết quả này. BERT trên Davidson nhị phân hoá đạt macro-F1 0.70 với micro-F1 0.94, suy ra F1 lớp hate chỉ khoảng 0.43 ([Necessity and Sufficiency](https://arxiv.org/pdf/2205.03302)). Trong các lỗi của BERT, 63% là tweet hate bị phân thành offensive ([Mozafari et al.](https://arxiv.org/pdf/1910.12574)).

Tối ưu mạnh trên nhãn này còn có hại. Dữ liệu được thu bằng một từ điển slur. Tweet viết bằng tiếng Anh của người Mỹ gốc Phi bị gán offensive nhiều gấp tới hai lần ([Sap et al. 2019](https://preview.aclanthology.org/menus/P19-1163/)). Ngược lại, định nghĩa "(hate + offensive) so với neither" có **κ ≈ 0.73** và trần oracle khoảng 0.999. TF-IDF đã đạt **0.996** trên định nghĩa này ([label_definition.csv](../results/ceiling/label_definition.csv)).

### HatEval: train dạy một quy luật mà test đảo ngược

HatEval hỏng theo kiểu khác. Tỉ lệ hate gần như không đổi giữa các split (42.0% ở train, 42.2% ở test). Nhưng các tín hiệu mà mô hình học được từ train lại mang nghĩa ngược ở test.

| Từ khoá | Tần suất trong train | Tần suất trong test | P(hate) ở train | P(hate) ở test |
|---|---|---|---|---|
| "bitch" | 12.3% | 48.2% | 0.79 | 0.44 |
| "hoe" | 3.8% | 26.0% | 0.78 | 0.52 |
| "#buildthatwall" | 3.7% | 15.9% | 1.00 | 0.25 |
| "#maga" | 2.8% | 12.4% | 0.89 | 0.29 |

Bảng trên được tính trực tiếp từ split của TweetEval ([TweetEval hate split](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/hate)), và khớp với phân tích của đội MITRE ([MITRE](https://aclanthology.org/S19-2080.pdf)). Tệ hơn nữa, **233 tweet test trùng y hệt tweet train, nhưng chỉ 61% giữ nguyên nhãn**. Một phần lỗi trên HatEval vì vậy không thể học được.

Repo cho thấy cùng hiện tượng. TF-IDF đạt PR-AUC 0.829 khi kiểm định chéo trên train nhưng chỉ 0.538 trên test. Một bộ phân loại tách được tin train khỏi tin test với AUC 0.93 ([shift.csv](../results/ceiling/shift.csv)).

Hệ thống chính thức tốt nhất chỉ đạt **0.651 macro-F1**, và đó là một SVM trên embedding USE. Trên bảng TweetEval, BERTweet đạt 56.4, còn RoBERTa-base chỉ 46.6, thua cả LSTM ([HatEval overview](https://aclanthology.org/S19-2007.pdf); [TweetEval](https://github.com/cardiffnlp/tweeteval)). Trên benchmark này, học sâu từ train là có hại. Trong repo, LLM zero-shot đạt **0.761**, còn TF-IDF đã huấn luyện chỉ đạt 0.511 ([báo cáo kết quả nội bộ](../docs/02_bao_cao_ket_qua.md)).

### Bốn benchmark còn lại có khoảng trống, nhưng hẹp

**HateXplain nhị phân** có α khoảng 0.56 trên toàn bộ dữ liệu và 0.61 trên tập test. 28.3% tin test có tỉ lệ phiếu 2–1, và lớp toxic chiếm 59.4%. Trần oracle khoảng 0.98 ([HateXplain dataset.json](https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json)), còn trần gán lại của repo là 0.944 ([human_agreement.csv](../results/ceiling/human_agreement.csv)).

**OLID** có kỷ lục 0.829 macro-F1 cho mô hình đơn và 0.838 cho ensemble ([OffensEval overview](https://arxiv.org/pdf/1903.08983); [Kungfupanda](https://arxiv.org/pdf/2004.13432)). Nhưng tập test chỉ có 860 tweet, và κ = 0.83 chỉ được đo trên 21 tweet ([Zampieri et al.](https://aclanthology.org/N19-1144.pdf)).

**Jigsaw** đạt 0.9886 AUC trung bình theo cột ở top Kaggle ([Detoxify](https://github.com/unitaryai/detoxify)). RoBERTa-large nhị phân đạt AUROC 0.98 nhưng F1 chỉ 0.76 ([s-nlp card](https://huggingface.co/s-nlp/roberta_toxicity_classifier/blob/71af465f4ed1a47cd8639252191463171834ad5a/README.md)). Chưa ai công bố PR-AUC cho nhãn "toxic bất kỳ".

**SMS** gần bão hoà, nhưng có **414 dòng trùng lặp**. Chia ngẫu nhiên sẽ đưa cùng một tin vào cả train lẫn test và thổi phồng điểm ([SMS mirror](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)).

| Benchmark (định nghĩa) | Tỉ lệ dương | TF-IDF, toàn bộ test | Tốt nhất đã đo trong repo (tập con 300, không rò rỉ) | SOTA công bố (thước đo gốc) | Trần PR-AUC ước tính | Mục tiêu PR-AUC thực tế |
|---|---|---|---|---|---|---|
| Jigsaw toxic bất kỳ | 10.2% | 0.862 | 0.945 (WaterSheep ⊕ TF-IDF) | AUC TB 6 cột 0.9886; nhị phân AUROC 0.98 | chưa ước lượng được (α ≈ 0.45 trên dữ liệu gốc) | 0.90–0.93 |
| TweetEval offensive (OLID) | 27.9% | 0.701 | 0.786 (WaterSheep ⊕ TF-IDF) | macro-F1 0.829–0.838 | chưa ước lượng được | 0.80–0.85 |
| TweetEval hate (HatEval) | 42.2% | 0.538 | 0.761 (llama-3.2-3b 8-shot) | macro-F1 0.651 | bị chặn bởi nhãn không nhất quán | 0.70–0.80 |
| Davidson hate so với phần còn lại | 5.8% | 0.430 | 0.55 (chỉ 17 tin dương) | F1 hate ≈ 0.43–0.51 | **0.45–0.56** | 0.43–0.50 |
| Davidson (hate + offensive) so với neither | 83.2% | 0.996 | — | — | ≈ 0.999 | ≥ 0.995 |
| HateXplain toxic so với normal | 59.4% | 0.889 | 0.923 (TF-IDF Stacking) | nhị phân F1 0.81; 3 lớp AUROC 0.862 | 0.944–0.981 | 0.92–0.94 |
| SMS spam | 13.4% | 0.983 | 1.000 (WaterSheep) | accuracy 99.4%, F1 spam 0.98 | ≈ 1.0 (nhãn khách quan) | ≥ 0.99 (sau khử trùng lặp) |

Cột mục tiêu là suy luận của tôi từ các trần và kết quả ở trên, chưa có số đo cụ thể. Các ô còn lại lấy từ những nguồn đã dẫn trong phần này, cùng với [ExplainableDetector](https://arxiv.org/pdf/2405.08026), [tum-nlp/bert-hateXplain](https://huggingface.co/tum-nlp/bert-hateXplain), [bảng HateXplain của hyper.ai](https://www.hyper.ai/cn/sota/tasks/hate-speech-detection/benchmark/hate-speech-detection-on-hatexplain), [arXiv 1810.01869](https://arxiv.org/pdf/1810.01869) (tỉ lệ dương của Jigsaw) và [Wulczyn et al.](https://arxiv.org/pdf/1610.08914) (α của dữ liệu Wikipedia gốc).

### Đổi định nghĩa Davidson đáng giá hơn mọi nâng cấp mô hình

Bảng trên đổi câu hỏi từ "làm sao đạt 0.90" thành "đang đo cái gì". PR-AUC phụ thuộc tỉ lệ dương, mà tỉ lệ này trải từ 5.8% đến 83%. Cùng một mốc 0.90 vì vậy mang nghĩa rất khác nhau trên từng benchmark. Nên báo cáo PR-AUC kèm tỉ lệ dương, kèm "PR-AUC lift" = (PR-AUC − tỉ lệ dương) / (1 − tỉ lệ dương), ROC-AUC, recall ở FPR cố định và khoảng tin cậy bootstrap. Mục tiêu nên đặt dưới dạng phần trăm của trần ([ghi chú kiểm toán nhãn HateXplain](https://github.com/yogvidwankhede/hatexplain-label-audit)).

Thử một kịch bản điểm giữa với định nghĩa Davidson mới: Jigsaw 0.91, OLID 0.82, HatEval 0.75, Davidson-lăng mạ 0.996, HateXplain 0.93 và SMS 0.99. Trung bình là **0.899**, sát ngưỡng. Từ kịch bản bi quan đến lạc quan, trung bình nằm trong khoảng 0.88–0.92.

Riêng việc đổi định nghĩa Davidson, từ 0.430 lên 0.996, cộng khoảng **+0.094** vào trung bình. Mức này lớn hơn tổng mọi nâng cấp mô hình trên năm benchmark kia, ước tính khoảng +0.07. Đổi định nghĩa không phải gian lận khi nó khớp nhu cầu nghiệp vụ. Trong CS, cả lời thù ghét lẫn lời lăng mạ nhắm vào nhân viên đều dẫn tới cùng một nhóm hành động: cảnh báo hoặc chuyển cho người xử lý.

## Hơn 50 bản Jev mở, chưa bản nào công bố số trên sáu benchmark này

### Bốn họ tái hiện

Đến giữa tháng 10/2026, danh sách awesome-jev có **46 mục mã nguồn mở**. Bảng Jev Decision Index của cộng đồng chạy **70 bản tái hiện** trên 43 benchmark ([omnijev/awesome-jev](https://github.com/omnijev/awesome-jev)). Các bản này chia thành bốn họ. Họ thứ nhất là encoder gắn decision head: WaterSheep, Laya, rlcd-modernbert-151m, Verdict. Họ thứ hai là decoder nhỏ dùng LoRA, kèm head vô hướng hoặc pointer head: Open-Jev, kev, Nimble, decider, eve-rlcd. Họ thứ ba là bộ đọc logit không cần huấn luyện như LitJev và Simple Jev; chính tác giả của chúng nói đây "không phải xác suất đã hiệu chỉnh". Họ thứ tư là các server tương thích giao thức `/v1/systemone`. Bảng dưới gom các ứng viên đáng cân nhắc cho kiểm duyệt tiếng Anh.

| Mô hình | Kiến trúc | Cách huấn luyện | Bằng chứng liên quan kiểm duyệt | Nguồn |
|---|---|---|---|---|
| WaterSheep | ModernBERT-base (~150M) + decision head | Dữ liệu mở cộng quyết định tổng hợp từ Qwen3.5-4B; một temperature cho mỗi loại câu hỏi | HateCheck 75.1% (ECE 0.139, ngoài tập train). Trong repo: PR-AUC trung bình 0.688; Jigsaw 0.913, OLID 0.780, SMS 1.000, **Davidson 0.129** | [WaterSheep](https://github.com/SamratDuttaOfficial/WaterSheep); [e1_pr_auc_by_task.csv](../results/jev/e1_pr_auc_by_task.csv) |
| Laya | ModernBERT-large (421M) | RL theo quy tắc chấm điểm proper, kiểu GRPO | ToxicChat held-out: **accuracy 0.530, macro-F1 0.400**; bản phát hành quá tự tin | [Laya BENCHMARKS](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) |
| Verdict (Manavarya09) | multilingual-e5-small (118M) + head trên embedding đóng băng | Head prototype hoặc logistic, fit vài giây trên CPU | ToxicChat AUROC **0.892 zero-shot → 0.939 với 16 nhãn/lớp → 0.955 với toàn bộ nhãn** | [Verdict](https://github.com/Manavarya09/verdict) |
| kev-4b | Qwen3.5-4B + LoRA r16 + pointer head | Cross-entropy thuần; 10000 ví dụ từ dữ liệu công khai | sysone-bench **0.8556**, bản mở tốt nhất | [kev](https://github.com/jaredpalmer/kev); [sysone panel](https://github.com/instax-dutta/sysone-bench/blob/main/docs/decision-index-panel-coverage.md) |
| decider-4b | Qwen3.5-4B | Nhãn từ teacher Qwen3.5-27B chạy cục bộ | sysone-bench 0.8411 | [decider](https://github.com/Mapika/decider) |
| Open-Jev-9B | Qwen3.5-9B + LoRA + head vô hướng | Soft CE + 0.1 × Brier; temperature fit trên tập hiệu chỉnh riêng | 179/231 bài JevBench | [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) |
| eve-rlcd | Qwen3-0.6B-Base | Bandit REINFORCE với phần thưởng c − p_a | accuracy 0.748 → 0.808, ECE 0.023 | [eve-rlcd](https://github.com/anthony-maio/eve-rlcd) |

### Bằng chứng độc lập vẫn nghiêng về Jev và về bộ phân loại có giám sát

Bằng chứng so sánh độc lập còn mỏng và đang nghiêng về Jev. Trên sysone-bench v2 (niêm phong ngày 26/9/2026), Jev đạt **0.9065** tổng thể, còn Laya đạt 0.6863. Ở bộ kiểm duyệt 144 quyết định, Jev đạt **0.944**, Laya đạt 0.757 ([sysone-bench REPORT](https://github.com/instax-dutta/sysone-bench/blob/main/results/v2/report-20260926/REPORT.md)). Tuy vậy, bộ kiểm duyệt đó chỉ có 48 ca đánh giá và trông giống dữ liệu tổng hợp. Nhãn do một người duyệt lại bản nháp của AI, không có đo đồng thuận giữa người gán nhãn ([manifest](https://github.com/instax-dutta/sysone-bench/blob/main/datasets/v2/manifest.jsonl)). Trên bảng mở rộng gồm 49 mô hình mở, bản mạnh nhất là kev-4b, vẫn **kém Jev 0.0509** ([sysone panel](https://github.com/instax-dutta/sysone-bench/blob/main/docs/decision-index-panel-coverage.md)).

Chưa bản tái hiện nào công bố kết quả trên Jigsaw, Davidson, HateXplain hay OLID. Vì vậy, notebook 05 của repo nhiều khả năng là phép đo đầu tiên của một mô hình kiểu Jev trên các tập này. Kết quả đúng như dự đoán. WaterSheep mạnh trên toxic, offensive và spam, nhưng gần như không phân biệt được ranh giới hate/offensive của Davidson ([e1_pr_auc_by_task.csv](../results/jev/e1_pr_auc_by_task.csv)).

Các bằng chứng ngược chiều đều cho thấy bộ phân loại có giám sát thắng ở FPR chặt. Hai nguồn đầu dưới đây chỉ đọc được qua trích đoạn tìm kiếm, nên các con số cần được xác nhận lại.

Cisco huấn luyện một encoder 1B theo taxonomy riêng. Mô hình này có AUC nhị phân cao nhất trên năm trong sáu tập, và recall cao nhất ở **FPR 0.5% trên cả sáu tập**. Trên hội thoại nhiều lượt, recall của nó là 61%, so với 34% của Jev; Laya gần như không bắt được ca nào ([Cisco](https://blogs.cisco.com/ai/jev-decision-models-on-trial)).

Red Hat đo granite-guardian-hap-125m đạt 80.27% ở 33 ms, so với 86.20% ở 360 ms của Jev. Một DeBERTa-v3-base phát hiện prompt injection đạt 89.01%, vượt mức 86.35% của Jev ([Red Hat Developer](https://developers.redhat.com/articles/2026/10/02/benchmarking-ai-decision-models-against-traditional-guardrails)).

Bài "Type-Safe Is Not Error-Free" cho thấy các mô hình này bám theo *tên* của lựa chọn thay vì rubric. Tỉ lệ lật câu trả lời yes/no của Laya lên tới **76.9%**, và giảm còn 6.5% khi đổi sang tên trung tính ([arXiv 2609.26758](https://arxiv.org/abs/2609.26758)). Vì vậy, câu hỏi kiểm duyệt CS nên dùng nhãn trung tính kèm định nghĩa rõ ràng, và bộ câu hỏi phải được khoá cố định.

### RLCD chưa công khai, và hiệu chỉnh không nâng được PR-AUC

TypeSafe chưa công bố hàm mục tiêu RLCD. Bài "Just Ask Jev" (Guo et al. 2026) chỉ *đánh giá* Jev, không mô tả cách huấn luyện. Bài này có một phát hiện quan trọng: xác suất của Jev hiệu chỉnh tốt khi gộp mọi benchmark, nhưng không hiệu chỉnh tốt trong từng benchmark. ECE gộp là 0.047, nhưng trung vị ECE theo từng benchmark là 0.168, do lệch tỉ lệ dương giữa các benchmark. Fit lại ngưỡng bằng **10 nhãn** đưa trung vị F1 từ 0.706 lên 0.793 ([RLCDAlignBench paper](https://github.com/sumleo/RLCDAlignBench/blob/main/paper.pdf)).

Các công thức mở gần với RLCD đều nhỏ và chạy được trên A100. eve-rlcd dùng phần thưởng c − p_a, tương đương một nửa gradient của điểm Brier. Giai đoạn RL của nó mất 4234 giây, với đỉnh **11.4 GB VRAM** trên một GPU tiêu dùng. Tác giả viết rằng RL "không cải thiện hiệu chỉnh so với warmup, chỉ giữ nguyên" ([eve-rlcd](https://github.com/anthony-maio/eve-rlcd)). Notebook fine-tune kiểu GRPO của Laya mất 4–5 giờ trên 2×T4 cho khoảng 30000 câu hỏi ([Laya](https://github.com/NandhaKishorM/laya)). Fine-tune kev từ checkpoint đã phát hành nâng Kev-4B từ 67.7% lên 73.6% accuracy trong 15 phút trên H100. Fine-tune từ mô hình base thay vì checkpoint chỉ đạt 0.33 ([kev](https://github.com/jaredpalmer/kev)).

Điểm mấu chốt là PR-AUC đo *thứ hạng*. Temperature scaling, loss Brier và RLCD chủ yếu sửa *độ tin cậy của xác suất*, mà một phép biến đổi đơn điệu không đổi thứ hạng. Lợi ích chính của hướng Jev vì vậy là hiệu chỉnh. Trong repo, 50 nhãn mỗi task đưa ECE của WaterSheep từ 0.269 xuống 0.053 ([báo cáo kết quả nội bộ](../docs/02_bao_cao_ket_qua.md)). Lợi ích này phục vụ triage và chính sách Bayes, không giúp tăng PR-AUC.

Các bộ phân loại chính sách mở khác cũng không giải được bài toán này. Qwen3Guard-4B ở chế độ loose đạt F1 82.8 trên ToxicChat ([Qwen3Guard report](https://github.com/QwenLM/Qwen3Guard/blob/main/Qwen3Guard_Technical_Report.pdf)). Nhưng nó gộp lăng mạ vào nhóm "Unethical Acts" và không có nhóm spam, nên các guard model trong repo chỉ đạt 0.45–0.60. gpt-oss-safeguard nhận chính sách viết bằng lời tự nhiên và đạt F1 79.3 trên ToxicChat ở bản 120b. Đây là mô hình suy luận nên chạy chậm ([gpt-oss-safeguard](https://github.com/openai/gpt-oss-safeguard)).

Họ mô hình zero-shot v2.0 của Laurer, trừ các bản có hậu tố "-c", đã học trên Davidson, HateXplain và Jigsaw-wiki ([zeroshot-classifier](https://github.com/MoritzLaurer/zeroshot-classifier)). deberta-v3-large-zeroshot-v2.0 đạt macro-F1 0.461 trên Davidson khi zero-shot, và 0.847 khi có dữ liệu train của chính tập này ([HF card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)). Vì vậy notebook 06 dùng bản "-c" (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c`) làm điểm khởi tạo ([finetune.py](../src/drm/finetune.py)), tránh nhiễm dữ liệu test trên ba tập này.

Kết luận cho hướng Jev như sau. WaterSheep và LLM zero-shot nên làm *thành viên đa dạng* trong ensemble, nhất là cho HatEval và OLID. WaterSheep cũng là thành viên bền nhất trước obfuscation: PR-AUC chỉ giảm 0.9%, so với 4.5% của TF-IDF ([báo cáo kết quả nội bộ](../docs/02_bao_cao_ket_qua.md)). Nếu cần một decision model nội bộ cho triage CS, hãy fine-tune kev-4b hoặc Laya từ checkpoint đã phát hành bằng nhãn CS. Đừng kỳ vọng hướng này phá được trần PR-AUC.

## Encoder fine-tune theo miền là đòn bẩy mô hình lớn nhất; ANN cổ điển chỉ còn vai phụ

### Mức SOTA nơ-ron cho thấy khoảng trống so với TF-IDF

Kết quả nơ-ron đã công bố chỉ ra khoảng trống thật so với TF-IDF nằm ở đâu. Trên OLID, đội NULI vô địch OffensEval 2019 chỉ dùng BERT-base với max_len 64 và 2 epoch, đạt **0.829** macro-F1. TimeLMs-2021 đạt 82.2, còn BERTweet và RoBERTa-base cùng đạt 79.5 ([TweetEval](https://github.com/cardiffnlp/tweeteval); [OffensEval overview](https://arxiv.org/pdf/1903.08983)).

Repo cho một cách quy đổi gián tiếp từ macro-F1 sang PR-AUC. twitter-roberta-base-offensive, vốn được huấn luyện đúng trên OLID train, đạt PR-AUC 0.757 trên tập con ([pr_auc_best_config_by_task.csv](../results/llm/pr_auc_best_config_by_task.csv)). Vì vậy, 0.80–0.85 là mức thực tế cho một mô hình large hoặc một ensemble.

Trên HateXplain 3 lớp, BERT-MRP đạt accuracy 0.704, macro-F1 0.699 và AUROC 0.862. BERT gốc đạt lần lượt 0.690, 0.674 và 0.843 ([SRA paper](https://arxiv.org/pdf/2511.07065); [HateXplain paper](https://arxiv.org/pdf/2012.10289)). Trên Jigsaw, Detoxify dùng bert-base đạt **0.9864**, chỉ kém ensemble RNN vô địch Kaggle 0.22 điểm ([Detoxify](https://github.com/unitaryai/detoxify)). Trên SMS, RoBERTa đạt 99.37% accuracy và F1 lớp spam 0.98 ([ExplainableDetector](https://arxiv.org/pdf/2405.08026)).

### Tiền huấn luyện theo miền thắng các mẹo tối ưu

Lợi ích lớn nhất đã được đo đến từ tiền huấn luyện tiếp trên miền. RoBERTa huấn luyện tiếp trên tweet tăng **+5.7** điểm trên hate và +1.0 trên offensive ([TweetEval](https://github.com/cardiffnlp/tweeteval)). fBERT, được tiền huấn luyện lại trên 1.4 triệu tweet xúc phạm, nâng HatEval từ 0.483 lên **0.596** ([fBERT review](https://liner.com/review/fbert-neural-transformer-for-identifying-offensive-content)). Huấn luyện gộp nhiều dataset hate cho mô hình bền hơn cả dataset đơn tốt nhất ([Antypas & Camacho-Collados 2023](https://aclanthology.org/2023.woah-1.25)). Có một báo cáo rằng tinh chỉnh ngưỡng nâng HatEval từ khoảng 0.55 lên 0.63 macro-F1. Nguồn này chỉ là một poster, đọc qua trích đoạn ([Carleton poster](https://www.sce.carleton.ca/faculty/green/papers/MetaModel_poster_AL_DK.pdf)).

So với đó, các mẹo tối ưu cho lợi ích nhỏ. Layer-wise LR decay chỉ thêm 0.1–0.2 điểm ([Sun et al.](https://arxiv.org/pdf/1905.05583)). Trên một tập hate, DeBERTa-v3 chỉ hơn ModernBERT 0.8 macro-F1 (78.6 so với 77.8) ([ToxiGAN](https://arxiv.org/pdf/2601.03121)).

Các công thức thắng cuộc đều đơn giản: lr 2e-5 đến 3e-5, 2–5 epoch, batch 16–32, max_len 64–128 cho tweet và SMS, 256 cho bình luận Wikipedia, có trọng số lớp, và chọn checkpoint trên validation ([HateXplain config](https://github.com/hate-alert/HateXplain/blob/master/best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json); [Detoxify config](https://github.com/unitaryai/detoxify/blob/master/configs/Toxic_comment_classification_BERT.json)).

### CNN, BiLSTM và fastText chỉ đáng làm thành viên phụ

ANN cổ điển thua transformer một khoảng khá ổn định. Trên OLID, LSTM đạt 71.7 và fastText 73.4, trong khi TimeLMs đạt 82.2. Trên HateXplain, CNN-GRU đạt accuracy 0.627, macro-F1 0.606 và AUROC 0.793, kém BERT khoảng 6 điểm ([TweetEval](https://github.com/cardiffnlp/tweeteval); [HateXplain paper](https://arxiv.org/pdf/2012.10289)).

Có hai ngoại lệ, và cả hai đều có lời giải thích. Trên HatEval, LSTM (52.6) và fastText (50.6) vượt RoBERTa-base (46.6). Lý do là lệch phân phối trừng phạt mô hình học thuộc train mạnh hơn. Trên Jigsaw 2018, một Bi-GRU đơn chạy trên embedding fastText và GloVe đạt 0.9862. Nhưng đội vô địch chỉ thắng nhờ ghép hàng chục mô hình, kèm back-translation và pseudo-label ([Kaggle 1st place](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/toxic-crusaders-1st-place-solution-overview)).

Vậy CNN, BiLSTM và fastText không phải đường đến 0.90. Giá trị của chúng nằm ở chỗ rẻ và mắc lỗi khác kiểu. Bản dùng đặc trưng ký tự còn bền hơn hệ thống dùng đặc trưng từ trước né bộ lọc ([Gröndahl et al.](https://arxiv.org/abs/1808.09115)).

### Embedding đóng băng mạnh khi ít nhãn

Embedding đóng băng cộng bộ phân loại tuyến tính là lựa chọn mạnh khi có ít nhãn, nhưng chất lượng embedding chênh nhau rất lớn. MTEB ToxicConversations đo với 16 ví dụ mỗi nhãn và logistic regression ([mteb](https://pypi.org/project/mteb/1.36.15/)). Trên benchmark này, Qwen3-Embedding-4B đạt accuracy **91.44**, so với 67.04 của bge-base-en-v1.5 và 61.05 của all-mpnet-base-v2 ([MTEB results](https://github.com/embeddings-benchmark/results)). Cột "AP" của MTEB được tính trên dự đoán cứng, nên không phải PR-AUC xếp hạng.

SetFit học từ 8 ví dụ trong khoảng 30 giây trên V100 ([SetFit blog](https://raw.githubusercontent.com/huggingface/blog/main/setfit.md)). Fine-tune embedder gốc LLM cho hate ngầm ẩn cho mức tăng tới **+20.35 macro-F1 khi đánh giá chéo dataset** ([arXiv 2508.20750](https://arxiv.org/abs/2508.20750)).

Khi có hàng nghìn nhãn như sáu benchmark này, fine-tune toàn bộ mô hình sẽ vượt embedding cộng head tuyến tính. Dù vậy, embedding vẫn đáng giữ làm thành viên nhanh. Đội hạng 3 Jigsaw Agile dùng bge-base và Qwen3-Embedding-0.6B bên cạnh DeBERTa ([3rd place](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution)).

## LoRA LLM và ghép mô hình cộng thêm một đến ba điểm, không phải mười

### LLM LoRA thường chỉ ngang một encoder mạnh

Các bằng chứng sạch cho thấy LLM 7–8B fine-tune bằng LoRA thường chỉ *ngang* một encoder mạnh trên văn bản ngắn.

Blog Hugging Face so sánh trên tập Disaster Tweets. RoBERTa-large đạt F1 **0.8077** trong 538 giây, dùng khoảng 9 GB VRAM. Mistral-7B chỉ đạt 0.7364 và Llama-2-7B đạt 0.7638, trong khi tốn khoảng bốn lần thời gian và bộ nhớ ([HF LoRA blog](https://raw.githubusercontent.com/huggingface/blog/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md)). Nghiên cứu Ettin huấn luyện encoder và decoder trên cùng dữ liệu, cùng công thức. Ở đó, một encoder 150M đạt 89.2 trên MNLI, vượt decoder 400M ở mức 88.2 ([Ettin](https://raw.githubusercontent.com/huggingface/blog/main/ettin.md)).

Thắng lợi rõ nhất của decoder là cuộc thi Jigsaw Agile 2025, nơi đầu vào có kèm văn bản luật. Ở đó, ghép 2×Qwen3-14B đạt **0.92049** AUC private, so với 0.8894 của DeBERTa-v3-small. Một Qwen3-4B LoRA đơn lẻ đạt 0.91425, so với 0.92930 của đội vô địch ([3rd place](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution); [jigsaw-rule-classifier](https://github.com/alvaromendizabal/jigsaw-rule-classifier)).

Các tuyên bố "LLM thắng 8–14 điểm" thường so với mBERT, một baseline yếu cho tiếng Anh ([arXiv 2506.12744](https://arxiv.org/abs/2506.12744)). Nên lập kế hoạch với chênh lệch khoảng ±2 macro-F1 so với DeBERTa-v3-large. Giá trị của LLM ở đây là đa dạng lỗi, không phải thay thế encoder.

### LLM zero-shot kém trong miền nhưng mạnh khi dữ liệu lệch

LLM zero-shot yếu trên dữ liệu cùng phân phối với train. GPT-3.5 chỉ đạt 0.39 macro-F1 trên HateXplain ([Roy et al.](https://aclanthology.org/2023.findings-emnlp.407/)). RoBERTa fine-tune thắng LLM zero-shot trên 16 dataset ([Edwards & Camacho-Collados](https://arxiv.org/abs/2403.17661)). Ngược lại, LLM zero-shot mạnh khi dữ liệu lệch phân phối, như HatEval trong repo: 0.761 so với 0.511.

LLM cũng có hai điểm yếu đã được ghi nhận. Thứ nhất, chúng quá nhạy với từ chỉ danh tính và cho độ tin cậy cực đoan ([ACL 2024](https://aclanthology.org/2024.acl-long.652/)). Thứ hai, precision sụp đổ khi tỉ lệ dương thấp. Trên một ngày Twitter thật với khoảng 0.7% hate, AP của các detector công khai chỉ còn **9.4%**, so với khoảng 40% trên dataset học thuật ([HateDay](https://aclanthology.org/2025.acl-long.115.pdf)).

Bài học kỹ thuật quan trọng nhất là chấm điểm bằng logit chứ không bằng nhãn sinh ra. MULI đặt logistic regression thưa lên logit token đầu tiên của LLM. Nó đạt **42.54% TPR ở FPR 0.1%** trên ToxicChat, so với 5.25% của Llama Guard ([MULI](https://arxiv.org/abs/2405.18822)). Repo đã gặp đúng vấn đề này khi sửa lỗi bão hoà float32, vốn làm nhiều điểm hoà nhau ở p = 1. Vì vậy qwen3-4b cần được chấm lại bằng log-odds trước mọi so sánh.

### Ghép mô hình cho lợi ích nhỏ nhưng đều đặn

Ghép mô hình cho lợi ích nhỏ nhưng đều đặn, và lợi ích lớn hơn khi các thành viên khác kiến trúc. Trên Jigsaw 2018, một Detoxify đơn lẻ đạt 98.64, còn ensemble top đạt 98.86 (+0.22 điểm AUC). Trên Unintended Bias 2019, khoảng cách là +0.99 điểm ([Detoxify](https://github.com/unitaryai/detoxify)). Bagging BERT trên TRAC thêm khoảng 2 F1 ([TRAC 2020](https://aclanthology.org/2020.trac-1.9.pdf)). Một bản tái hiện Jigsaw Agile ghép năm mô hình được thêm +0.0058 AUC, với khoảng tin cậy [+0.0001; +0.0117]. Bản tái hiện này loại bỏ các bộ trọng số tối ưu quá mạnh, vì chúng làm tụt điểm trên một chính sách ([TOP_SOLUTION_INTEGRATION](https://raw.githubusercontent.com/alvaromendizabal/jigsaw-rule-classifier/main/docs/TOP_SOLUTION_INTEGRATION.md)).

Trong repo, lợi ích của ghép mô hình phụ thuộc mạnh vào task. Ghép WaterSheep với TF-IDF chỉ nâng trung bình từ 0.756 lên 0.773. Nhưng từng task lại rất khác nhau: OLID tăng từ 0.662 lên 0.786, HatEval tăng từ 0.511 lên 0.610, còn Davidson giảm từ 0.550 xuống 0.492 ([e4_ensemble.csv](../results/jev/e4_ensemble.csv)). Trọng số vì vậy phải chọn riêng cho từng task, trên tập validation.

### Chưng cất, nhãn mềm và hiệu chỉnh phục vụ triển khai, không nâng trần

Chưng cất và pseudo-label phục vụ triển khai, không nâng trần. Nhãn bạc do LLM gán chỉ thu hồi được 60–75% khoảng cách so với nhãn vàng ([Lu & Smith](https://arxiv.org/abs/2504.15432)). OpenAI dùng dữ liệu tổng hợp nâng AUPRC hate từ 0.417 lên 0.551, nhưng mức lợi này co lại khi đã có đủ nhãn thật ([Markov et al.](https://arxiv.org/abs/2208.03274)). Nhãn mềm lấy từ tỉ lệ phiếu (Davidson, HateXplain) chủ yếu cải thiện hiệu chỉnh và khả năng dự đoán bất đồng. Lợi ích trên F1 cứng nhỏ ([Uma et al. JAIR](https://jair.org/index.php/jair/article/view/12752)). Temperature scaling đưa ECE trong miền xuống dưới 1%, nhưng không thay đổi PR-AUC ([Desai & Durrett](https://github.com/shreydesai/calibration)).

Độ bền trước né bộ lọc mới là chỗ dễ mất điểm khi triển khai. Trên HateCheck, BERT fine-tune chỉ đúng **37.6–48%** ở các ca chèn dấu cách hoặc viết leetspeak ([HateCheck](https://aclanthology.org/2021.acl-long.4.pdf)). Tấn công ở mức ký tự cắt hơn 30% recall ([Kurita et al.](https://arxiv.org/abs/1912.06872)). Cách phòng thủ có ba lớp: một bộ chuẩn hoá tất định, tăng cường dữ liệu bằng nhiễu, và một thành viên dùng đặc trưng ký tự. Trong repo, bộ chuẩn hoá ngược đã lấy lại 64–73% recall bị mất ([báo cáo kết quả nội bộ](../docs/02_bao_cao_ket_qua.md)).

## Công thức bốn tầng chạy trong khoảng 6–12 GPU-giờ trên một A100

### Tầng 0: sửa giao thức đánh giá trước khi huấn luyện

Tầng 0 không cần GPU nhưng không thể bỏ qua. Việc đầu tiên là chấm trên toàn bộ tập test: Jigsaw 6000 tin (tập test của repo), OLID 860, HatEval 2970, Davidson 4957, HateXplain 1924 và SMS 1115. Khử trùng lặp SMS trước khi chia. Báo cáo Davidson theo cả hai định nghĩa. Giữ tập test chính thức của HatEval, đồng thời báo cáo thêm bản đã bỏ 233 tin trùng.

Mỗi mô hình chạy 3 seed và báo cáo khoảng tin cậy bootstrap 95%. Lưu điểm `.npz` sau mỗi mô hình, vì lần chạy Colab trước không lưu điểm nên chưa làm được paired bootstrap ([báo cáo kết quả nội bộ](../docs/02_bao_cao_ket_qua.md)).

Loại các checkpoint có rò rỉ dữ liệu: toxic-bert trên Jigsaw, twitter-roberta-base-hate-latest (nghi đã học HatEval), và các mô hình zero-shot v2.0 không có hậu tố "-c". Đặt bộ chuẩn hoá NFKC, homoglyph và leetspeak trước mọi mô hình. Tạo thêm một bản test bị obfuscate 30% để đo độ bền.

Thời gian ở các tầng sau là ước tính, ngoại suy từ thông lượng điển hình. Tôi không tìm thấy phép đo công bố nào trên A100 cho đúng các tập này. Hai mốc tham chiếu: QLoRA r = 256 trên 50000 ví dụ mất khoảng 3 giờ trên A100, và QLoRA chậm hơn LoRA khoảng 39% ([Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)).

### Tầng 1–4: cấu hình cụ thể

| Tầng | Thành phần | Cấu hình | Áp dụng cho | Thời gian A100 40 GB (ước tính) |
|---|---|---|---|---|
| 1. Encoder fine-tune | `microsoft/deberta-v3-large` (Jigsaw, HateXplain; thêm `FacebookAI/roberta-large` 1 seed để đa dạng); `cardiffnlp/twitter-roberta-large-2022-154m` và `vinai/bertweet-large` (OLID, HatEval, Davidson); `diptanu/fBERT` (HatEval); `roberta-base` (SMS) | AdamW; lr 1e-5 (large) hoặc 2e-5 (base); warmup 6%; weight decay 0.01; 2–4 epoch; batch 16–32; max_len 64–128 (tweet, SMS, HateXplain) hoặc 256 (Jigsaw); bf16; BCE với `pos_weight`; dừng sớm theo PR-AUC trên validation; ngưỡng chọn trên validation. Nhãn mềm (tỉ lệ phiếu) cho Davidson và HateXplain; dữ liệu train của OLID, HateXplain và Davidson làm dữ liệu phụ cho HatEval; một ablation label smoothing 0.05 | Cả 6 task; 3 seed cho Jigsaw, OLID, HatEval, HateXplain | 3–5 giờ (base 3–6 phút mỗi lần chạy; large 10–20 phút; Jigsaw ở max_len 256 tốn gấp đôi) |
| 2. LLM LoRA | `Qwen/Qwen2.5-7B-Instruct` (hoặc Qwen3-8B, Llama-3.1-8B) | LoRA r 16, alpha 32, dropout 0.05, all-linear; lr 1e-4; 1 epoch; batch 8 × accum 2; gradient checkpointing; bf16; max_len 128–256. Điểm = logit(Yes) − logit(No) tại vị trí trả lời. Prompt chứa định nghĩa chính sách CS với tên nhãn trung tính. Đặt `pad_token = eos` trên cả tokenizer lẫn `model.config`. Chỉ dùng QLoRA cho 14B | Jigsaw (lấy mẫu khoảng 15000), OLID, HatEval, HateXplain | 2–6 giờ (0.5–1.5 giờ mỗi epoch cho 10000–20000 văn bản ngắn) |
| 3. Thành viên rẻ, đa dạng | TF-IDF từ + ký tự với LogReg (đã có); Qwen3-Embedding-4B đóng băng + LogReg `class_weight="balanced"`; qwen3-4b và llama-3.2-3b zero-shot chấm lại bằng log-odds; WaterSheep với câu hỏi riêng cho từng task | Embedding dùng instruction kiểu MTEB; LLM chấm bằng logit, không dùng nhãn sinh ra | Cả 6 task, đặc biệt HatEval và OLID | Dưới 1 giờ (khoảng 17800 tin test × 30–40 ms) |
| 4. Ghép và hiệu chỉnh | Đổi điểm sang thứ hạng hoặc logit. Mặc định rank-average trọng số đều. Nếu tối ưu: trọng số không âm cực đại AP trên validation, hoặc một logistic-regression stacker | Chỉ giữ trọng số khi lợi ích vượt khoảng tin cậy bootstrap. Platt hoặc temperature trên đầu ra cuối. Hiệu chỉnh prior khi tỉ lệ dương lúc triển khai khác lúc huấn luyện. Ngưỡng theo FPR mục tiêu; báo cáo TPR@FPR 1% | Cả 6 task | Vài phút |

Tầng 1 chắc chắn nhất. Cấu hình bám sát các công thức thắng cuộc đã dẫn ở trên và khớp với `FTConfig` đã có trong repo ([finetune.py](../src/drm/finetune.py)).

HatEval cần xử lý riêng vì học sâu từ train ở đây là có hại. Cho các thành viên zero-shot (LLM, WaterSheep) trọng số cao hơn. Huấn luyện encoder trên dữ liệu gộp nhiều nguồn thay vì chỉ HatEval train. Chọn ngưỡng trên validation.

Tầng 2 chỉ chạy sau khi Tầng 1 đã ổn định. Mục đích của nó là đa dạng lỗi, theo mô hình ghép "nhanh + chậm" của đội hạng 3 Jigsaw Agile ([3rd place](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/3rd-place-solution)).

Nếu nhóm cần thêm một decision model kiểu System One chạy nội bộ cho triage CS, có thể làm trên cùng A100. Fine-tune kev-4b từ checkpoint đã phát hành (`--init_from`) với loss soft CE + 0.1 × Brier trên câu hỏi CS có nhãn. Fit temperature trên một tập hiệu chỉnh riêng. Nếu muốn, thêm một giai đoạn bandit REINFORCE kiểu eve-rlcd; giai đoạn này chỉ cần khoảng 11.4 GB VRAM ([Open-Jev](https://github.com/Zefan-Cai/Open-Jev); [eve-rlcd](https://github.com/anthony-maio/eve-rlcd)).

### Kịch bản kỳ vọng và các điểm chưa chắc chắn

Nếu đạt các mục tiêu trong bảng trần, trung bình sáu benchmark là **0.79–0.84 khi giữ Davidson-hate** và **0.88–0.92 khi dùng Davidson-lăng mạ**. Đây là suy luận, chưa phải số đo.

Có ba nguồn bất định lớn. Thứ nhất, chưa ai công bố PR-AUC cho OLID, HatEval hay Jigsaw "toxic bất kỳ", nên các dải mục tiêu này phải được kiểm chứng bằng chính lần chạy notebook 06. Thứ hai, nhiều con số văn liệu chỉ lấy được qua trích đoạn tìm kiếm, gồm số của Cisco, Red Hat, fBERT và kết quả tinh chỉnh ngưỡng trên HatEval. Thứ ba, các trần của Davidson và HateXplain dựa trên giả định người gán nhãn hoán đổi được cho nhau. Hiệu ứng có hệ thống của người gán nhãn có thể cho phép mô hình vượt trần một chút. Ngược lại, lệch phân phối kiểu HatEval kéo mô hình xuống xa dưới trần.

## Thực nghiệm trong repo

### ANN huấn luyện từ đầu trên CPU (`scripts/run_nn_benchmark.py` → `results/nn/`)
Đánh giá trên **toàn bộ** tập test; không dùng mô hình tiền huấn luyện. Bốn mô hình chạy song song trên 4 CPU, nên thời gian chỉ mang tính tương đối.

| Mô hình | Jigsaw | OLID | HatEval | Davidson | HateXplain | SMS | **PR-AUC TB** | ROC-AUC TB | Thời gian huấn luyện (trung vị) |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (nb 01) | 0.862 | 0.701 | 0.538 | 0.430 | 0.889 | 0.983 | **0.734** | 0.855 | vài giây |
| BiGRU | 0.836 | 0.698 | 0.427 | 0.384 | 0.884 | 0.971 | 0.700 | 0.834 | 196 s |
| MLP trên TF-IDF | 0.806 | 0.643 | 0.498 | 0.360 | 0.876 | 0.970 | 0.692 | 0.825 | 429 s |
| fastText-style | 0.785 | 0.677 | 0.468 | 0.370 | 0.877 | 0.975 | 0.692 | 0.836 | 229 s |
| TextCNN | 0.820 | 0.680 | 0.447 | 0.344 | 0.876 | 0.973 | 0.690 | 0.830 | 108 s |
| BiGRU ⊕ TF-IDF (trung bình hạng) | 0.865 | 0.720 | 0.486 | 0.417 | 0.895 | 0.980 | 0.727 | 0.852 | |
| 4 ANN ⊕ TF-IDF | 0.857 | 0.717 | 0.480 | 0.417 | 0.897 | 0.984 | 0.725 | 0.852 | |

* **ANN không tiền huấn luyện thua TF-IDF** ở mọi task (−0.03 đến −0.04 PR-AUC trung bình). Kết quả này khớp với văn liệu: LSTM/fastText kém transformer 8–10 điểm trên OLID.
* **Ghép với TF-IDF** chỉ giúp OLID (+0.02) và HateXplain (+0.01), nhưng làm giảm HatEval (−0.05, do lệch phân phối) và Davidson. Trung bình không tăng.
* → **Đòn bẩy là tiền huấn luyện**, không phải kiến trúc mạng. ANN cổ điển chỉ nên làm thành viên rẻ trong ensemble. Encoder tiền huấn luyện được fine-tune trong notebook 06.

### Mức trần và giao thức đo (`scripts/analyze_ceilings.py` → `results/ceiling/`)
| Phép đo | Kết quả |
|---|---|
| Davidson: một người gán nhãn so với đa số của những người còn lại (lớp hate) | precision 0.20, recall 0.43, **F1 0.28**; chỉ 18% tin hate được cả nhóm đồng ý |
| Davidson: PR-AUC "gán nhãn lại" của mô hình hoàn hảo | **0.45** (repo); 0.56 (mô hình beta-binomial khác); TF-IDF đã đạt 0.43 |
| HateXplain: F1 của một người gán nhãn so với đa số; trần gán nhãn lại | F1 0.91; PR-AUC trần **0.94** (0.98 theo mô hình khác) |
| HatEval: PR-AUC kiểm định chéo trên train so với trên test; AUC phân biệt train/test | 0.83 so với **0.54**; **0.93** |
| Tin test trùng với train | SMS 10.8% (nhãn nhất quán), HatEval 7.9% (chỉ **61%** cùng nhãn), các task khác < 1% |
| Khoảng tin cậy 95% của PR-AUC trên tập con 300 tin | Davidson [0.33, 0.77] với 17 ca dương; Jigsaw [0.84, 0.99] so với 0.862 trên toàn bộ test |
| Davidson theo định nghĩa "lăng mạ" (hate + offensive so với bình thường) | PR-AUC **0.996**, ROC-AUC 0.984 (TF-IDF) |

### Notebook 06 (Colab A100): đang chờ chạy
Hiện thực công thức ở mục trên, gồm:
* encoder fine-tune (DeBERTa-v3-large, Twitter-RoBERTa-large-2022, RoBERTa-large, BERTweet-large, fBERT);
* WaterSheep fine-tune có giám sát, so với ModernBERT-base cùng kiến trúc;
* DeBERTa-v3-large-zeroshot-**c**;
* Qwen3-Embedding + LogReg; Qwen3-8B LoRA;
* ghép tham lam trên validation.

Notebook chạy trên 6 task gốc và 2 biến thể `davidson_abusive`, `sms_spam_dedup`. Kết quả được đánh giá trên toàn bộ test kèm khoảng tin cậy, cùng độ bền trước obfuscation và chi phí.

## Kết luận

Câu hỏi "mô hình nào đưa 0.79 lên trên 0.90" hoá ra đặt sai chỗ. Đòn bẩy lớn nhất nằm ở định nghĩa nhãn và giao thức đo. Riêng việc đo Davidson theo định nghĩa lăng mạ đã cộng khoảng +0.09 vào trung bình. Mức này lớn hơn toàn bộ nâng cấp mô hình cộng lại (ước tính +0.07). Trong khi đó, chuyển từ tập con 300 tin sang toàn bộ tập test có thể làm con số hiện tại *giảm* trước khi tăng.

Mỗi hướng mô hình đều có vai trò và giới hạn riêng. Jev mở cho xác suất hiệu chỉnh và độ bền, không cho thứ hạng tốt hơn. Encoder fine-tune theo miền là xương sống. LLM LoRA và ensemble thêm một đến ba điểm nhờ đa dạng lỗi. ANN cổ điển chỉ còn là thành viên rẻ. Mục tiêu nên viết lại thành "đạt ít nhất 90% trần trên từng benchmark, báo cáo kèm tỉ lệ dương và khoảng tin cậy". Cách viết này trung thực và kiểm chứng được. Còn "trung bình trên 0.90" thì phụ thuộc vào việc chọn benchmark nào.

Với chat CS e-commerce, bài học này càng quan trọng. Các lối tắt từ vựng giúp ăn điểm benchmark, như hashtag của HatEval hay từ điển slur của Davidson, chính là thứ không chuyển giao được. Đổi nguồn dữ liệu đã làm repo mất 20–35% PR-AUC ([báo cáo kết quả nội bộ](../docs/02_bao_cao_ket_qua.md)), và văn liệu ghi nhận mức sụt khoảng 30 điểm macro-F1 khi đánh giá chéo dataset hate ([Yin & Zubiaga](https://arxiv.org/pdf/2102.08886)). Ở tỉ lệ dương thực tế, AP còn có thể rơi xuống một chữ số phần trăm như HateDay cho thấy.

Vì vậy, sau notebook 06, phần lớn ngân sách nên chuyển từ GPU-giờ cho benchmark sang hai việc. Việc thứ nhất là gán nhãn một mẫu chat CS nội bộ, có đo độ đồng thuận và chủ động lấy mẫu các ca offensive nhưng không hate. Việc thứ hai là dùng LLM LoRA tốt nhất làm teacher để chưng cất xuống một encoder đủ nhanh cho thời gian thực. Đó cũng là nơi lời hứa "xác suất đã hiệu chỉnh" của họ Jev thực sự có giá trị: tầng triage Bayes, nơi xác suất quyết định chi phí, chứ không phải bảng xếp hạng PR-AUC.
