# Kết quả notebook 06: fine-tune ANN (Colab A100, 2026-10-15)

Trích từ output đã lưu trong [`notebooks/06_finetune_ann_colab.ipynb`](../../notebooks/06_finetune_ann_colab.ipynb) (code `e8c8a20`, A100-SXM4-40GB, bf16). Các file điểm `.npz`/`.json` của lần chạy nằm ở Google Drive của người chạy (`drm_results/finetune`), chưa đưa vào repo.

Toàn bộ tập test (không dùng tập con 300 tin). Mỗi mô hình 1 seed, trừ khi ghi khác. Số làm tròn 3 chữ số.

## PR-AUC theo task

| Mô hình | davidson_abusive | davidson_hate | hatexplain_toxic | jigsaw_toxic | sms_spam | sms_spam_dedup | tweeteval_hate | tweeteval_offensive | **TB** | số task ≥ 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|
| ENS all (rank-average mọi mô hình) | 0.998 | 0.500 | 0.932 | 0.935 | 0.999 | 1.000 | 0.584 | 0.834 | **0.848** | 5 |
| ENS greedy (chọn trên validation) | 0.998 | 0.504 | 0.935 | 0.934 | 0.997 | 0.999 | 0.546 | 0.833 | 0.843 | 5 |
| qwen3-emb-4b + LogReg | 0.995 | 0.447 | 0.904 | 0.920 | 0.993 | 0.992 | 0.653 | 0.821 | 0.841 | 5 |
| deberta-v3-large zero-shot-c (fine-tune) | 0.997 | 0.476 | 0.927 | 0.920 | 0.997 | 0.997 | 0.582 | 0.802 | 0.837 | 5 |
| qwen3-8b LoRA | 0.998 | 0.485 | 0.925 | 0.925 | 0.992 | 0.996 | 0.549 | 0.825 | 0.837 | 5 |
| twitter-roberta-large-2022 | 0.997 | 0.452 | 0.923 | 0.915 | 0.993 | 0.998 | 0.582 | 0.807 | 0.834 | 5 |
| bertweet-large | 0.998 | 0.462 | 0.925 | 0.912 | 0.992 | 0.995 | 0.535 | 0.817 | 0.830 | 5 |
| roberta-large | 0.996 | 0.451 | 0.918 | 0.906 | 0.993 | 0.999 | 0.550 | 0.812 | 0.828 | 5 |
| deberta-v3-large | 0.997 | 0.469 | 0.930 | 0.910 | 0.997 | 0.996 | 0.526 | 0.798 | 0.828 | 5 |
| qwen3-emb-8b + LogReg | 0.998 | 0.421 | 0.916 | 0.923 | 0.994 | 0.985 | 0.547 | 0.815 | 0.825 | 5 |
| watersheep-ft (encoder WaterSheep fine-tune) | 0.997 | 0.447 | 0.906 | 0.909 | 0.997 | 0.989 | 0.513 | 0.787 | 0.818 | 5 |
| fbert (110M) | 0.997 | 0.419 | 0.914 | 0.892 | 0.973 | 0.993 | 0.532 | 0.814 | 0.817 | 4 |
| modernbert-base | 0.998 | 0.415 | 0.909 | 0.902 | 0.992 | 0.996 | 0.469 | 0.792 | 0.809 | 5 |
| TF-IDF LogReg | 0.996 | 0.430 | 0.889 | 0.862 | 0.983 | 0.988 | 0.538 | 0.701 | 0.798 | 3 |

Khoảng tin cậy 95% (bootstrap 500 lần) của mô hình tốt nhất mỗi task: jigsaw 0.920–0.948, hatexplain 0.922–0.947, tweeteval_offensive 0.787–0.878, tweeteval_hate 0.623–0.680, davidson_hate 0.449–0.568.

**Lưu ý khi đọc:**
- Bảng "mô hình tốt nhất mỗi task" trong notebook (ô 11) chọn mô hình **theo điểm trên test**, nên lạc quan. Dùng ENS greedy (chọn trên validation) hoặc một mô hình cố định để so sánh công bằng.
- Validation cũng dùng để dừng sớm và chọn C, và cùng phân phối với train. Với TweetEval (test lệch phân phối), số trên test là thước đo đúng hơn validation.
- Chưa có paired bootstrap giữa các mô hình. Chênh lệch dưới khoảng 0.01 ở cột trung bình chưa đủ để kết luận.
- Backbone tiền huấn luyện (Qwen3, DeBERTa…) có thể đã thấy Jigsaw/TweetEval khi pretrain; chưa kiểm.

## Độ bền trước né bộ lọc (% PR-AUC thay đổi, trung bình các task)

| Mô hình | 30% từ bị làm nhiễu | sau khi chuẩn hoá ngược |
|---|---|---|
| qwen3-emb-4b + LogReg | −1.30 | +0.30 |
| qwen3-emb-8b + LogReg | −1.52 | −0.22 |
| qwen3-8b LoRA | −1.66 | −0.68 |
| bertweet-large | −2.04 | −0.15 |
| twitter-roberta-large-2022 | −2.69 | −0.44 |
| deberta-v3-large zero-shot-c | −2.82 | −1.38 |
| deberta-v3-large | −2.90 | −0.58 |
| fbert | −3.30 | −1.01 |
| roberta-large | −3.44 | −1.20 |
| watersheep-ft | −3.93 | −1.10 |
| modernbert-base | −4.45 | −1.23 |

## Chi phí (trung vị theo task)

| Mô hình | ms/tin (A100) | tham số huấn luyện | phút huấn luyện |
|---|---|---|---|
| qwen3-emb-4b + LogReg | 4.03 | 2 560 (đầu LogReg) | 0.96 |
| qwen3-emb-8b + LogReg | 6.14 | 4 096 | 1.46 |
| qwen3-8b LoRA | 3.56 | 43.7M | 9.72 |
| deberta-v3-large (cả hai biến thể) | 0.57–0.60 | 435M | 4.2 |
| roberta-large / twitter-roberta-large / bertweet-large | 0.46–0.48 | 355M | 2.9–4.4 |
| watersheep-ft / modernbert-base | 0.22 | 150M | 0.9 |
| fbert | 0.11 | 109M | 0.58 |

Số ms/tin đo trên GPU A100, không dùng để suy ra độ trễ trên CPU. Đo CPU một lần (laptop i7-13620H, onnxruntime int8, chưa có script đo trong repo): BERT-base khoảng 29 ms/tin 1 thread, MiniLM-L6 khoảng 6 ms, ModernBERT-base khoảng 145 ms (int8 kém với ModernBERT).
