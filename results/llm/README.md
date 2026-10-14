# Kết quả notebook 03 (LLM / guard / classifier chuyên dụng)

Chạy trên Colab GPU A100-SXM4-40GB, N_TEST = 300 tin nhắn/task, 6 task.
Các file `results/llm/*.csv` gốc của notebook không được đẩy lên, nên bảng ở đây được **trích từ output đã lưu trong notebook**
bằng `scripts/extract_notebook_results.py`. Số đã làm tròn 3 chữ số, và không có file điểm `.npz`.

* `pr_auc_best_config_by_task.csv`: PR-AUC theo task, lấy cấu hình tốt nhất của từng mô hình. Cột `leakage` đánh dấu task nằm (hoặc nghi nằm) trong dữ liệu huấn luyện của classifier.
* `summary_by_config.csv`, `llm_shots_variant.csv`: trung bình 6 task theo shots/variant. `llm_by_task.csv`, `guard_by_task.csv`, `spec_by_task.csv` lấy từ log khi chạy.
* `ens_llm_tfidf_mean.csv`: ghép LLM zero-shot với TF-IDF LogReg bằng rank-average.
* `dm_llm_ahp.csv`: trọng số AHP do LLM đề xuất, được notebook 04 (B7) sử dụng.

Lưu ý: điểm của qwen3-4b bị bão hoà float32 (nhiều điểm hoà nhau ở p = 1), nên PR-AUC của mô hình này có thể bị **đánh giá thấp**. Lỗi đã được sửa trong `drm.llm`.
