# Mô hình ra quyết định & phân loại ML/LLM cho Trust & Safety trong CS e-commerce

> **Kết quả chính:** xem [docs/02_bao_cao_ket_qua.md](docs/02_bao_cao_ket_qua.md).

Nghiên cứu tổng hợp nhằm **phát hiện** và **ra quyết định xử lý** các tình huống rủi ro mà đội Customer Service gặp hằng ngày:

| Bài toán | Ví dụ | Notebook |
|---|---|---|
| Ngôn từ độc hại / công kích / thù ghét | khách chửi bới, đe doạ nhân viên CS | 01, 03 |
| Spam / lừa đảo | tin nhắn trúng thưởng, dẫn link thanh toán ngoài sàn | 01, 03 |
| Gian lận thanh toán & **chargeback** | thẻ ăn cắp, nhận hàng rồi đòi hoàn tiền (friendly fraud) | 02 |
| **Bùng hàng COD** | đặt rồi không nhận, "boom hàng" | 02 |
| Hoá đơn / bằng chứng giả do AI | biên lai chuyển khoản giả, ảnh hàng lỗi do AI tạo | docs/03 |
| **Ra quyết định** | chọn mô hình nào (MCDM), xử lý ca nào trước/chặn/chuyển người (triage) | 04 |
| **Decision model zero-shot (WaterSheep mở / Jev)** | không cần dữ liệu gán nhãn: xác suất đã hiệu chỉnh + hành động đề xuất | 05 |

Ngôn ngữ dữ liệu: **tiếng Anh**. Mã nguồn chạy được trên CPU; phần LLM chạy trên **Google Colab GPU**.

## Cấu trúc

```
src/drm/            thư viện dùng chung
  data.py           dataset bảng (UCI/Kaggle) – phụ lục
  text.py           dataset kiểm duyệt (Jigsaw, TweetEval, Davidson, HateXplain, SMS spam) + TF-IDF models + obfuscation
  fraud.py          ULB credit card, IEEE-CIS (Colab), bộ MÔ PHỎNG bùng hàng/chargeback, hook dữ liệu nội bộ
  benchmark.py      model zoo, biến thể cây (pruned/oblique/rotation forest), kiểm định Friedman/Nemenyi/Wilcoxon
  mcdm.py           AHP, Entropy, CRITIC, SAW, TOPSIS, VIKOR, PROMETHEE II, EDAS, MOORA, SMAA, rank reversal
  decision.py       mô hình ra quyết định triage: luật, Bayes chi phí kỳ vọng + năng lực review, scorecard MCDM
  llm.py            LLM/guard model/classifier chuyên dụng: chấm điểm, few-shot, calibration, LLM-as-AHP-expert
  jev.py            decision model "System One": client Jev /v1/systemone (cache), adapter WaterSheep (mở) và Jev-like LLM, câu hỏi kiểm duyệt/đơn hàng/AHP
notebooks/
  01_text_moderation_benchmark.ipynb      CPU
  02_fraud_chargeback_cod.ipynb           CPU
  03_llm_moderation_colab.ipynb           Colab GPU
  05_jev_zero_shot_decisions.ipynb        decision model zero-shot: WaterSheep (mở, miễn phí, mặc định) + Jev (nếu có credit)
  04_decision_models_cs_triage.ipynb      CPU
  appendix/A1, A2                         benchmark & MCDM trên 14 dataset UCI (phương pháp luận)
scripts/            chạy benchmark từ dòng lệnh
results/            kết quả (CSV, NPZ, hình) – đã chạy sẵn
docs/               tổng quan tài liệu, báo cáo, thiết kế pipeline hoá đơn giả
tests/              kiểm thử (MCDM đối chiếu pymcdm; dry-run LLM; chính sách quyết định)
```

## Chạy trên Google Colab (máy yếu vẫn dùng được)

Mỗi notebook có nút **Open in Colab** ở đầu, trỏ tới nhánh `main` (hoạt động sau khi merge các PR).
Repo đã **public** nên Colab mở và clone được ngay, không cần token.

1. *(Notebook 03, LLM)*: **Hugging Face token**: huggingface.co → Settings → Access Tokens → *Read*.
   Vào trang các mô hình gated (`meta-llama/Llama-Guard-3-1B`, `meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-3-1b-it`, `google/shieldgemma-2b`) và bấm *Agree / Request access*.
2. *(Tuỳ chọn, để đẩy kết quả từ Colab lên GitHub)*: **GitHub token**: Settings → Developer settings → *Fine-grained tokens*, chọn repo này, quyền **Contents: Read and write**.
3. Trong Colab: biểu tượng **🔑 Secrets** ở thanh trái → thêm `HF_TOKEN` (và `GH_TOKEN` nếu cần) → bật *Notebook access*.
   Notebook 05: mặc định chạy **WaterSheep** (mã nguồn mở, miễn phí, không cần key; nên bật GPU). Có credit Jev thì thêm `TYPESAFE_API_KEY` để so sánh thêm Jev.
4. Notebook 03: *Runtime → Change runtime type* → **A100** (lần chạy tham khảo: A100-SXM4-40GB, `BATCH = 32`) hoặc **L4** (`BATCH = 16`). T4 chạy được nhưng chậm.
   **Không chọn TPU**: code dùng PyTorch/CUDA, trên TPU notebook sẽ chỉ chạy chế độ thử (DRY_RUN). Notebook 05 (WaterSheep) chỉ cần L4/T4.
5. Ô đầu tiên tự clone repo, cài thư viện và thêm `src/` vào path. Ở ô cuối, đặt `PUSH_RESULTS = True` để commit thư mục `results/` lên nhánh **`colab-results`** (sau đó mở PR vào `main`).
   Nếu chỉ lưu notebook (*File → Save a copy in GitHub*), các file kết quả sẽ không đi kèm. Khi đó dùng `python scripts/extract_notebook_results.py` để trích bảng từ output đã lưu.

> Trước khi các PR được merge, mở notebook theo nhánh của PR, ví dụ:
> `https://colab.research.google.com/github/HieNguyen08/Research-decision-model-and-ML-classification/blob/research/docs-report/notebooks/03_llm_moderation_colab.ipynb`
> và sửa `BRANCH = "research/docs-report"` trong ô đầu tiên.

## Chạy local

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests -q              # kiểm thử
PYTHONPATH=src python scripts/run_text_benchmark.py   # benchmark kiểm duyệt (~20-30 phút, 4 CPU)
PYTHONPATH=src python scripts/run_fraud_benchmark.py  # benchmark gian lận
jupyter lab notebooks/
```

## Dùng dữ liệu nội bộ

```python
from drm import text as T, fraud as F
chat = T.load_internal_csv("cs_chat_labeled.csv", text_col="message", label_col="is_abusive")
orders = F.load_internal_orders("orders.csv", label_col="refused_delivery", amount_col="order_value", time_col="created_at")
```
Mọi hàm đánh giá, MCDM và chính sách quyết định đều nhận các đối tượng này mà không cần sửa code.

## Tài liệu
* [docs/02_bao_cao_ket_qua.md](docs/02_bao_cao_ket_qua.md): **báo cáo kết quả tổng hợp** (bắt đầu đọc từ đây)
* [docs/01_tong_quan_tai_lieu.md](docs/01_tong_quan_tai_lieu.md): lý thuyết mô hình ra quyết định, MCDM, cây quyết định, kiểm duyệt nội dung, LLM, gian lận, hoá đơn giả, khoảng trống nghiên cứu
* [docs/04_jev_system_one.md](docs/04_jev_system_one.md): **Jev & họ decision model "System One"**: API, bằng chứng, bản tái hiện mở, rủi ro, giả thuyết
* [docs/03_pipeline_hoa_don_gia.md](docs/03_pipeline_hoa_don_gia.md): thiết kế pipeline phát hiện hoá đơn/bằng chứng giả do AI tạo
