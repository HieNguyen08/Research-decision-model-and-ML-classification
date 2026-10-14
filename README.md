# Mô hình ra quyết định & phân loại ML/LLM cho Trust & Safety trong CS e-commerce

Nghiên cứu tổng hợp nhằm **phát hiện** và **ra quyết định xử lý** các tình huống rủi ro mà đội Customer Service gặp hằng ngày:

| Bài toán | Ví dụ | Notebook |
|---|---|---|
| Ngôn từ độc hại / công kích / thù ghét | khách chửi bới, đe doạ nhân viên CS | 01, 03 |
| Spam / lừa đảo | tin nhắn trúng thưởng, dẫn link thanh toán ngoài sàn | 01, 03 |
| Gian lận thanh toán & **chargeback** | thẻ ăn cắp, nhận hàng rồi đòi hoàn tiền (friendly fraud) | 02 |
| **Bùng hàng COD** | đặt rồi không nhận, "boom hàng" | 02 |
| Hoá đơn / bằng chứng giả do AI | biên lai chuyển khoản giả, ảnh hàng lỗi do AI tạo | docs/03 |
| **Ra quyết định** | chọn mô hình nào (MCDM), xử lý ca nào trước/chặn/chuyển người (triage) | 04 |

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
notebooks/
  01_text_moderation_benchmark.ipynb      CPU
  02_fraud_chargeback_cod.ipynb           CPU
  03_llm_moderation_colab.ipynb           Colab GPU
  04_decision_models_cs_triage.ipynb      CPU
  appendix/A1, A2                         benchmark & MCDM trên 14 dataset UCI (phương pháp luận)
scripts/            chạy benchmark từ dòng lệnh
results/            kết quả (CSV, NPZ, hình) – đã chạy sẵn
docs/               tổng quan tài liệu, báo cáo, thiết kế pipeline hoá đơn giả
tests/              kiểm thử (MCDM đối chiếu pymcdm; dry-run LLM; chính sách quyết định)
```

## Chạy trên Google Colab (máy yếu vẫn dùng được)

Mỗi notebook có nút **Open in Colab** ở đầu. Repo đang để **private** nên cần cấu hình một lần:

1. **GitHub token**: GitHub → Settings → Developer settings → *Fine-grained tokens* → Generate.
   Chọn *Only select repositories* (repo này) và quyền **Contents: Read and write**.
2. **Hugging Face token** (cho notebook 03): huggingface.co → Settings → Access Tokens → *Read*.
   Vào trang các mô hình gated (`meta-llama/Llama-Guard-3-1B`, `meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-3-1b-it`, `google/shieldgemma-2b`) và bấm *Agree / Request access*.
3. Trong Colab: biểu tượng **🔑 Secrets** ở thanh trái → thêm `GH_TOKEN` và `HF_TOKEN` → bật *Notebook access*.
4. Lần đầu mở notebook từ repo private: Colab → *File → Open notebook → GitHub* → tick *Include private repos* → cấp quyền.
5. Notebook 03: *Runtime → Change runtime type → T4 GPU*.
6. Ô đầu tiên của mỗi notebook tự clone repo, cài thư viện và thêm `src/` vào path. Ô cuối có `PUSH_RESULTS = True` để **commit kết quả ngược lên GitHub** (điền tên/email git của bạn).

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
* [docs/01_tong_quan_tai_lieu.md](docs/01_tong_quan_tai_lieu.md): lý thuyết mô hình ra quyết định, MCDM, cây quyết định, kiểm duyệt nội dung, LLM, gian lận, hoá đơn giả, khoảng trống nghiên cứu
* [docs/03_pipeline_hoa_don_gia.md](docs/03_pipeline_hoa_don_gia.md): thiết kế pipeline phát hiện hoá đơn/bằng chứng giả do AI tạo
