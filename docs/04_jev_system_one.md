# Jev và họ mô hình "System One": decision model zero-shot

> Ghi chú nghiên cứu, tháng 10/2026. Trang chủ `jevmodel.org` và tài liệu `docs.typesafe.ai` không truy cập được từ môi trường nghiên cứu.
> Giao thức API dưới đây được **đối chiếu trực tiếp từ mã nguồn SDK chính thức** `typesafe-sdk` 0.7.4 (PyPI, giấy phép MIT).
> Các số liệu khác lấy từ bài báo, blog đánh giá và README công khai, kèm mức độ tin cậy.
> Các tuyên bố của nhà phát triển hoặc cộng đồng chưa kiểm chứng độc lập đều được ghi rõ.

## 1. Jev là gì

| Thuộc tính | Nội dung | Nguồn / độ tin cậy |
|---|---|---|
| Nhà phát triển | TypeSafe AI | SDK chính thức |
| Phát hành | early access 15/09/2026 | báo chí công nghệ |
| Loại mô hình | **decision model phân biệt** (không sinh văn bản): *state* vào, **quyết định có kiểu + xác suất** ra. Nhà phát triển gọi là "System One" (tư duy nhanh, theo Kahneman), đối lập với LLM "System Two" (suy luận dài) | bài ra mắt; CEO xác nhận kiến trúc kiểu *encoder + các đầu ra* trên Hacker News (nguồn thứ cấp) |
| Huấn luyện | **RLCD**: Reinforcement Learning for Calibrated Decisions, tối ưu để xác suất khớp tần suất thật. Chi tiết kỹ thuật chưa công bố | bài ra mắt; Guo et al. (2026) |
| Hiệu năng công bố | 70–500 ms/lần gọi, một lượt song song cho mọi câu hỏi | nhà phát triển |
| Giá công bố | khoảng $0.042 / 1 triệu token đầu vào, đầu ra miễn phí (có nguồn cho rằng có thể đang được trợ giá) | nhà phát triển; review bên thứ ba |
| Truy cập | REST `POST https://api.typesafe.ai/v1/systemone`, mô hình `jev-latest`; SDK Python `typesafe-sdk`; nhà phát triển cũng có *System One Adapter* chính thức phục vụ cùng giao diện bằng OpenAI/Anthropic (baseline) | SDK chính thức; awesome-jev |

### Giao thức (từ SDK chính thức)

```jsonc
// POST /v1/systemone   Authorization: Bearer $TYPESAFE_API_KEY
{
  "state": {"customer_message": "This is the 3rd time I ask. You people are useless."},
  "model": "jev-latest",
  "questions": {
    "toxic":    {"type": "noul",   "instructions": "Is this message toxic?"},
    "severity": {"type": "score",  "instructions": "How severe for a CS agent?",
                 "criteria": ["harmless", "impolite", "abusive", "threatening"]},
    "action":   {"type": "choice", "instructions": "What should CS do?",
                 "criteria": {"reply_normally": null, "reply_with_deescalation": null,
                              "escalate_to_supervisor": null, "block_and_report": null}}
  }
}
// -> {"model": "...", "usage": {"input_tokens": .., "output_tokens": ..},
//     "answers": {"toxic":    {"type": "noul", "noul": 0.71},
//                 "severity": {"type": "score", "score": 1.4, "confidence": .., "probabilities": {"0": .., ...}, "legend": {...}},
//                 "action":   {"type": "choice", "choice": "reply_with_deescalation", "confidence": .., "probabilities": {...}}}}
```

| Primitive | Câu hỏi | Trả về |
|---|---|---|
| **Noul** | có/không (có thể mô tả nghĩa của "true"/"false") | `noul` = P(yes) |
| **Choice** | chọn 1 trong các nhãn (có mô tả tuỳ chọn) | `choice`, `confidence`, `probabilities` theo nhãn |
| **Score** | thang bậc có thứ tự (rubric 0..k) | `score` = kỳ vọng bậc, `confidence`, `probabilities`, `legend` |

## 2. Vì sao Jev quan trọng cho đề tài này

Jev đúng là một **"mô hình ra quyết định"**. Nó khớp với cả hai tầng của khung nghiên cứu:

| Tầng của đề tài | Jev đóng góp | Thí nghiệm |
|---|---|---|
| Dự báo (nb 01–02) | xác suất **zero-shot** cho toxic/hate/spam, bùng hàng/chargeback từ JSON đơn hàng, **không cần dữ liệu gán nhãn**, giải quyết cold-start | nb 05: E1, E5 |
| Hiệu chỉnh | RLCD hứa hẹn xác suất đã hiệu chỉnh, đúng điều kiện cần của quyết định Bayes (nb 04) | nb 05: E2 |
| Quyết định vận hành (nb 04) | (a) cấp xác suất cho chính sách Bayes; (b) **tự chọn hành động** bằng `choice` | nb 05: E6 |
| Quyết định chiến lược / MCDM | sinh ma trận AHP (mọi cặp tiêu chí trong **một lần gọi**, đọc phân phối xác suất trên thang Saaty) | nb 05: E7 |
| Hoá đơn giả (docs/03) | Noul trên JSON các trường OCR + dữ kiện đơn hàng ("số tiền có khớp đơn không?") | đề xuất |

## 3. Bằng chứng hiện có

### 3.1 Nghiên cứu có phương pháp rõ ràng
* **Guo et al. (2026), "Just Ask Jev"** (arXiv 2609.29429; code và dữ liệu: `sumleo/RLCDAlignBench`):
  * Dùng Jev làm bộ phát hiện **zero-shot** lỗi alignment của LLM trên **44 benchmark**, 10 loại lỗi, 7 193 mẫu.
  * Chỉ một câu hỏi chung đạt **AUROC trung vị 0.886** và **thắng baseline TF-IDF / độ dài có giám sát ở 25/31 benchmark**.
  * Cách diễn đạt câu hỏi ảnh hưởng ít. Nhưng **đọc kết quả dưới dạng xác suất thay vì argmax quan trọng hơn nhiều**. Ngữ cảnh giúp chủ yếu qua các trường "rò nhãn".
  * Mức đồng thuận với nhãn người tương đương bộ chấm tham chiếu, và rẻ hơn LLM-judge khoảng **63 lần**.
* **Cisco, "Jev: Decision Models on Trial"** (blog kỹ thuật): 24 nhóm nội dung hại, 6 dataset, ngân sách FPR 0.5%.
  * **Encoder 1B được huấn luyện riêng theo chính sách vẫn cho recall tốt nhất ở FPR nghiêm ngặt trên mọi dataset.**
  * Jev zero-shot tiệm cận một LLM-judge 31B.
* **Ling, Xue & Ye (2026), "Jev in the Wild"** (arXiv 2609.30216): phân tích 2 170 dự án GitHub dùng Jev.
  * Ứng dụng phổ biến nhất: đánh giá thuộc tính và chấm điểm. Sau đó là lọc nội dung, định tuyến, chọn mô hình/công cụ.
  * Jev đóng vai **thành phần quyết định tái sử dụng** trong workflow.

### 3.2 Đánh giá cộng đồng (chưa kiểm chứng, tổng hợp từ danh sách awesome-jev)
| Bài toán | Kết quả báo cáo | Hàm ý cho CS |
|---|---|---|
| Spam email (18 514 email), Noul zero-shot | accuracy 0.983, ngang TF-IDF huấn luyện trên 14 800 nhãn | spam: zero-shot có thể đủ |
| Prompt injection (662 tin) | ROC-AUC 0.993, ECE 0.059, p50 325 ms | bảo vệ chatbot CS |
| Moderation (sysone-bench) | 98.9% (Jev) so với 83.3% (Laya, bản mở) | |
| ChaosNLI (mẫu người gán nhãn bất đồng) | độ tự tin trung bình 0.807 so với mức đồng thuận người 0.468 → **quá tự tin khi mơ hồ** | ca mơ hồ nên chuyển người |
| JevAdvBench (arXiv 2609.31142) | tấn công hộp đen dịch chuyển được câu trả lời có kiểu | người dùng cố tình né bộ lọc |
| Poker probe | đổi tên nhãn làm kết quả dao động 15–30 điểm | cố định cách đặt nhãn; kiểm tra độ nhạy |
| Hỏi trực tiếp 1 câu vs chấm 12–14 chiều rồi gộp | 1 câu: dương tính giả trên ca "lành nhưng khó" cao gấp 25 lần (37.2% vs 1.5%) | tách quyết định thành nhiều câu hỏi đặc trưng |

* **Lỗi đã được nhà phát triển thừa nhận** (Jev 1.13 "jaggedness"): đọc quá sát chữ, đếm, ngày tháng, quan hệ gián tiếp, state chứa thông tin gây nhiễu, nội dung đối kháng.

## 4. "Họ" Jev: các bản tái hiện mở

Cộng đồng nhanh chóng tái hiện "hình dạng" System One (state + N lựa chọn → một xác suất cho mỗi lựa chọn, một lượt chạy). Có ba hướng:

| Hướng | Ví dụ (tự báo cáo, chưa kiểm chứng) | Ghi chú |
|---|---|---|
| Encoder + đầu quyết định huấn luyện kiểu RLCD | Laya (ModernBERT-large, khoảng 38 ms), rlcd-modernbert-151m, sokudan (ModernBERT-ja), Phocinae-Largha-150M | nhanh, chạy CPU/GPU nhỏ |
| Decoder nhỏ + LoRA + đầu vô hướng | Open-Jev (Qwen3.5 2B/9B), kev (Qwen 0.5–8B), Bespoke Nimble (Qwen3.5-9B), decider, NanoJev (0.6B), LFM2.5-RLCD | có công thức huấn luyện mở |
| **Không huấn luyện**: đọc logit nhãn của LLM có sẵn | LitJev, Simple Jev, open-jev (Gemma 3 4B), Jev-on-a-laptop | **cùng kỹ thuật với `HFScorer`/`LocalJevAdapter` của repo này** |
| Server tương thích `/v1/systemone` | WaterSheep, Jebadiah, LocalJev, openjev-sglang | dùng được với `JevClient(base_url=...)` |

→ Trong notebook 05, **`LocalJevAdapter`** (Qwen2.5-1.5B đọc log-likelihood nhãn, *không* có RLCD) là baseline để tách riêng giá trị của huấn luyện RLCD.

## 4b. Khi không có credit Jev: WaterSheep (mở, miễn phí)

API của Jev **tính phí theo token** và yêu cầu nạp credit trước. Vercel AI Gateway và OpenRouter cũng tính phí; không tìm thấy gói miễn phí chính thức nào.
Repo vì vậy dùng **WaterSheep** làm backend mặc định cho notebook 05:

| | WaterSheep |
|---|---|
| Kiến trúc | encoder **ModernBERT-base** + đầu quyết định, hiệu chỉnh xác suất bằng temperature theo loại câu hỏi |
| Giao diện | `noul` / `choice` / `score` (thêm `multi`), cùng định dạng request/response với Jev; có server `POST /v1/systemone` dùng được với SDK của TypeSafe |
| Mã nguồn | [SamratDuttaOfficial/WaterSheep](https://github.com/SamratDuttaOfficial/WaterSheep) (Apache-2.0); trọng số `samratduttaofficial/WaterSheep` trên Hugging Face |
| Kết quả tác giả tự báo cáo | accuracy 77.8% (ECE 0.026) trong phân phối huấn luyện; 61.2% (ECE 0.043) trên dataset chưa thấy; HateCheck 75.1% (ECE 0.139) |
| Kiểm chứng trong repo | mọi bộ câu hỏi (kiểm duyệt, bùng hàng/chargeback, AHP 9 bậc) đã chạy qua **mã kiểm tra hợp lệ thật** của WaterSheep, chỉ thay phần mô hình (không tải được trọng số từ môi trường nghiên cứu) |

Cách dùng: `jev.WaterSheepAdapter()` trong Python, hoặc chạy server `watersheep --model samratduttaofficial/WaterSheep --serve`
rồi dùng `jev.JevClient(api_key="local", base_url="http://127.0.0.1:8766")`.
**Lưu ý:** WaterSheep là bản tái hiện độc lập, *không* phải Jev. Kết luận rút ra từ nó là về **lớp mô hình "System One" mở**. Khi có credit, notebook chạy cả hai để so sánh trực tiếp.

## 5. Rủi ro khi đưa vào CS e-commerce

1. **Quyền riêng tư:** tin nhắn và đơn hàng của khách bị gửi sang API bên thứ ba. Cần thoả thuận xử lý dữ liệu (DPA), che PII (SĐT, địa chỉ, số thẻ) trước khi gửi. Nếu không được phép thì cân nhắc bản tái hiện mở chạy nội bộ.
2. **FPR nghiêm ngặt:** khi chặn nhầm khách rất đắt, mô hình **huấn luyện theo chính sách nội bộ** vẫn thắng (Cisco). Nên dùng Jev cho tầng ưu tiên/triage, không dùng làm bộ chặn cuối.
3. **Quá tự tin khi mơ hồ** (ChaosNLI) và **nhạy với cách đặt nhãn**: giữ cố định bộ câu hỏi, hiệu chỉnh lại bằng một ít nhãn nội bộ (E2), chuyển ca mơ hồ cho người.
4. **Tấn công đối kháng**: kết hợp với chuẩn hoá ngược (nb 01) và mô hình huấn luyện riêng.
5. **Phụ thuộc nhà cung cấp và giá/độ trễ do nhà phát triển công bố**: luôn đo lại (nb 05 ghi lại token và độ trễ thực tế); có phương án dự phòng qua server tương thích.

## 6. Giả thuyết kiểm định trong notebook 05 và kết quả

Đã chạy trên Colab với **WaterSheep** (bản mở cùng giao thức) và baseline **Jev-like Qwen2.5-1.5B**. **Jev thật chưa chạy** vì cần credit.
Kết quả dưới đây là của WaterSheep, chưa phải của Jev. Số liệu: `results/jev/`; phân tích chi tiết: kết luận của notebook 05 và docs/02 mục 2.6.

| # | Giả thuyết | Đo bằng | Kết quả với WaterSheep |
|---|---|---|---|
| H0 | Một decision model **mở, miễn phí** (WaterSheep) đủ dùng cho tầng triage, so với TF-IDF đã huấn luyện | E1, E6 | **Chưa đủ nếu đứng một mình.** Dùng được cho tin nhắn (sau khi hiệu chỉnh), không dùng được cho đơn hàng; chi phí triage 1746 so với 1090 |
| H1 | Jev zero-shot đạt ≥ 90% PR-AUC của TF-IDF *đã huấn luyện* với toxic/offensive/spam, nhưng thấp hơn rõ với hate (Davidson) | E1 | **Đúng:** Jigsaw 98%, offensive 118%, spam 102%; Davidson 23% |
| H2 | Jev có ECE thấp hơn LLM mở không RLCD; vài chục nhãn đủ để hiệu chỉnh lại | E2 | **Vế 1 sai** (ECE 0.269 so với 0.230 của Jev-like Qwen; quá tự tin ở vùng 0.8–0.97). **Vế 2 đúng** (50 nhãn/task: ECE 0.053) |
| H3 | Obfuscation làm Jev mất ít PR-AUC hơn TF-IDF | E1/E3 | **Đúng:** −0.9%, so với −4.5% |
| H4 | Kết hợp Jev + TF-IDF tốt hơn từng mô hình | E4 | **Đúng, mức vừa:** 0.773, so với 0.756 / 0.688 (w = 0.25) |
| H5 | Prior Jev giúp LogReg khi chỉ có k ≤ 64 nhãn (cold-start bùng hàng) | E5 | **Sai:** WaterSheep trên JSON đơn hàng có ROC-AUC < 0.5; prior thêm ≤ 0.004. Cần thử prior của Jev-like Qwen (ROC-AUC 0.71–0.73) |
| H6 | Bayes·Jev có chi phí gần Bayes·TF-IDF; `action` trực tiếp của Jev kém hơn Bayes vì không biết chi phí | E6 | **Vế 1 sai** (+60%, do phần đơn hàng và xác suất chưa hiệu chỉnh). **Vế 2 đúng** (`action` 1891 là kém nhất) |
| H7 | Ma trận AHP do Jev sinh nhất quán (CR < 0.1) | E7 | **Đúng về hình thức** (CR ≈ 0.01), nhưng trọng số gần đều và gần như không đổi giữa hai kịch bản → nhất quán vì "thờ ơ", không phải vì phán đoán tốt |

**Ý nghĩa cho họ Jev:**
* Giao thức System One (câu hỏi Noul/Choice/Score → xác suất) dùng tốt cho **văn bản ngắn**: một mô hình 150M tham số đã ngang các LLM 3–4B zero-shot.
* Lời hứa "xác suất đã hiệu chỉnh" **không tự động đúng** với bản tái hiện mở. Cần đo lại ECE trên Jev thật trước khi đưa xác suất vào chính sách Bayes.
* Với **dữ liệu dạng bảng** (đơn hàng), cần thận trọng: phải kiểm tra chiều của tín hiệu trên một mẫu có nhãn trước khi dùng.

## 7. Hướng cải tiến đề xuất

* **Jev làm bộ trích xuất đặc trưng đã hiệu chỉnh, quyết định giao cho tầng Bayes chi phí** (nb 04), thay vì để Jev chọn hành động trực tiếp.
* **Zero-shot + hiệu chỉnh với ít nhãn** (20–50 nhãn/nhóm) cho chat CS nội bộ.
* **Nhiều câu hỏi đặc trưng trong một lần gọi** (toxic, spam, severity, đe doạ, PII...) rồi học trọng số gộp bằng ít nhãn. Cách này giảm dương tính giả so với hỏi trực tiếp một câu (bằng chứng cộng đồng).
* **Cổng con người theo độ tự tin** (selective classification) với `confidence` của Choice/Score.
* **Bản mở chạy nội bộ** (Laya / tái hiện decoder + LoRA) fine-tune bằng *proper scoring rule* (log-loss/Brier) trên dữ liệu CS, khi không thể gửi dữ liệu ra ngoài.

## Nguồn

* SDK chính thức: [typesafe-sdk trên PyPI](https://pypi.org/project/typesafe-sdk/) · [typesafe-ai/typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python)
* Guo et al. (2026). *Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures.* [arXiv:2609.29429](https://arxiv.org/abs/2609.29429) · [RLCDAlignBench](https://github.com/sumleo/RLCDAlignBench)
* Ling, Xue & Ye (2026). *Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem.* [arXiv:2609.30216](https://arxiv.org/abs/2609.30216)
* [Cisco – Jev: Decision Models on Trial](https://blogs.cisco.com/ai/jev-decision-models-on-trial)
* [Red Hat – Benchmarking AI decision models against traditional guardrails](https://developers.redhat.com/articles/2026/10/02/benchmarking-ai-decision-models-against-traditional-guardrails)
* JevAdvBench: [arXiv:2609.31142](https://arxiv.org/abs/2609.31142)
* Danh sách cộng đồng: [omnijev/awesome-jev](https://github.com/omnijev/awesome-jev) · [kraayenjon/awesome-jev](https://github.com/kraayenjon/awesome-jev)
* Tài liệu chính thức (không truy cập được từ môi trường nghiên cứu): [docs.typesafe.ai](https://docs.typesafe.ai/) · [jevmodel.org](https://jevmodel.org/)
