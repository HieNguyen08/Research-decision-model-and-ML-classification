# Pipeline phát hiện hoá đơn / bằng chứng giả do AI tạo (thiết kế)

> Phạm vi đã chốt: **tổng quan + thiết kế pipeline**, chưa huấn luyện mô hình ảnh. Mục tiêu là đưa ra một
> **điểm rủi ro tài liệu** `p_doc_forged` để cắm vào mô hình ra quyết định triage (notebook 04), cùng với
> `p_toxic`, `p_spam` và `p_risky`.

## 1. Bối cảnh trong CS e-commerce

Các tình huống thường gặp khi khách hoặc người bán gửi "bằng chứng" để yêu cầu hoàn tiền, đổi trả hoặc khiếu nại:

| Tình huống | Bằng chứng giả điển hình |
|---|---|
| Đòi hoàn tiền / chargeback | Ảnh chụp **biên lai chuyển khoản** giả, **hoá đơn** sửa số tiền |
| Khiếu nại hàng lỗi | Ảnh **sản phẩm hỏng** do AI tạo hoặc chỉnh sửa, ảnh lấy từ internet |
| Tranh chấp giao hàng | Ảnh **vận đơn / phiếu giao** giả, ảnh chụp màn hình tracking bị sửa |
| Người bán gian lận | Hoá đơn VAT / chứng từ nhập hàng giả để mở khoá tiền |

Theo báo cáo ngành năm 2025, hoá đơn do AI tạo **tăng rất nhanh**:
* SAP Concur phát hiện số hoá đơn nghi do AI tạo tăng 18 lần sau khi triển khai bộ kiểm tra, với khoảng 1% hoá đơn bị gắn cờ.
* AppZen ghi nhận hoá đơn AI chiếm khoảng 14% tài liệu gian lận vào tháng 9/2025.

Chi phí để làm giả gần như bằng 0, nên không thể dựa vào việc nhân viên nhìn bằng mắt.

## 2. Kiến trúc nhiều lớp (defense in depth)

```
 ảnh/PDF ──► [L0 Tiền xử lý] ──► [L1 Nguồn gốc & metadata] ──┐
                     │                                        │
                     ├──► [L2 OCR + trích xuất trường] ──► [L3 Đối chiếu dữ liệu nội bộ] ──┤
                     │                                        │
                     └──► [L4 Pháp y ảnh/tài liệu] ──► [L5 Phát hiện ảnh do AI sinh] ──────┤
                                                              ▼
                                         [L6 Tổng hợp rủi ro: p_doc_forged + lý do]
                                                              ▼
                                    Mô hình ra quyết định triage (auto / review / restrict)
```

| Lớp | Kiểm tra | Kỹ thuật / công cụ | Tín hiệu ra |
|---|---|---|---|
| L0 | Chuẩn hoá ảnh, chống trùng lặp | perceptual hash (pHash), so khớp ảnh đã từng nộp | `dup_score` |
| L1 | Nguồn gốc | EXIF (thiết bị, thời gian, phần mềm), **C2PA content credentials**; ảnh mất toàn bộ metadata hoặc có tag phần mềm sinh ảnh là dấu hiệu | `meta_flags` |
| L2 | Đọc nội dung | OCR (Tesseract/PaddleOCR) hoặc **VLM** (Qwen2.5-VL, Florence-2) trích xuất: mã đơn, số tiền, ngày, tên ngân hàng/người bán, mã giao dịch | JSON các trường |
| L3 | **Nhất quán nghiệp vụ** (lớp mạnh nhất, rẻ nhất) | so với DB: mã đơn tồn tại? số tiền khớp? ngày hợp lệ (không trước ngày đặt, không phải tương lai)? tổng = Σ dòng + thuế? định dạng mã giao dịch của ngân hàng? | `n_violations`, danh sách lý do |
| L4 | Pháp y | Error Level Analysis, phát hiện copy-move/splicing, font và kerning không đồng nhất, nền in trơn bất thường | `forensic_score` |
| L5 | Ảnh do AI sinh | bộ phát hiện tổng quát hoá (Wang et al., 2020; Ojha et al., 2023); cần **cập nhật liên tục** theo bộ sinh mới | `p_ai_generated` |
| L6 | Tổng hợp | **cây quyết định đã cắt tỉa** hoặc logistic trên các tín hiệu trên (dễ giải thích cho nhân viên), hiệu chỉnh xác suất | `p_doc_forged` + top lý do |

**Nguyên tắc:** ưu tiên L3. Một hoá đơn "đẹp" về hình ảnh vẫn thất bại nếu số tiền hoặc mã giao dịch không khớp dữ liệu thật. Kẻ gian không truy cập được DB nội bộ, nên lớp này khó qua mặt nhất.

## 3. Ví dụ luật nhất quán (L3)

```python
def invoice_checks(doc: dict, order: dict) -> list[str]:
    """doc: các trường trích từ ảnh; order: bản ghi đơn hàng trong DB. Trả về danh sách vi phạm."""
    v = []
    if doc.get("order_id") != order["order_id"]:
        v.append("mã đơn không khớp")
    if abs(doc.get("total", 0) - order["paid_amount"]) > 0.01 * order["paid_amount"]:
        v.append("số tiền lệch >1%")
    if doc.get("date") and not (order["created_at"] <= doc["date"] <= order["now"]):
        v.append("ngày ngoài khoảng hợp lệ")
    lines = doc.get("line_items", [])
    if lines and abs(sum(l["qty"] * l["price"] for l in lines) + doc.get("tax", 0) - doc.get("total", 0)) > 0.02:
        v.append("tổng ≠ Σ dòng + thuế")
    if doc.get("txn_id") and not BANK_TXN_PATTERNS.get(doc.get("bank"), r".*").match(doc["txn_id"]):
        v.append("mã giao dịch sai định dạng ngân hàng")
    return v
```

## 4. Dữ liệu & đánh giá đề xuất

* **Dữ liệu nội bộ:** các yêu cầu hoàn tiền có ảnh, kèm kết luận cuối cùng của chuyên viên (giả / thật). Đây là nhãn tốt nhất.
* **Dữ liệu bổ sung:** dataset forgery biên nhận công khai (tìm theo từ khoá *receipt forgery detection dataset*, ICDAR); tự sinh hoá đơn giả bằng chính các mô hình sinh ảnh (red-teaming) để huấn luyện L5.
* **Metric:** recall @ FPR thấp (không được làm phiền khách thật), độ trễ, tỉ lệ ca phải chuyển người, và **chi phí kỳ vọng** (giống notebook 02).
* **Theo dõi drift:** bộ sinh ảnh thay đổi hằng tháng, nên phải giám sát phân phối `p_ai_generated` và tỉ lệ bị chuyên viên lật lại.

## 5. Tích hợp vào mô hình ra quyết định

`p_doc_forged` trở thành một tiêu chí hoặc một xác suất nữa trong `CostModel` (notebook 04):
chi phí bỏ sót bằng số tiền hoàn sai cộng phí chargeback; chi phí báo động nhầm bằng thời gian review cộng trải nghiệm khách.
Chính sách Bayes dùng nguyên khung công thức hiện có, chỉ thêm một thành phần rủi ro.
