# Tại sao vẫn cần làm tuần 9-12 dù đã có kết quả tuần 1-8

## Tóm tắt nhanh

Tuần 1-8 chủ yếu trả lời câu hỏi "có chạy được pipeline hay không".  
Tuần 9-12 mới trả lời câu hỏi "pipeline có đáng tin, có scale được, và có đủ sẵn sàng để báo cáo/demo hay không".

Nói cách khác:
- Tuần 1-8 = xây nền tảng kỹ thuật.
- Tuần 9-12 = kiểm chứng chất lượng, trình bày kết quả, và chốt sản phẩm cuối.

## Giới hạn nếu dừng ở tuần 1-8

Nếu chỉ dừng ở tuần 1-8, nhóm mới có "kết quả ban đầu", nhưng còn thiếu nhiều điểm quan trọng:

- Chưa có hình ảnh tổng hợp cho phân bố similarity để giải thích duplicate pattern.
- Chưa có visualization hiệu năng để chứng minh lý do chọn LSH thay vì brute-force.
- Chưa có bộ tài liệu trình bày (slide structure, chart/table mapping) để demo mạch lạc.
- Chưa có phần report dataset + kết quả đủ chuẩn để nộp final.
- Chưa có checklist thực hành demo để giảm rủi ro lỗi khi trình bày.

## Giá trị riêng của tuần 9-12

### Tuần 9: Chuyển từ "có cặp duplicate" sang "hiểu cấu trúc duplicate"

- Vẽ histogram similarity để thấy phân bố Jaccard.
- Vẽ cluster graph để minh họa cụm nội dung lặp.
- Mục tiêu: giúp người xem hiểu bản chất dữ liệu, không chỉ nhìn bảng số.

### Tuần 10: Chuyển từ "dùng được" sang "dùng được khi scale"

- So sánh runtime và số pair giữa brute-force và MinHash+LSH.
- Minh họa candidate reduction ratio.
- Mục tiêu: có bằng chứng kỹ thuật để bảo vệ kiến trúc đã chọn.

### Tuần 11: Chuyển từ "kết quả kỹ thuật" sang "nội dung trình bày"

- Gom chart/table vào khung slide có trình tự.
- Chọn case exact/near duplicate để demo.
- Mục tiêu: biến output kỹ thuật thành câu chuyện để hội đồng dễ theo dõi.

### Tuần 12: Chuyển từ "demo tạm" sang "sản phẩm final"

- Viết phần dataset + kết quả theo format report.
- Chốt insight, hạn chế, kiến nghị.
- Lập checklist practice demo và Q&A.
- Mục tiêu: đảm bảo nộp đủ, thuyết phục, và trình bày ổn định.

## Nếu bỏ qua tuần 9-12 sẽ gặp gì?

- Dễ bị hỏi "tại sao chọn LSH?" mà không có bằng chứng hiệu năng rõ ràng.
- Dễ bị đánh giá "có code nhưng chưa có phân tích kết quả".
- Slide/demo dễ rơi vào trạng thái mạnh ai nấy nói, thiếu logic liên kết.
- Báo cáo final thiếu phần tổng hợp và dễ bị xem là chưa hoàn tất vòng đời đề tài.

## Kết luận

Làm tuần 9-12 không phải lặp lại việc đã làm ở tuần 1-8.  
Đó là bước bắt buộc để nâng cấp từ:

- prototype -> evidence-backed solution
- kết quả kỹ thuật -> kết quả có thể báo cáo/demo
- "chạy được" -> "chạy tốt, giải thích được, bảo vệ được"

Vì vậy, dù đã có output tuần 1-8, vẫn cần hoàn thành tuần 9-12 để đề tài đạt mức "hoàn chỉnh" thay vì chỉ "có tính năng cơ bản".
