# Week 11 Demo Test Log - Hùng

## Mục tiêu

Kiểm tra demo từ đầu đến cuối để đảm bảo phần LSH có thể trình bày ổn định trước khi đưa vào báo cáo cuối.

## Môi trường kiểm tra

- Artifact directory: `jupyter/output/lsh`
- Metrics: `jupyter/output/lsh/metrics.json`
- Search index: `jupyter/output/lsh/search_index.pkl`
- Demo notebook: `jupyter/LSH_Demo.ipynb`
- Query notebook: `jupyter/LSH_Query_Demo.ipynb`

## Checklist demo

| Hạng mục | Trạng thái | Ghi chú |
| --- | --- | --- |
| Load metrics | Pass | Đọc được số liệu pipeline từ `metrics.json` |
| Load verified pairs | Pass | Có `31,875` verified near-duplicate pairs |
| Load clusters | Pass | Có `42,519` clusters |
| Top cluster | Pass | Cluster lớn nhất có `60` bài |
| Exact duplicate cases | Pass | Có nhiều cặp Jaccard = `1.0` |
| Near duplicate cases | Pass | Có ví dụ Jaccard từ `0.8` đến dưới `1.0` |
| Search index | Pass | Có `683,795` buckets cho `49,852` documents |
| Query demo | Pass | Query text có thể trả về top similar posts |
| Benchmark report | Pass | Có bảng brute-force vs LSH cho tuần 10 |

## Demo cases được chọn

Các case dùng cho demo nằm ở:

- `docs/hung_week9_12/week11_top_clusters.csv`
- `docs/hung_week9_12/week11_exact_duplicate_examples.csv`
- `docs/hung_week9_12/week11_near_duplicate_examples.csv`

## Kịch bản test demo

1. Mở `metrics.json` để xác nhận pipeline đã chạy xong.
2. Mở `week10_lsh_benchmark_report.md` để trình bày vì sao không brute-force toàn bộ dữ liệu.
3. Mở `week11_demo_cases.md` để chọn cluster và pair tiêu biểu.
4. Chạy query demo bằng một câu trong exact duplicate examples.
5. Kiểm tra kết quả trả về có `jaccard`, `cluster_id`, `cluster_size` và text gốc.

## Rủi ro còn lại

- Nếu chạy demo trực tiếp trên máy yếu, nên dùng artifact đã precompute thay vì rebuild toàn bộ pipeline.
- Nếu query quá ngắn, hệ thống có thể không tạo đủ shingles để search meaningful.
- Dữ liệu gốc có một số ký tự encoding lạ từ social media; khi đưa lên slide nên chọn các case hiển thị rõ ràng.

## Kết luận

Demo đủ điều kiện trình bày phần LSH:

- có số liệu tổng quan
- có benchmark
- có case exact duplicate
- có case near duplicate
- có cluster lớn
- có query search tương tự
