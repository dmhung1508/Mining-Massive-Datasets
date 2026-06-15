# Hướng dẫn chạy bản tin AI (Runbook)

## TL;DR — mỗi lần bật máy

Hai bước:

```bash
# B1 (1 lần, chuẩn bị sẵn kịch bản + ảnh — phần chậm làm offline)
python scripts/media/pregen_broadcast.py --artifact-dir jupyter/output/lsh_combined --top-n 5

# B2 (chạy server)
python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_combined --port 8765
```

Mở **http://localhost:8765/** → bấm **Bắt đầu**. Vì ảnh + kịch bản đã chuẩn bị sẵn,
bản tin vào gần như tức thì (không phải chờ gen ảnh ~40s mỗi tấm).

> Nếu lười chờ ảnh, bỏ qua B1 và chỉ chạy B2 — bản tin vẫn nói (Grok ~3s), chỉ là
> không có ảnh minh họa. Hoặc B1 thêm `--no-images` để chỉ dựng kịch bản.

Kịch bản + trạng thái ảnh được lưu ở `jupyter/output/news/metadata/`; ảnh được
lưu theo cache id trong `jupyter/output/news/images/<cache-id>/`. Nếu artifact
parquet đổi, cache tự hết hiệu lực và lần chạy kế tiếp sẽ dựng lại metadata.

---

## 0. Cài đặt (chỉ khi dựng máy mới / mất package)

```bash
pip install -e ".[dev,api,dashboard,media,telegram]"
```

Kiểm tra `.env`:
- `API_VEO` — sinh ảnh/video (đã có sẵn).
- `API_KEY` — key xAI Grok để **phân tích tin bằng tiếng Việt**. Trống thì bản tin
  đọc câu chung chung.
- `NEWS_MONGO_URI` — chỉ cần khi muốn tin X realtime (mục 3).

---

## 1. CÁCH NHANH NHẤT — xem bản tin ngay (dùng cụm có sẵn)

Bạn đã có sẵn cụm tin trong `jupyter/output/lsh_combined` và `jupyter/output/lsh_full`.
Chỉ cần chạy API trỏ vào một trong hai:

```bash
# Nhẹ, mở nhanh (vài nghìn bài) — khuyên dùng cho máy yếu
python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_combined --port 8765
```

hoặc

```bash
# Toàn bộ corpus 11M bài (lần đọc đầu ~16 giây, sau đó cache nhanh)
python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_full --port 8765
```

Rồi mở trình duyệt:

```
http://localhost:8765/
```

Nhấn **Bắt đầu** → quả địa cầu quay 3s → vào bản tin.

> Lỗi "Run the LSH pipeline first" nghĩa là artifact dir đang trỏ tới thư mục
> chưa có `clusters.parquet`. Đổi `--artifact-dir` sang `lsh_combined` hoặc
> `lsh_full` là hết.

Khi bật server hoặc bấm **Bắt đầu**, hệ thống sẽ preflight trước: kiểm tra số
dòng parquet, dung lượng `scale_shingles`, RAM/disk còn trống và cache metadata.
Nếu báo `danger` thì nên dừng, đổi sang `lsh_combined`, hoặc đóng app nặng trước.
Với `lsh_full`, lần đầu cache miss có thể lâu vì phải quét parquet lớn; các lần
sau sẽ đọc metadata đã lưu và không gọi lại Grok/ảnh.

---

## 2. (Tùy chọn) Sinh ảnh minh họa cho từng cụm

Ảnh hiện lên khung "HÌNH ẢNH MINH HỌA" trong lúc đọc tin. Chạy trước khi mở bản tin:

```bash
# B1: cụm -> news objects (tiếng Việt nếu có API_KEY Grok)
python scripts/media/build_news_objects.py --artifact-dir jupyter/output/lsh_combined --top-n 5

# B2: news objects -> ảnh (gpt-image qua API_VEO)
python scripts/media/generate_images.py --quality low
```

Ảnh lưu ở `jupyter/output/news/images/`. Sau đó chạy lại mục 1.

---

## 3. (Tùy chọn) Lấy tin MỚI NHẤT realtime từ Telegram + X

Chỉ cần khi muốn cụm tin cập nhật theo thời gian thực (không phải corpus cũ).

Điều kiện: điền connection string MongoDB chứa `news_monitoring` vào `.env`:

```
NEWS_MONGO_URI=mongodb+srv://<user>:<pass>@<cluster-cua-X>/
MONGO_URI=mongodb+srv://<user>:<pass>@<cluster-telegram>/
```

Rồi chạy:

```bash
# Lấy bài 2 ngày gần nhất -> chạy LSH -> ra cụm mới ở jupyter/output/lsh_latest
python scripts/pipeline/refresh_latest_clusters.py --since-days 2 --scale-size 50000

# Phát bản tin từ cụm mới
python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_latest --port 8765
```

Muốn cập nhật liên tục: đặt lệnh `refresh_latest_clusters.py` vào cron (ví dụ mỗi giờ).

---

## 4. Chạy lại pipeline LSH từ đầu (nếu cần dựng lại cụm)

Chỉ làm khi muốn build lại từ dataset, không phải để xem bản tin.

```bash
python scripts/pipeline/extract_subsets.py    --artifact-dir jupyter/output/lsh_combined
python scripts/pipeline/build_shingles.py     --artifact-dir jupyter/output/lsh_combined
python scripts/pipeline/run_baseline.py       --artifact-dir jupyter/output/lsh_combined
python scripts/pipeline/run_lsh.py            --artifact-dir jupyter/output/lsh_combined
python scripts/pipeline/verify_and_cluster.py --artifact-dir jupyter/output/lsh_combined
```

---

## Tóm tắt nhanh

| Muốn gì | Lệnh |
|---|---|
| Xem bản tin ngay | `python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_combined --port 8765` rồi mở `http://localhost:8765/` |
| Có ảnh minh họa | chạy mục 2 trước |
| Tin realtime mới nhất | điền `NEWS_MONGO_URI` + chạy mục 3 |
| Phân tích tin tiếng Việt chi tiết | điền `API_KEY` (xAI) vào `.env` |

## Lỗi thường gặp

- **"Run the LSH pipeline first"** → artifact dir rỗng. Đổi `--artifact-dir` sang
  `lsh_combined` hoặc `lsh_full`.
- **Máy lag khi bấm Bắt đầu** → dùng `lsh_combined` (nhẹ) thay vì `lsh_full`, đóng bớt tab.
- **Bản tin đọc câu chung chung** → chưa có `API_KEY` Grok trong `.env`.
- **Không lấy được tin X** → kiểm tra `NEWS_MONGO_URI`; collection là
  `x_russia_ukraine_posts` và `x_us_iran_posts` trong DB `news_monitoring`.
