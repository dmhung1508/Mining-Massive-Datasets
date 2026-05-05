# Practice Demo Checklist (Bao)

## Truoc buoi demo
- [ ] Mo san cac file chart/table trong docs/bao_deliverables
- [ ] Mo san case files trong docs/hung_deliverables
- [ ] Kiem tra lai thu tu slide va luong ke chuyen

## Dry-run 1
- [ ] Trinh bay duoc bai toan trong <= 60 giay
- [ ] Di qua pipeline va threshold verify ro rang
- [ ] Minh hoa 1 exact duplicate va 1 near duplicate
- [ ] Giai thich duoc vi sao LSH nhanh hon brute-force

## Dry-run 2
- [ ] Giu tong thoi luong trong 8-10 phut
- [ ] Chuyen slide khong bi ngat quang
- [ ] Tra loi duoc 3 cau hoi thuong gap ve precision/recall, false positive, tuning

## Q&A quick answers
- Tai sao threshold 0.8? -> Can bang giua precision va recall cho near-duplicate.
- Tai sao khong brute-force full data? -> Do O(N^2), khong practical khi scale lon.
- LSH co bo sot khong? -> Co, vi vay can buoc verify exact Jaccard sau candidate generation.
