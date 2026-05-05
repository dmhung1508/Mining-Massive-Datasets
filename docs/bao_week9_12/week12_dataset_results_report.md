# Week 12 - Dataset and Results (Bao)

## 1) Dataset mo ta ngan

- Du lieu duoc xu ly theo pipeline LSH cua nhom (Twitter + Telegram sau khi chuan hoa).
- Tap benchmark va demo cases duoc lay tu artifact da precompute de dam bao tai lap.
- Cac bang case duoc dung: top clusters, exact duplicates, near duplicates.

## 2) Ket qua thuc nghiem

- Week 9 pairs tong hop: `10`
- Week 9 connected components: `6`
- Week 10 max documents benchmark: `49852`
- Week 10 candidate reduction trung binh: `99.996778%`
- Week 11 exact demo examples: `5`
- Week 11 near-duplicate demo examples: `5`

## 3) Insight

- LSH giup cat giam candidate pairs rat lon so voi toan bo cap co the.
- Van giu duoc cac cap duplicate/near-duplicate co chat luong de phuc vu demo.
- Cluster lon giup minh hoa cac luong noi dung lap tren social stream.

## 4) Han che

- Nguong Jaccard co the bo sot noi dung paraphrase xa.
- Ket qua benchmark va demo phu thuoc vao chat luong preprocess.
- Cac case dua vao slide nen duoc loc theo do ro rang ve noi dung va encoding.

## 5) Kien nghi cho final report

- Dat histogram similarity + cluster graph o phan ket qua.
- Dat benchmark runtime ngay sau pipeline de giai thich ly do chon LSH.
- Dung 1 exact case + 1 near case + 1 top cluster de chot thong diep.
