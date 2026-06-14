# Week 10 Benchmark - Hung

## Muc tieu

- Do scalability cua brute-force Jaccard va MinHash + LSH tren cac kich thuoc mau khac nhau.
- Chung minh brute-force chi phu hop voi baseline subset, con LSH phu hop hon cho scale run.
- Tao bang so sanh runtime de dua vao demo va bao cao.

## Cau hinh LSH cuoi

- shingle_size: `3`
- num_perm: `128`
- bands: `16`
- rows: `8`

## Brute-force Jaccard

| stage | documents | pairs_considered | positive_pairs | candidate_pairs | runtime_seconds | pairs_per_second | candidate_reduction_ratio | signature_runtime_seconds | bucket_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brute_force_jaccard | 400 | 79800 | 2 |  | 0.027726 | 2878164.9 |  |  |  |
| brute_force_jaccard | 800 | 319600 | 10 |  | 0.103094 | 3100083.42 |  |  |  |
| brute_force_jaccard | 1200 | 719400 | 17 |  | 0.253602 | 2836728.42 |  |  |  |

## MinHash + LSH

| stage | documents | pairs_considered | positive_pairs | candidate_pairs | runtime_seconds | pairs_per_second | candidate_reduction_ratio | signature_runtime_seconds | bucket_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minhash_lsh_candidates | 5000 | 12497500 |  | 448 | 0.289794 |  | 0.99996415 | 0.11072 | 76250.0 |
| minhash_lsh_candidates | 10000 | 49995000 |  | 1573 | 0.588244 |  | 0.99996854 | 0.228557 | 150063.0 |
| minhash_lsh_candidates | 49852 | 1242586026 |  | 36487 | 5.674673 |  | 0.99997064 | 2.644158 | 683795.0 |

## Ket luan

- Brute-force co so cap tang theo `N * (N - 1) / 2`, nen chi dung de tao ground truth tren tap mau.
- LSH sinh candidate pairs nho hon rat nhieu so voi tong so cap co the co.
- Ket qua benchmark nay dung cho task tuan 10: scalability test, runtime comparison, va toi uu demo.
