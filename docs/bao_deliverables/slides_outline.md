# Slide Outline (Bao)

## Slide 1 - Title
- Mining Massive Datasets: Near-Duplicate Detection with MinHash + LSH

## Slide 2 - Problem and pipeline
- Bai toan: tim noi dung gan trung lap tren social stream.
- Pipeline: preprocessing -> shingling -> MinHash -> LSH -> verify -> cluster.

## Slide 3 - Similarity distribution
- Dung `docs/bao_deliverables/similarity_histogram.png` de trinh bay phan bo similarity.
- Nhan manh nguong xac minh Jaccard >= 0.8.

## Slide 4 - Cluster graph
- Dung `docs/bao_deliverables/cluster_graph.png` de mo ta cau truc cum.
- Chi ra so component va component lon nhat.

## Slide 5 - Scalability and runtime
- Dung `docs/bao_deliverables/performance_dashboard.png` de so sanh hieu nang.
- Ket luan: brute-force chi hop cho subset, LSH moi scale duoc.

## Slide 6 - Top cluster examples
- Nguon bang: `docs/hung_deliverables/top_clusters.csv`
- Chon 2-3 cum co y nghia de minh hoa story.

## Slide 7 - Exact duplicate demo
- Nguon bang: `docs/hung_deliverables/exact_duplicate_examples.csv`
- Giai thich vi sao Jaccard = 1.0 va tai sao can dedup.

## Slide 8 - Near duplicate demo
- Nguon bang: `docs/hung_deliverables/near_duplicate_examples.csv`
- Giai thich overlap shingles du text khac nhau.

## Slide 9 - Risks and limitations
- Nguong Jaccard co the bo sot paraphrase xa.
- Sampling va nguon du lieu co the gay lech nhan xet.

## Slide 10 - Final takeaway
- LSH giam candidate rat manh, giu lai kha nang tim duplicate.
- KQ du tot de support demo va bao cao cuoi ky.
