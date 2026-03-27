from __future__ import annotations

import argparse
import sys

from _bootstrap import add_src_to_path

add_src_to_path()

from uk_russia_lsh import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED, prepare_search_index, search_similar_tweets


def main() -> None:
    parser = argparse.ArgumentParser(description="Search similar tweets for an input query using the selected LSH config.")
    parser.add_argument("--text", help="Query text to search for similar tweets.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-jaccard", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    query_text = args.text
    if not query_text:
        query_text = input("Nhap query text: ").strip()
    if not query_text:
        raise SystemExit("Query text is required.")

    prepare_search_index(
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        force_rebuild=args.rebuild_index,
    )
    results, metadata = search_similar_tweets(
        query_text=query_text,
        artifact_dir=args.artifact_dir,
        top_k=args.top_k,
        min_jaccard=args.min_jaccard,
        seed=args.seed,
    )

    sys.stdout.reconfigure(encoding="utf-8")
    print("Search metadata:")
    print(metadata)
    if results.empty:
        print("\nNo similar tweets found for this query.")
        return

    printable = results.copy()
    printable["text"] = printable["text"].map(lambda text: " ".join(str(text).split()))
    print("\nTop similar tweets:")
    print(printable.to_string(index=False, max_colwidth=120))


if __name__ == "__main__":
    main()
