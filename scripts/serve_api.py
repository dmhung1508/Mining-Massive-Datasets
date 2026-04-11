from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from _bootstrap import add_src_to_path

add_src_to_path()

from uk_russia_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe, read_metrics
from uk_russia_lsh.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED
from uk_russia_lsh.search import prepare_search_index, search_similar_tweets


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def make_handler(artifact_dir: Path, seed: int) -> type[BaseHTTPRequestHandler]:
    class LSHApiHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/health":
                _json_response(self, 200, {"status": "ok", "artifact_dir": str(artifact_dir)})
                return

            if parsed.path == "/metrics":
                _json_response(self, 200, read_metrics(artifact_dir / "metrics.json"))
                return

            if parsed.path == "/clusters/top":
                limit = _parse_int(query.get("limit", [None])[0], 10)
                clusters = read_dataframe(artifact_path("clusters", artifact_dir))
                top = (
                    clusters[["cluster_id", "cluster_size"]]
                    .drop_duplicates()
                    .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
                    .head(limit)
                )
                _json_response(self, 200, top.to_dict(orient="records"))
                return

            if parsed.path == "/search":
                text = query.get("text", [""])[0].strip()
                if not text:
                    _json_response(self, 400, {"error": "Missing required query parameter: text"})
                    return
                top_k = _parse_int(query.get("top_k", [None])[0], 5)
                min_jaccard = _parse_float(query.get("min_jaccard", [None])[0], 0.0)
                results, metadata = search_similar_tweets(
                    query_text=text,
                    artifact_dir=artifact_dir,
                    top_k=top_k,
                    min_jaccard=min_jaccard,
                    seed=seed,
                )
                _json_response(
                    self,
                    200,
                    {
                        "metadata": metadata,
                        "results": results.to_dict(orient="records"),
                    },
                )
                return

            _json_response(
                self,
                404,
                {
                    "error": "Unknown endpoint",
                    "endpoints": ["/health", "/metrics", "/clusters/top?limit=10", "/search?text=..."],
                },
            )

    return LSHApiHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a small JSON API for the LSH demo artifacts.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    artifact_dir = ensure_artifact_dir(Path(args.artifact_dir))
    prepare_search_index(artifact_dir=artifact_dir, seed=args.seed, force_rebuild=args.rebuild_index)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(artifact_dir, args.seed))
    print(f"Serving LSH API at http://{args.host}:{args.port}")
    print("Endpoints: /health, /metrics, /clusters/top?limit=10, /search?text=...")
    server.serve_forever()


if __name__ == "__main__":
    main()
