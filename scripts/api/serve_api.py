"""Launch the FastAPI backend with uvicorn.

Examples:
    python scripts/api/serve_api.py
    python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_combined --port 8765
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from social_lsh.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED
from social_lsh.runtime_check import build_run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Social LSH FastAPI backend.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reload", action="store_true", help="Enable autoreload for development.")
    parser.add_argument("--skip-preflight", action="store_true", help="Do not print the startup data/resource check.")
    args = parser.parse_args()

    # Pass configuration to the app module through environment variables so the
    # FastAPI app object stays importable by uvicorn workers.
    os.environ["SOCIAL_LSH_ARTIFACT_DIR"] = str(Path(args.artifact_dir))
    os.environ["SOCIAL_LSH_SEED"] = str(args.seed)

    if not args.skip_preflight:
        check = build_run_preflight(Path(args.artifact_dir), top_n=5, samples_per_cluster=12, use_llm=True)
        print("Preflight:")
        print(f"  status: {check['status']}")
        print(f"  artifact: {check['artifact_dir']}")
        print(f"  rows: {check['artifact_rows']}")
        print(f"  scale size: {check['artifact_sizes_gib']['scale_shingles']} GiB")
        print(f"  RAM available: {check['memory'].get('available_gib')} GiB")
        print(f"  metadata cache: {'hit' if check['cache']['hit'] else 'miss'} ({check['cache']['metadata_path']})")
        for warning in check["warnings"]:
            print(f"  warning: {warning}")
        for recommendation in check["recommendations"]:
            print(f"  plan: {recommendation}")
        if check["status"] == "danger":
            raise SystemExit("Preflight failed. Use a valid artifact dir or free resources before serving.")

    # Ensure the FastAPI app module (app.py, same folder) is importable by uvicorn.
    api_dir = Path(__file__).resolve().parent
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(api_dir), os.environ.get("PYTHONPATH", "")])
    )
    sys.path.insert(0, str(api_dir))

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is required to serve the API. Install it with: pip install -e '.[api]'"
        ) from exc

    print(f"Serving Social LSH API at http://{args.host}:{args.port}")
    print("Docs: http://%s:%d/docs" % (args.host, args.port))
    print("Endpoints: /health, /metrics, /clusters/top?limit=10, /search?text=...")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
