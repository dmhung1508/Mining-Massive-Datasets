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


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Social LSH FastAPI backend.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reload", action="store_true", help="Enable autoreload for development.")
    args = parser.parse_args()

    # Pass configuration to the app module through environment variables so the
    # FastAPI app object stays importable by uvicorn workers.
    os.environ["SOCIAL_LSH_ARTIFACT_DIR"] = str(Path(args.artifact_dir))
    os.environ["SOCIAL_LSH_SEED"] = str(args.seed)

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
