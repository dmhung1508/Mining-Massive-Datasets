from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Single output root for every generated artifact (datasets, LSH outputs, visuals).
OUTPUT_ROOT = REPO_ROOT / "jupyter" / "output"

# Twitter parquet dataset. It normally lives at the repo root; fall back to the
# output directory if a regenerated copy is stored there instead.
def _default_input_parquet() -> Path:
    candidates = [
        REPO_ROOT / "tweets_final.parquet",
        OUTPUT_ROOT / "tweets_final.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_INPUT_PARQUET = _default_input_parquet()
DEFAULT_ARTIFACT_DIR = OUTPUT_ROOT / "lsh_combined"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "metrics.json"

# Default location for the merged Twitter + Telegram dataset.
DEFAULT_COMBINED_DATASET = OUTPUT_ROOT / "combined_social.parquet"
DEFAULT_TELEGRAM_EXPORT = OUTPUT_ROOT / "telegram_messages.parquet"
DEFAULT_X_EXPORT = OUTPUT_ROOT / "x_messages.parquet"

# Default directory for weekly visualization artifacts (was the ad-hoc "datatele").
DEFAULT_VISUALS_DIR = OUTPUT_ROOT / "visuals"

DEFAULT_SEED = 42
DEFAULT_SHINGLE_SIZE = 3
DEFAULT_VERIFY_THRESHOLD = 0.8
BASELINE_SIZE = 3_000
SCALE_SIZE = 50_000


@dataclass(frozen=True)
class LSHConfig:
    shingle_size: int
    num_perm: int
    bands: int
    rows: int

    @property
    def name(self) -> str:
        return (
            f"k{self.shingle_size}_perm{self.num_perm}"
            f"_bands{self.bands}_rows{self.rows}"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "shingle_size": self.shingle_size,
            "num_perm": self.num_perm,
            "bands": self.bands,
            "rows": self.rows,
        }


CONFIG_GRID = (
    LSHConfig(shingle_size=3, num_perm=128, bands=32, rows=4),
    LSHConfig(shingle_size=3, num_perm=128, bands=16, rows=8),
    LSHConfig(shingle_size=3, num_perm=160, bands=20, rows=8),
    LSHConfig(shingle_size=4, num_perm=160, bands=20, rows=8),
)
