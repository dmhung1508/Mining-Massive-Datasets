from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PARQUET = REPO_ROOT / "jupyter" / "output" / "tweets_final.parquet"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "jupyter" / "output" / "lsh"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "metrics.json"

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
