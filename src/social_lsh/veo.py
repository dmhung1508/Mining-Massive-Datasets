"""Client for the YEScale media-generation API (video via grok-video, images via gpt-image).

Flow:
1. submit_task(prompt, config) -> task_id
2. poll_task(task_id) until status is terminal
3. download the returned media URL to disk

Configuration is read from environment variables (see .env):
- API_VEO        : bearer token (shared by image and video models)
- base_url_veo   : submit endpoint (default https://api.yescale.io/task/submit)
- VEO_MODEL      : video model (default grok-video; veo3.1 also supported)
- IMAGE_MODEL    : image model (default gpt-image)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

DEFAULT_SUBMIT_URL = "https://api.yescale.io/task/submit"
DEFAULT_MODEL = "grok-video"
DEFAULT_IMAGE_MODEL = "gpt-image"

# Allowed config values reported by the API.
VALID_SIZES = ("720p", "1080p", "2160p")
VALID_ASPECT_RATIOS = ("16:9", "9:16")
VALID_VIDEO_ASPECT_RATIOS = ("2:3", "3:2", "16:9", "9:16", "1:1")
VALID_VIDEO_DURATIONS = (6, 10)
VALID_IMAGE_SIZES = ("1024x1024", "1024x1536", "1536x1024")
VALID_IMAGE_QUALITY = ("low", "medium", "high")

TERMINAL_SUCCESS = {"SUCCESS", "SUCCEEDED", "COMPLETED"}
TERMINAL_FAILURE = {"FAILURE", "FAILED", "ERROR", "CANCELLED"}


class VeoError(RuntimeError):
    """Raised when the YEScale media API rejects a request or a task fails."""


# Backwards-compatible alias: the module covers both video and image tasks.
MediaError = VeoError


@dataclass
class GrokVideoConfig:
    """Config for the grok-video model (the working video model on YEScale)."""

    duration: int = 6
    aspect_ratio: str = "16:9"

    def as_dict(self) -> dict[str, Any]:
        if self.duration not in VALID_VIDEO_DURATIONS:
            raise ValueError(f"duration must be one of {VALID_VIDEO_DURATIONS}, got {self.duration!r}")
        if self.aspect_ratio not in VALID_VIDEO_ASPECT_RATIOS:
            raise ValueError(
                f"aspect_ratio must be one of {VALID_VIDEO_ASPECT_RATIOS}, got {self.aspect_ratio!r}"
            )
        return {"duration": self.duration, "aspect_ratio": self.aspect_ratio}


@dataclass
class ImageConfig:
    size: str = "1024x1024"
    quality: str = "low"
    background: str = "opaque"

    def as_dict(self) -> dict[str, Any]:
        if self.size not in VALID_IMAGE_SIZES:
            raise ValueError(f"size must be one of {VALID_IMAGE_SIZES}, got {self.size!r}")
        if self.quality not in VALID_IMAGE_QUALITY:
            raise ValueError(f"quality must be one of {VALID_IMAGE_QUALITY}, got {self.quality!r}")
        return {
            "background": self.background,
            "quality": self.quality,
            "size": self.size,
        }


@dataclass
class VeoConfig:
    size: str = "720p"
    aspect_ratio: str = "16:9"
    enhance_prompt: bool = True

    def as_dict(self) -> dict[str, Any]:
        if self.size not in VALID_SIZES:
            raise ValueError(f"size must be one of {VALID_SIZES}, got {self.size!r}")
        if self.aspect_ratio not in VALID_ASPECT_RATIOS:
            raise ValueError(f"aspect_ratio must be one of {VALID_ASPECT_RATIOS}, got {self.aspect_ratio!r}")
        return {
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
            "enhance_prompt": self.enhance_prompt,
        }


@dataclass
class VeoClient:
    api_key: str
    submit_url: str = DEFAULT_SUBMIT_URL
    model: str = DEFAULT_MODEL
    session: requests.Session = field(default_factory=requests.Session)

    @classmethod
    def from_env(cls) -> "VeoClient":
        api_key = (os.getenv("API_VEO") or "").strip()
        if not api_key:
            raise VeoError("API_VEO is not set. Put it in your .env file.")
        submit_url = (os.getenv("base_url_veo") or DEFAULT_SUBMIT_URL).strip()
        model = (os.getenv("VEO_MODEL") or DEFAULT_MODEL).strip()
        return cls(api_key=api_key, submit_url=submit_url, model=model)

    @classmethod
    def for_images(cls) -> "VeoClient":
        """Build a client configured for the image model (shares the YEScale key)."""
        api_key = (os.getenv("API_VEO") or "").strip()
        if not api_key:
            raise VeoError("API_VEO is not set. Put it in your .env file.")
        submit_url = (os.getenv("base_url_veo") or DEFAULT_SUBMIT_URL).strip()
        model = (os.getenv("IMAGE_MODEL") or DEFAULT_IMAGE_MODEL).strip()
        return cls(api_key=api_key, submit_url=submit_url, model=model)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @property
    def _task_base_url(self) -> str:
        # Derive ".../task" from the submit URL (".../task/submit").
        return self.submit_url.rsplit("/", 1)[0]

    def submit_task(self, prompt: str, config: Any | None = None, task_type: str = "generate") -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        config_obj = config if config is not None else VeoConfig()
        config_dict = config_obj.as_dict() if hasattr(config_obj, "as_dict") else dict(config_obj)
        payload = {
            "model": self.model,
            "task_type": task_type,
            "prompt": prompt.strip(),
            "config": config_dict,
        }
        response = self.session.post(self.submit_url, headers=self._headers, json=payload, timeout=60)
        data = _safe_json(response)
        if response.status_code != 200:
            raise VeoError(f"submit failed ({response.status_code}): {data}")
        task_id = data.get("task_id") or data.get("data", {}).get("task_id")
        if not task_id:
            raise VeoError(f"submit response missing task_id: {data}")
        return task_id

    def get_task(self, task_id: str) -> dict[str, Any]:
        url = f"{self._task_base_url}/{task_id}"
        response = self.session.get(url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
        return _safe_json(response)

    def poll_task(
        self,
        task_id: str,
        interval_seconds: float = 15.0,
        timeout_seconds: float = 600.0,
        on_update: Any = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            data = self.get_task(task_id)
            status = str(data.get("status", "")).upper()
            if on_update is not None:
                on_update(status, data.get("progress", ""))
            if status in TERMINAL_SUCCESS:
                return data
            if status in TERMINAL_FAILURE:
                reason = data.get("err_reason") or data.get("error") or data
                raise VeoError(f"task {task_id} failed: {reason}")
            if time.monotonic() >= deadline:
                raise VeoError(f"task {task_id} timed out after {timeout_seconds}s (last status: {status})")
            time.sleep(interval_seconds)

    def generate(
        self,
        prompt: str,
        config: Any | None = None,
        interval_seconds: float = 15.0,
        timeout_seconds: float = 600.0,
        on_update: Any = None,
    ) -> dict[str, Any]:
        task_id = self.submit_task(prompt, config=config)
        if on_update is not None:
            on_update("SUBMITTED", task_id)
        return self.poll_task(
            task_id,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            on_update=on_update,
        )

    def download_media(self, task_data: dict[str, Any], output_path: Path | str) -> Path:
        """Download the first media URL (video or image) from a task result."""
        url = extract_media_url(task_data)
        if not url:
            raise VeoError(f"no media URL found in task result: {task_data}")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with self.session.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with output.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
        return output

    # Backwards-compatible alias.
    def download_video(self, task_data: dict[str, Any], output_path: Path | str) -> Path:
        return self.download_media(task_data, output_path)


def extract_media_url(task_data: dict[str, Any]) -> str | None:
    """Find the first media URL (video or image) anywhere in the task payload.

    The exact result shape varies by model, so we search recursively for a
    URL-looking string rather than assuming a fixed key.
    """
    candidates: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            candidates.append(node)

    walk(task_data.get("task_result") or task_data.get("results") or task_data)

    media_exts = (".mp4", ".mov", ".webm", ".png", ".jpg", ".jpeg", ".webp")
    for value in candidates:
        lowered = value.lower()
        if lowered.startswith("http") and (any(ext in lowered for ext in media_exts) or "video" in lowered):
            return value
    # Fall back to any http URL.
    for value in candidates:
        if value.lower().startswith("http"):
            return value
    return None


def extract_video_url(task_data: dict[str, Any]) -> str | None:
    """Find the first video URL anywhere in the task result payload."""
    candidates: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            candidates.append(node)

    walk(task_data.get("task_result") or task_data.get("results") or task_data)

    for value in candidates:
        lowered = value.lower()
        if lowered.startswith("http") and (
            ".mp4" in lowered or "video" in lowered or ".mov" in lowered or ".webm" in lowered
        ):
            return value
    # Fall back to any http URL if no obvious video extension was found.
    for value in candidates:
        if value.lower().startswith("http"):
            return value
    return None


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        return {"_raw": response.text, "_status": response.status_code}
