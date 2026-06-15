"""OpenAI image generation (synchronous /images/generations).

Unlike the YEScale task/poll flow, the OpenAI Images API returns the result in a
single call (as a URL or base64). Configuration from .env:
- OPENAI_API_KEY : bearer token
- base_url       : API base (default https://api.openai.com/v1)
- model_image    : image model (e.g. gpt-image-2)
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
VALID_SIZES = ("1024x1024", "1024x1536", "1536x1024", "auto")


class ImageError(RuntimeError):
    """Raised when image generation fails."""


@dataclass
class OpenAIImageClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_IMAGE_MODEL

    @classmethod
    def from_env(cls) -> "OpenAIImageClient":
        api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "").strip()
        if not api_key:
            raise ImageError("OPENAI_API_KEY is not set in .env.")
        base_url = (os.getenv("base_url") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()
        model = (os.getenv("model_image") or os.getenv("IMAGE_MODEL") or DEFAULT_IMAGE_MODEL).strip()
        return cls(api_key=api_key, base_url=base_url, model=model)

    def generate_to_file(
        self,
        prompt: str,
        output_path: Path | str,
        size: str = "1024x1024",
        timeout: float = 180.0,
    ) -> Path:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        payload = {"model": self.model, "prompt": prompt.strip(), "size": size}
        response = requests.post(
            f"{self.base_url.rstrip('/')}/images/generations",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            raise ImageError(f"image generation failed ({response.status_code}): {response.text[:300]}")

        data = (response.json().get("data") or [{}])[0]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if data.get("b64_json"):
            output.write_bytes(base64.b64decode(data["b64_json"]))
            return output
        if data.get("url"):
            img = requests.get(data["url"], timeout=timeout)
            img.raise_for_status()
            output.write_bytes(img.content)
            return output
        raise ImageError(f"no image payload in response: {list(data.keys())}")
