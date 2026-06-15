"""Text-to-speech client for the PTIT Holobox synthesize endpoint.

The endpoint returns a 16-bit PCM mono WAV (24 kHz):

    POST https://aitools.ptit.edu.vn/holobox/synthesize
    {"text": "..."}  ->  audio/wav bytes
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

DEFAULT_TTS_URL = "https://aitools.ptit.edu.vn/holobox/synthesize"


class TTSError(RuntimeError):
    """Raised when the TTS endpoint fails."""


@dataclass
class TTSClient:
    url: str = DEFAULT_TTS_URL
    session: requests.Session = field(default_factory=requests.Session)

    @classmethod
    def from_env(cls) -> "TTSClient":
        url = (os.getenv("TTS_URL") or DEFAULT_TTS_URL).strip()
        return cls(url=url)

    def synthesize(self, text: str, timeout: float = 120.0) -> bytes:
        """Return WAV audio bytes for the given text."""
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        response = self.session.post(
            self.url,
            headers={"Content-Type": "application/json"},
            json={"text": text.strip()},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise TTSError(f"TTS failed ({response.status_code}): {response.text[:200]}")
        content_type = response.headers.get("Content-Type", "")
        if "audio" not in content_type and not response.content[:4] == b"RIFF":
            raise TTSError(f"unexpected TTS response content-type: {content_type}")
        return response.content

    def synthesize_to_file(self, text: str, output_path: Path | str, timeout: float = 120.0) -> Path:
        audio = self.synthesize(text, timeout=timeout)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(audio)
        return output
