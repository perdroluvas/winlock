import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

USER_AGENT = "winlock (personal data project)"


def _get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def portrait_urls(heroes_json: list[dict], image_key: str) -> dict[int, str]:
    return {
        h["id"]: h["images"][image_key] for h in heroes_json if h.get("images", {}).get(image_key)
    }


def fetch_portraits(
    api_url: str, image_key: str, dest: Path, hero_ids: list[int], size: int
) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    missing = [h for h in hero_ids if not (dest / f"{h}.png").exists()]
    if not missing:
        return []
    urls = portrait_urls(json.loads(_get(api_url)), image_key)
    written = []
    for hero_id in missing:
        if hero_id not in urls:
            continue
        img = Image.open(io.BytesIO(_get(urls[hero_id]))).convert("RGBA")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img.save(dest / f"{hero_id}.png")
        written.append(dest / f"{hero_id}.png")
    return written


def circle_crop(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    r = min(h, w) / 2
    outside = (xx + 0.5 - w / 2) ** 2 + (yy + 0.5 - h / 2) ** 2 > r**2
    out = rgba.copy()
    out[outside, 3] = 0
    return out


def load_portraits(dest: Path | None, hero_ids) -> dict[int, np.ndarray]:
    if dest is None:
        return {}
    out = {}
    for hero_id in hero_ids:
        path = dest / f"{int(hero_id)}.png"
        if path.exists():
            rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=float) / 255
            out[int(hero_id)] = circle_crop(rgba)
    return out
