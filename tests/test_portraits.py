"""Portrait helpers that need no network: URL picking, circle crop, cache loading."""

import numpy as np
from PIL import Image

from winlock.portraits import circle_crop, load_portraits, portrait_urls


def test_portrait_urls_skips_heroes_without_the_image():
    heroes = [
        {"id": 1, "images": {"icon_image_small": "a.png"}},
        {"id": 2, "images": {}},
        {"id": 3},
    ]
    assert portrait_urls(heroes, "icon_image_small") == {1: "a.png"}


def test_circle_crop_clears_corners_keeps_center():
    out = circle_crop(np.ones((10, 10, 4)))
    assert out[0, 0, 3] == 0  # corner: outside the circle
    assert out[5, 5, 3] == 1  # center: kept
    assert out[5, 5, :3].tolist() == [1, 1, 1]  # colors untouched


def test_load_portraits_reads_only_cached_heroes(tmp_path):
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(tmp_path / "7.png")
    got = load_portraits(tmp_path, [7, 8])
    assert list(got) == [7]
    assert got[7].shape == (8, 8, 4)
    assert load_portraits(None, [7]) == {}
