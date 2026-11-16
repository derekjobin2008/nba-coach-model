"""Helpers for tests that open the scout page in a real, headless Chrome (see chrome_driver.mjs)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

DRIVER = Path(__file__).with_name("chrome_driver.mjs")
PHONE = {"width": 375, "height": 812}
LAPTOP = {"width": 1280, "height": 800}


def find_chrome() -> str | None:
    candidates = [os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")]
    return next((path for path in candidates if path and Path(path).exists()), None)


CHROME = find_chrome()
needs_chrome = pytest.mark.skipif(CHROME is None or shutil.which("node") is None,
                                  reason="needs Google Chrome (or CHROME=path) and Node.js")


def in_chrome(steps: list[dict], tmp_path: Path) -> list:
    """Run the steps in Chrome; returns each "run" step's value. Fails on any error in the page."""
    steps_path = tmp_path / "steps.json"
    steps_path.write_text(json.dumps(steps))
    result = subprocess.run(["node", str(DRIVER), CHROME, str(steps_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    *values, errors = json.loads(result.stdout)
    assert errors["pageErrors"] == []
    return values


def open_page(url: str, size: dict, dark: bool = False, link: str = "") -> dict:
    return {"open": url + (f"#{link}" if link else ""), **size, "dark": dark}
