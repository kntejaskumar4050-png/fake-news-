"""
Capture application screenshots with Playwright.

The in-app browser needs a visible desktop window; this script uses headless
Chromium instead, so it works in CI and from a terminal.

    pip install playwright && playwright install chromium
    python app.py                      # in another terminal
    python scripts/capture_screenshots.py --base-url http://127.0.0.1:5000

Screenshots are written to static/images/screenshots/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "static" / "images" / "screenshots"

VIEWPORT = {"width": 1440, "height": 1000}
MOBILE_VIEWPORT = {"width": 390, "height": 844}


def _settle(page) -> None:
    """Disable scroll-reveal animations and wait for fonts/network."""
    page.wait_for_load_state("networkidle")
    page.add_style_tag(content="""
        .reveal { opacity: 1 !important; transform: none !important; }
        .hero-glow, .orb { animation: none !important; }
        /* A sticky header can bleed into element screenshots and toasts are
           transient UI, so neutralise both while capturing. */
        .site-header { position: static !important; }
        .toast-stack { display: none !important; }
    """)
    page.evaluate("document.fonts && document.fonts.ready")
    page.wait_for_timeout(400)


def _shot(page, name: str, full_page: bool = True, selector: str | None = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    if selector:
        page.locator(selector).screenshot(path=str(path), animations="disabled")
    else:
        page.screenshot(path=str(path), full_page=full_page, animations="disabled")
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def main(base_url: str) -> int:
    base_url = base_url.rstrip("/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        # ---------------- Desktop ---------------- #
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = context.new_page()
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))

        print("Home …")
        page.goto(f"{base_url}/", wait_until="load")
        _settle(page)
        _shot(page, "home.png")

        print("Detection workspace …")
        page.goto(f"{base_url}/detect", wait_until="load")
        _settle(page)
        _shot(page, "detect.png")

        print("Prediction result …")
        page.click('[data-sample="consumer"]')
        page.click("#analyzeBtn")
        page.wait_for_selector("#resultContent:not([hidden])", timeout=60_000)
        page.wait_for_timeout(1400)  # gauge + bar animations
        _shot(page, "result.png")
        _shot(page, "result-panel.png", selector="#resultPanel")

        print("Dashboard …")
        page.goto(f"{base_url}/dashboard", wait_until="load")
        page.wait_for_selector("#historyBody tr td:not(.table-empty)", timeout=30_000)
        _settle(page)
        page.wait_for_timeout(1600)  # Chart.js animations
        _shot(page, "dashboard.png")

        print("About …")
        page.goto(f"{base_url}/about", wait_until="load")
        _settle(page)
        _shot(page, "about.png")
        _shot(page, "results-section.png", selector="#results")

        context.close()

        # ---------------- Mobile ---------------- #
        print("Mobile home …")
        mobile = browser.new_context(
            viewport=MOBILE_VIEWPORT, device_scale_factor=3, is_mobile=True,
            has_touch=True,
        )
        mpage = mobile.new_page()
        mpage.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        mpage.goto(f"{base_url}/", wait_until="load")
        _settle(mpage)
        _shot(mpage, "mobile-home.png")

        print("Mobile detection …")
        mpage.goto(f"{base_url}/detect", wait_until="load")
        _settle(mpage)
        mpage.click('[data-sample="policy"]')
        mpage.click("#analyzeBtn")
        mpage.wait_for_selector("#resultContent:not([hidden])", timeout=60_000)
        mpage.wait_for_timeout(1200)
        _shot(mpage, "mobile-result.png")

        mobile.close()
        browser.close()

    if errors:
        print("\nConsole errors captured:")
        for message in errors:
            print("  -", message)
        return 1

    print("\nAll screenshots captured without console errors.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture UI screenshots")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    sys.exit(main(parser.parse_args().base_url))
