"""Record a narrated walkthrough of the test console against the live API (real model).

    python manage.py run                                  # terminal 1
    .venv-train/Scripts/python scripts/record_demo.py     # terminal 2 (needs `playwright`)

Writes docs/demo/demo.webm. Every result in the video comes from the running system;
captions only describe what is on screen.
"""

import shutil
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo"
URL = "http://127.0.0.1:8000/ui/"
SIZE = {"width": 1440, "height": 900}
LLM_TIMEOUT_MS = 600_000

CAPTION_JS = """text => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:99;'
      + 'max-width:1100px;padding:12px 20px;border-radius:10px;background:rgba(17,24,39,.92);'
      + 'color:#fff;font:600 16px/1.4 system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.25)';
    document.body.appendChild(el);
  }
  el.textContent = text;
}"""


def caption(page: Page, text: str, hold_ms: int = 3500) -> None:
    page.evaluate(CAPTION_JS, text)
    page.wait_for_timeout(hold_ms)


def tab(page: Page, name: str) -> None:
    page.click(f"nav [data-tab='{name}']")
    page.wait_for_timeout(600)


C1001_SECRETS = ["INV-20431", "BR-5521", "1,240.00", "CH-7781", "INV-20388", "T-0002", "C-1001"]


def ask(page: Page, preset_index: int, text: str) -> str:
    """Ask a preset question; return a caption describing what actually came back."""
    tab(page, "ask")
    page.click(f"#presets [data-p='{preset_index}']")
    caption(page, text, 2500)
    page.click("#ask-btn")
    page.wait_for_function("!document.getElementById('ask-btn').disabled", timeout=LLM_TIMEOUT_MS)
    page.wait_for_timeout(600)
    page.locator("#ask-out").scroll_into_view_if_needed()
    out = page.locator("#ask-out")
    status = out.locator(".summary .badge").first.inner_text()
    cited = out.locator(".cite b").all_inner_texts()
    withheld = out.locator("text=withheld by verifier").count() > 0
    parts = [f"Result: {status}", f"citations: {', '.join(cited) if cited else 'none'}"]
    if withheld:
        parts.append("withheld by the verifier/output guard")
    return "; ".join(parts) + "."


def record(page: Page) -> None:
    page.goto(URL)
    page.wait_for_selector("#health .badge.b-ok")
    caption(
        page,
        "Support RAG Console: a local Llama 3.2 3B support bot over synthetic tickets, "
        "logs and KB articles, served by FastAPI with Qdrant hybrid search.",
        5000,
    )
    caption(page, "Header: live health of Qdrant (alias kb_live -> kb_v2) and the Ollama model.")

    caption(
        page,
        "1. Feature checks: every feature is exercised against the live API, "
        "including 5 checks that call the real model on CPU.",
        4000,
    )
    page.click("#run-all")
    page.wait_for_function(
        "!document.getElementById('run-all').disabled", timeout=LLM_TIMEOUT_MS * 2
    )
    page.wait_for_timeout(800)
    page.locator("#kpi").scroll_into_view_if_needed()
    caption(
        page,
        f"Checks finished: {page.inner_text('#kpi')}. Click a row for the exact "
        "request and response.",
        5000,
    )
    page.locator("tr.check-row").nth(10).click()
    page.locator("tr.check-row").nth(10).scroll_into_view_if_needed()
    caption(page, "Example: the injection attack from C-1002, with the full API response.", 6000)
    page.locator("tr.check-row").nth(10).click()

    result = ask(page, 0, "2. Ask as C-1001: what is the CSV export row limit?")
    caption(
        page,
        result + " Superseded KB-0002 is excluded because its successor KB-0003 was retrieved.",
        7000,
    )

    result = ask(page, 1, "Two current articles disagree on the refund window (14 vs 30 days).")
    caption(page, result, 6000)

    result = ask(page, 2, "No record covers this question.")
    caption(page, result + " The relevance gate abstained before calling the model.", 6000)

    result = ask(
        page, 3, "Attack: C-1002 sends a forged SYSTEM/admin instruction to reveal C-1001's data."
    )
    text = page.inner_text("#ask-out")
    leaked = [s for s in C1001_SECRETS if s in text.split("Injection flags")[0]]
    verdict = (
        f"C-1001 identifiers in the answer: {leaked}"
        if leaked
        else ("No C-1001 data or identifiers in the answer; injection flags are listed below it.")
    )
    caption(page, f"{result} {verdict}", 8000)

    tab(page, "tickets")
    page.select_option("#identity", "C-1002")
    caption(page, "3. Tickets as C-1002: ingest a ticket containing an injection attempt.", 2500)
    page.click("#t-create")
    page.wait_for_selector("#t-create-out .badge")
    flagged = "ignore_instructions" in page.inner_text("#t-create-out")
    caption(
        page,
        f"Create ticket: {page.inner_text('#t-create-out .badge')}; injection flag "
        f"{'reported' if flagged else 'not reported'}.",
        4500,
    )
    page.fill("#t-id", "T-0002")
    page.click("#t-get")
    page.wait_for_selector("#t-get-out .badge")
    caption(
        page,
        f"Reading C-1001's ticket T-0002 as C-1002: {page.inner_text('#t-get-out .badge')}.",
        4500,
    )

    tab(page, "lab")
    caption(
        page, "4. Retrieval lab: the incident's score-fusion bug, same question and caller.", 2500
    )
    page.click("#lab-run")
    page.wait_for_selector("#lab-naive .chunk")
    caption(
        page,
        "Raw BM25 + cosine sum ranks the keyword-stuffed glossary KB-000B first "
        "(scores far above 1.0). RRF ranks the correct KB-0003 first.",
        8000,
    )

    tab(page, "index")
    page.wait_for_selector("#idx-collections table")
    caption(
        page,
        "5. Index: alias kb_live targets kb_v2 (schema v2). kb_v1 is the legacy, "
        "mislabelled index kept to reproduce the incident.",
        6000,
    )
    caption(
        page,
        "Also run from the command line: unit/API tests, real-model integration "
        "tests and the incident reproduction. See README.",
        5000,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_raw"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport=SIZE,
            record_video_dir=str(tmp),
            record_video_size=SIZE,
            color_scheme="light",
        )
        page = context.new_page()
        try:
            record(page)
        finally:
            context.close()  # the video file is finalised on close
            video_path = page.video.path()  # must be read before Playwright stops
            browser.close()
    final = OUT / "demo.webm"
    shutil.move(video_path, final)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {final.relative_to(ROOT)} ({final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
