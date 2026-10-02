"""Build phone-friendly PDFs of the viva notes: docs/VIVA_GUIDE.pdf, docs/VIVA_QA.pdf.

Needs `pip install markdown` and Google Chrome (used headless to print the HTML).
Run from the repo root:  python docs/build_pdf.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["VIVA_GUIDE.md", "VIVA_QA.md"]
CHROME = {
    "darwin": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "win32": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
}.get(sys.platform, "google-chrome")

# A5 pages with large text: readable on a phone without zooming.
CSS = """
@page { size: A5; margin: 12mm 10mm; }
body { font: 11.5pt/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1d1d1f; }
h1 { font-size: 20pt; border-bottom: 3px solid #2e7d32; padding-bottom: 4px; }
h2 { font-size: 15pt; color: #2e7d32; margin-top: 22px; break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 18px; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font: 9.5pt Menlo, Consolas, monospace; background: #f0f0f2; padding: 1px 4px; border-radius: 4px; }
pre { background: #1e1e1e; color: #e6e6e6; padding: 26px 10px 10px; border-radius: 8px;
      position: relative; white-space: pre-wrap; word-break: break-all; font-size: 8pt;
      line-height: 1.35; break-inside: avoid; }
pre::before { content: "\\25CF  \\25CF  \\25CF"; position: absolute; top: 6px; left: 10px;
              font-size: 8pt; color: #ff5f57; letter-spacing: 1px; }
pre code { background: none; color: inherit; padding: 0; font-size: 8pt; }
table { border-collapse: collapse; font-size: 9pt; margin: 8px 0; break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
th { background: #e8f5e9; }
img { max-width: 100%; border-radius: 6px; border: 1px solid #ddd; break-inside: avoid; }
p:has(> img) { text-align: center; margin: 10px 0 2px; }
.caption { font-size: 9pt; color: #555; text-align: center; margin: 0 0 12px; }
hr { border: 0; border-top: 1px solid #ddd; margin: 18px 0; }
"""


def build(name):
    md = (ROOT / name).read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    # alt text of each image doubles as its caption
    body = re.sub(r'<p>(<img alt="([^"]*)"[^>]*>)</p>', r'<p>\1</p><p class="caption">\2</p>', body)
    html = (f'<!doctype html><html><head><meta charset="utf-8"><base href="{ROOT.as_uri()}/">'
            f"<title>{name[:-3]}</title><style>{CSS}</style></head><body>{body}</body></html>")
    out = ROOT / "docs" / (name[:-3] + ".pdf")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={out}", Path(f.name).as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    for doc in DOCS:
        build(doc)
