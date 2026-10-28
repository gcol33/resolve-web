"""Run the site's prediction path in a real browser and time it.

    python tests/browser_bench.py <suite dir> <species.csv> <header.csv> <out.csv> \
        [--ep wasm|webgpu] [--threads N] [--targets a,b] [--n N] [--channel msedge]

Serves this repository, the suite and the two tables from one local origin
with the cross-origin isolation headers, opens tests/browser_bench.html in
headless Chromium (Edge by default) through Playwright, prints the seconds
per target and writes the predictions as run_suite.mjs does, for
compare_engine.py.

    python tests/browser_bench.py --site <models dir>

instead opens the page itself with `?models=` pointing at a local models
directory (a suite per subdirectory and taxonomy.json), served without the
isolation headers, and checks that its service worker makes the page
cross-origin isolated and that the example plot predicts.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
BROWSER_ARGS = ["--enable-unsafe-webgpu"]


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream"}

    def __init__(self, *args, roots: dict[str, Path], isolate: bool, **kwargs):
        self.roots, self.isolate = roots, isolate
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        parts = unquote(urlsplit(path).path).lstrip("/").split("/", 1)
        root = self.roots.get(parts[0])
        if root is None:
            return str(REPO / "_missing_")
        if root.is_file():
            return str(root)
        return str(root / (parts[1] if len(parts) > 1 else ""))

    def end_headers(self) -> None:
        if self.isolate:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args) -> None:
        pass


def serve(roots: dict[str, Path], isolate: bool) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, roots=roots, isolate=isolate))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def bench(args) -> int:
    roots = {"tests": REPO / "tests", "site": REPO / "site", "suite": args.suite.resolve(),
             "species.csv": args.species.resolve(), "header.csv": args.header.resolve()}
    server, base = serve(roots, isolate=True)
    query = (f"suite=/suite&species=/species.csv&header=/header.csv&ep={args.ep}"
             f"&threads={args.threads}&targets={args.targets}&n={args.n}" + (f"&budget={args.budget}" if args.budget else ""))
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True, args=BROWSER_ARGS)
        page = browser.new_page()
        page.on("console", lambda m: m.type in ("error", "warning") and print(f"[{m.type}] {m.text}", file=sys.stderr))
        page.goto(f"{base}/tests/browser_bench.html?{query}")
        page.wait_for_function("window.bench !== undefined", timeout=args.timeout * 1000)
        result = page.evaluate("window.bench")
        browser.close()
    server.shutdown()
    if not result["ok"]:
        print(result["error"], file=sys.stderr)
        return 1
    print(f"{args.ep}, {result['threads']} thread(s), isolated {result['isolated']}"
          + (f", adapter {result['adapter']}" if result["adapter"] else ""))
    for name, s in result["seconds"].items():
        print(f"{name}: {result['n']} plots in {s:.1f} s")
    args.out.write_text(result["csv"], encoding="utf-8")
    return 0


def site(args) -> int:
    roots = {"site": REPO / "site", "models": args.site.resolve()}
    server, base = serve(roots, isolate=False)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True, args=BROWSER_ARGS)
        page = browser.new_page()
        page.on("console", lambda m: m.type == "error" and print(f"[error] {m.text}", file=sys.stderr))
        page.goto(f"{base}/site/index.html?models=/models")
        page.wait_for_function("self.crossOriginIsolated === true", timeout=30_000)
        page.click("#run-paste")
        page.wait_for_function("document.querySelectorAll('#preview tr').length > 1"
                               " || document.getElementById('status').textContent.startsWith('Stopped')",
                               timeout=args.timeout * 1000)
        status = page.text_content("#status")
        threads = page.evaluate("globalThis.ort.env.wasm.numThreads")
        gpu = page.evaluate("Boolean(navigator.gpu)")
        rows = page.eval_on_selector_all("#preview tr", "rs => rs.map(r => r.innerText.replace(/\\t/g, ' | '))")
        browser.close()
    server.shutdown()
    print(f"cross-origin isolated under the service worker, {threads} thread(s), WebGPU offered {gpu}")
    print("\n".join(rows))
    if status.startswith("Stopped"):
        print(status, file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path, nargs="?")
    ap.add_argument("species", type=Path, nargs="?")
    ap.add_argument("header", type=Path, nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--site", type=Path)
    ap.add_argument("--ep", choices=["wasm", "webgpu"], default="wasm")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--targets", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0, help="padded positions per batch (predictTarget tokenBudget)")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the predictions")
    args = ap.parse_args()
    if args.site:
        return site(args)
    if not all([args.suite, args.species, args.header, args.out]):
        ap.error("suite, species, header and out are required without --site")
    return bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
