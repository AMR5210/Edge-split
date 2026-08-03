from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path
import re


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.anchors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if "id" in values and values["id"]:
            self.ids.append(values["id"])
        if tag == "a" and values.get("href", "").startswith("#"):
            self.anchors.append(values["href"] or "")


class StaticSiteTests(unittest.TestCase):
    def test_required_sections_are_ordered_and_linked(self) -> None:
        html = Path(__file__).with_name("static") / "index.html"
        parser = SiteParser()
        parser.feed(html.read_text(encoding="utf-8"))
        expected = [
            "overview", "architecture", "live-demo", "v1-v2",
            "results", "challenges", "summary",
        ]
        self.assertEqual(expected, [item for item in parser.ids if item in expected])
        self.assertEqual([f"#{item}" for item in expected], parser.anchors[1:1 + len(expected)])
        self.assertEqual(len(parser.ids), len(set(parser.ids)))

    def test_live_demo_retains_the_existing_control_surface(self) -> None:
        html = (Path(__file__).with_name("static") / "index.html").read_text(encoding="utf-8")
        for element_id in (
            "prompt", "run-button", "response", "metric-ttft",
            "laptop-terminal", "phone-terminal", "history-table", "run-history",
        ):
            self.assertIn(f'id="{element_id}"', html)

    def test_architecture_is_a_connected_static_svg_without_mermaid_source(self) -> None:
        html = (Path(__file__).with_name("static") / "index.html").read_text(encoding="utf-8")
        self.assertIn('class="static-architecture"', html)
        self.assertNotIn("flowchart LR", html)
        self.assertNotIn("diagram-source", html)
        for label in (
            "Caller", "FastAPI router", "WSL2 CUDA llama-server", "prefill",
            "V1: slot file", "HTTP state receiver", "V2: raw sequence state",
            "TCP sender/listener", "Android llama-server", "decode stream",
            "back to router",
        ):
            self.assertIn(label, html)
        self.assertGreaterEqual(html.count('class="architecture-wire"'), 6)

    def test_overview_has_requested_two_column_content_and_preview(self) -> None:
        root = Path(__file__).with_name("static")
        html = (root / "index.html").read_text(encoding="utf-8")
        css = (root / "style.css").read_text(encoding="utf-8")
        self.assertIn('class="overview-grid"', html)
        self.assertNotIn("compact-pipeline", html)
        self.assertNotIn("preview-return-arrow", html)
        self.assertNotIn("C88 31 70 31 62 8", html)
        for text in (
            "Split LLM inference across the", "hardware you already own",
            "Prefill needs your laptop's GPU, decode doesn't.",
            "Laptop Prefill", "Wi-Fi State Transfer", "Phone Decode",
            'href="#architecture"', 'href="#live-demo"',
            "WSL2 CUDA llama-server", "Android llama-server",
            "Wi-Fi State Transfer</strong>: Sequence state handed to the phone over your LAN, with a SHA-256 integrity check on every frame.",
            "Two phases. Two devices. One request.",
        ):
            self.assertIn(text, html)
        self.assertLess(
            html.index("LLMs have two distinct phases:"),
            html.index("Prefill needs your laptop's GPU, decode doesn't."),
        )
        self.assertIn(".overview-grid { display: grid; grid-template-columns:", css)
        self.assertIn("@media (max-width: 1100px) { .overview-grid { grid-template-columns: 1fr;", css)

    def test_light_analytics_theme_keeps_terminal_panels_dark(self) -> None:
        root = Path(__file__).with_name("static")
        html = (root / "index.html").read_text(encoding="utf-8")
        css = (root / "style.css").read_text(encoding="utf-8")
        self.assertIn('color-scheme" content="light"', html)
        self.assertIn("family=Inter:wght@400;500;600", html)
        self.assertIn("family=JetBrains+Mono:wght@400", html)
        for token in (
            "#F6F8FC", "#FFFFFF", "#0F172A", "#2563EB", "#7C3AED",
            "#DC2626", "#EA580C", "#16A34A", "#E2E8F0",
        ):
            self.assertIn(token, css)
        self.assertNotIn("#0d1117", css.lower())
        self.assertNotRegex(css, r"\b(?:700|800|900)\b")
        self.assertEqual({"3px", "6px", "8px", "999px"}, set(re.findall(r"border-radius: ([0-9]+px)", css)))
        self.assertIn("box-shadow: var(--card-shadow)", css)
        self.assertIn(".terminal pre", css)
        self.assertIn("background: var(--navy)", css)
        self.assertIn(".section { display: flex; min-height: 100vh; flex-direction: column; justify-content: center;", css)
        for class_name in ("metric-blue", "metric-purple", "metric-red", "metric-orange"):
            self.assertIn(class_name, css)

    def test_generated_response_uses_safe_markdown_renderer(self) -> None:
        root = Path(__file__).with_name("static")
        app = (root / "app.js").read_text(encoding="utf-8")
        css = (root / "style.css").read_text(encoding="utf-8")

        self.assertIn("function renderMarkdownResponse", app)
        self.assertIn("function appendInlineMarkdown", app)
        self.assertIn("renderMarkdownResponse(result.content)", app)
        self.assertIn("document.createElement", app)
        self.assertIn("strong.textContent", app)
        self.assertIn(".response.markdown-response", css)

    def test_terminal_mirrors_have_persistent_clear_controls(self) -> None:
        root = Path(__file__).with_name("static")
        html = (root / "index.html").read_text(encoding="utf-8")
        app = (root / "app.js").read_text(encoding="utf-8")
        self.assertIn('data-terminal-source="laptop"', html)
        self.assertIn('data-terminal-source="phone"', html)
        self.assertIn('class="terminal-clear"', html)
        self.assertIn('api("/api/events/clear"', app)
        self.assertIn("terminalFor(source).textContent = \"\"", app)

    def test_compact_result_headers_have_muted_unit_legends(self) -> None:
        root = Path(__file__).with_name("static")
        html = (root / "index.html").read_text(encoding="utf-8")
        css = (root / "style.css").read_text(encoding="utf-8")
        self.assertIn("TTFT = Time to First Token (s)", html)
        self.assertIn("GPU = laptop GPU power draw (W)", html)
        self.assertNotIn("LAPTOP GPU DRAW, V1 / V2", html)
        self.assertGreaterEqual(html.count("<th>V1 GPU</th><th>V2 GPU</th>"), 3)
        self.assertIn(".table-legend", css)

    def test_summary_states_scope_without_authorship_claims(self) -> None:
        html = (Path(__file__).with_name("static") / "index.html").read_text(encoding="utf-8")
        for claim in (
            "Prefix caching remains optional follow-up work.",
            "is not part of the V1/V2",
            "not a production distributed inference service",
        ):
            self.assertIn(claim, html)
        for removed in ("AI USAGE", "Codex", "GPT-5.6", "The builder independently"):
            self.assertNotIn(removed, html)

    def test_public_dashboard_uses_generic_phone_label(self) -> None:
        html = (Path(__file__).with_name("static") / "index.html").read_text(encoding="utf-8")
        self.assertIn("controlled phone comparison", html)
        self.assertIn("NEON accumulator comparison: Phone", html)
        self.assertIn("compatibility limitation is documented in the README", html)

    def test_live_demo_keeps_a_browser_session_run_history(self) -> None:
        root = Path(__file__).with_name("static")
        html = (root / "index.html").read_text(encoding="utf-8")
        app = (root / "app.js").read_text(encoding="utf-8")
        self.assertIn("THIS BROWSER SESSION", html)
        self.assertIn('id="run-history"', html)
        self.assertIn("Last 8 successful requests", html)
        self.assertIn("runHistory: []", app)
        self.assertIn("function addRunHistory", app)
        self.assertIn("state.runHistory = state.runHistory.slice(0, 8)", app)
        self.assertIn("addRunHistory(payload)", app)

    def test_results_note_the_llama_gpu_stability_rerun(self) -> None:
        repo_root = Path(__file__).parent.parent
        root = Path(__file__).with_name("static")
        html = (root / "index.html").read_text(encoding="utf-8")
        benchmarks = repo_root / "docs" / "BENCHMARKS.md"
        result_note = repo_root / "results" / "llama32_v1_v2_phone2_comparison.md"
        self.assertIn("GPU stability rerun", html)
        self.assertIn("current templated workload", html)
        self.assertIn("<td>2.62 +/- 0.55 W</td><td>2.16 +/- 0.32 W</td>", html)
        self.assertIn("initial high-variance window did not reproduce", html)
        self.assertIn("75.1% (V1) and 69.1% (V2)", benchmarks.read_text(encoding="utf-8"))
        self.assertIn("workload-identical power replication", result_note.read_text(encoding="utf-8"))
        self.assertIn("Stability rerun", result_note.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
