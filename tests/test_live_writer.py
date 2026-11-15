"""Live/mock writer suite. Skips when the LLM endpoint is unavailable."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm import load_dotenv, model_available
from agent.pipeline import run
from agent.store import cold_start
from eval.consultant_metrics import catalog_coverage, directness_naturalness, long_copy_overlap

_DUMP = (
    "таблица в разделе",
    "основной вопрос -",
    "документ:",
    "назначение:",
    "раздел:",
    "согласно разделу",
)

_PRODUCTS = [
    "Выход на оптовый рынок",
    "Облачная платформа учёта",
    "Солнечные электростанции",
    "Поиск коммерческих потерь",
    "Распознавание показаний по фотографии",
]


def _llm_ready() -> bool:
    load_dotenv()
    if os.environ.get("SKIP_LIVE_WRITER") == "1":
        return False
    return model_available()


@unittest.skipUnless(_llm_ready(), "LLM endpoint is not configured")
class LiveWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cold_start(force=True)

    def _run(self, q: str, history=None):
        return run(q, history or [], skip_llm=False, skip_rerank=False)

    def test_catalog_live_covers_all_and_sounds_human(self) -> None:
        out = self._run("расскажи о продуктах")
        ans = out.get("answer") or ""
        low = ans.lower().replace("ё", "е")
        for bad in _DUMP:
            self.assertNotIn(bad, low, msg=ans)
        cov = catalog_coverage(ans, list(out.get("products") or []), _PRODUCTS)
        self.assertGreaterEqual(cov, 0.99, msg=ans)
        self.assertGreaterEqual(directness_naturalness(ans), 0.7)
        self.assertLessEqual(int(out.get("llm_calls") or 0), 3)

    def test_product_fact_live_is_direct(self) -> None:
        out = self._run("что такое облачная платформа учёта")
        ans = (out.get("answer") or "").strip()
        low = ans.lower().replace("ё", "е")
        self.assertFalse(low.startswith("раздел:"))
        self.assertTrue(any(x in low for x in ("учёт", "учет", "показан", "аскуэ", "платформ")), msg=ans)
        self.assertGreaterEqual(directness_naturalness(ans), 0.6)

    def test_recommendation_live_names_options(self) -> None:
        out = self._run("хотим снизить счета за электричество на производстве что предложите")
        ans = (out.get("answer") or "").lower().replace("ё", "е")
        self.assertNotIn("таблица в разделе", ans)
        self.assertTrue(
            any(x in ans for x in ("солнеч", "оптов", "учёт", "учет", "категор")),
            msg=out.get("answer"),
        )


class MockWriterOverlapTests(unittest.TestCase):
    """Deterministic writer path still beats extractive dump on overlap."""

    @classmethod
    def setUpClass(cls) -> None:
        cold_start(force=True)

    def test_catalog_overlap_vs_threshold_chunk(self) -> None:
        from agent.store import get_store

        store = get_store()
        ch = store.get_chunk("20-portfolio.md#Промышленное предприятие")
        out = run("какие у вас решения", [], skip_llm=True, skip_rerank=True)
        src = (ch.text_raw if ch else "") or ""
        overlap = long_copy_overlap(out.get("answer") or "", src, n=10)
        self.assertLess(overlap, 0.2, msg=out.get("answer"))


if __name__ == "__main__":
    unittest.main()
