"""Writer-style regressions: no handbook dump, catalog completeness, overlap."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


class WriterStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cold_start(force=True)

    def _run(self, q: str, history=None):
        return run(q, history or [], skip_llm=True, skip_rerank=True)

    def test_catalog_is_consultant_not_dump(self) -> None:
        out = self._run("расскажи о продуктах")
        ans = (out.get("answer") or "").lower().replace("ё", "е")
        for bad in _DUMP:
            self.assertNotIn(bad, ans, msg=out.get("answer"))
        cov = catalog_coverage(
            out.get("answer") or "",
            list(out.get("products") or []),
            [
                "Выход на оптовый рынок",
                "Облачная платформа учёта",
                "Солнечные электростанции",
                "Поиск коммерческих потерь",
                "Распознавание показаний по фотографии",
            ],
        )
        self.assertGreaterEqual(cov, 0.99, msg=out.get("answer"))
        self.assertGreaterEqual(directness_naturalness(out.get("answer") or ""), 0.7)

    def test_product_fact_not_section_header(self) -> None:
        out = self._run("что такое облачная платформа учёта")
        ans = (out.get("answer") or "").strip()
        low = ans.lower().replace("ё", "е")
        self.assertFalse(low.startswith("раздел:"))
        self.assertFalse(low.startswith("документ:"))
        self.assertFalse(low.startswith("назначение:"))
        self.assertTrue(
            any(x in low for x in ("учёт", "учет", "показан", "аскуэ", "платформ")),
            msg=ans,
        )

    def test_recommendation_names_options_not_threshold_table(self) -> None:
        out = self._run("хотим снизить счета за электричество на производстве что предложите")
        ans = (out.get("answer") or "").lower().replace("ё", "е")
        self.assertNotIn("таблица в разделе", ans)
        self.assertTrue(
            any(x in ans for x in ("солнеч", "оптов", "учёт", "учет", "категор")),
            msg=out.get("answer"),
        )

    def test_catalog_low_copy_overlap_vs_raw_chunk(self) -> None:
        from agent.store import get_store

        store = get_store()
        ch = store.get_chunk("20-portfolio.md#Промышленное предприятие")
        out = self._run("какие у вас решения")
        src = (ch.text_raw if ch else "") or ""
        overlap = long_copy_overlap(out.get("answer") or "", src, n=10)
        self.assertLess(overlap, 0.25, msg=f"overlap={overlap} answer={out.get('answer')}")


if __name__ == "__main__":
    unittest.main()
