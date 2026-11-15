"""Catalog → product → recommendation → question → CTA → lead (skip_llm)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.pipeline import run
from agent.store import cold_start
from eval.consultant_metrics import catalog_coverage

_PRODUCTS = [
    "Выход на оптовый рынок",
    "Облачная платформа учёта",
    "Солнечные электростанции",
    "Поиск коммерческих потерь",
    "Распознавание показаний по фотографии",
]


class ConsultFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cold_start(force=True)

    def test_full_presale_path(self) -> None:
        history: list[dict] = []

        r1 = run("расскажи о продуктах", history, skip_llm=True, skip_rerank=True)
        self.assertIn(r1.get("behavior"), ("answer", "partial"))
        self.assertGreaterEqual(
            catalog_coverage(r1.get("answer") or "", list(r1.get("products") or []), _PRODUCTS),
            0.99,
            msg=r1.get("answer"),
        )
        history += [
            {"role": "user", "content": "расскажи о продуктах"},
            {"role": "assistant", "content": r1["answer"]},
        ]

        r2 = run("подробнее про облачную платформу учёта", history, skip_llm=True, skip_rerank=True)
        ans2 = (r2.get("answer") or "").lower().replace("ё", "е")
        self.assertTrue(any(x in ans2 for x in ("учёт", "учет", "аскуэ", "показан", "платформ")), msg=r2.get("answer"))
        history += [
            {"role": "user", "content": "подробнее про облачную платформу учёта"},
            {"role": "assistant", "content": r2["answer"]},
        ]

        r3 = run(
            "хотим снизить счета за электричество на производстве что предложите",
            history,
            skip_llm=True,
            skip_rerank=True,
        )
        ans3 = (r3.get("answer") or "").lower().replace("ё", "е")
        self.assertTrue(any(x in ans3 for x in ("солнеч", "оптов", "учёт", "учет")), msg=r3.get("answer"))
        history += [
            {"role": "user", "content": "хотим снизить счета за электричество на производстве что предложите"},
            {"role": "assistant", "content": r3["answer"]},
        ]

        r4 = run("а фото показания можно сразу пускать в начисления?", history, skip_llm=True, skip_rerank=True)
        ans4 = (r4.get("answer") or "").lower().replace("ё", "е")
        self.assertTrue(any(x in ans4 for x in ("верифик", "начисл", "основан")), msg=r4.get("answer"))
        history += [
            {"role": "user", "content": "а фото показания можно сразу пускать в начисления?"},
            {"role": "assistant", "content": r4["answer"]},
        ]

        r5 = run("да, передайте менеджеру", history, skip_llm=True, skip_rerank=True)
        self.assertEqual(r5.get("behavior"), "lead")
        self.assertIn("ФИО", r5.get("answer") or "")
        history += [
            {"role": "user", "content": "да, передайте менеджеру"},
            {"role": "assistant", "content": r5["answer"]},
        ]

        r6 = run("Иван Петров", history, skip_llm=True, skip_rerank=True)
        self.assertEqual(r6.get("behavior"), "lead")
        history += [
            {"role": "user", "content": "Иван Петров"},
            {"role": "assistant", "content": r6["answer"]},
        ]
        r7 = run("+79991234567", history, skip_llm=True, skip_rerank=True)
        self.assertEqual(r7.get("behavior"), "lead")
        history += [
            {"role": "user", "content": "+79991234567"},
            {"role": "assistant", "content": r7["answer"]},
        ]
        r8 = run("ООО ТестЭнерго", history, skip_llm=True, skip_rerank=True)
        self.assertEqual(r8.get("behavior"), "lead")
        ans8 = r8.get("answer") or ""
        self.assertTrue(
            any(x in ans8.lower() for x in ("переда", "менеджер", "спасибо", "свяж")),
            msg=ans8,
        )


if __name__ == "__main__":
    unittest.main()
