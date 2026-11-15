"""Regressions for lead hijack + photo-meter relevance."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.guard import extract_explicit_name, looks_like_name
from agent.lead import lead_active_from_history
from agent.pipeline import run
from agent.slots import slots_from_history
from agent.store import cold_start
from agent.understand import fallback_understand
from agent.retrieve import retrieve
from agent.fuse import fuse


PHOTO_Q = "У нас тарифные счетчики. Сервис разберет показания по фото?"
ENERGY_Q = "Что такое Energytool"
PHONE_TMPL = "Оставьте, пожалуйста, номер телефона для связи."
FORBIDDEN = ("сетевых и сбытовых", "зона ответственности", "кто отвечает за")


class LeadAndNameTests(unittest.TestCase):
    def test_energytool_not_a_name(self) -> None:
        self.assertFalse(looks_like_name(ENERGY_Q))
        self.assertIsNone(extract_explicit_name(ENERGY_Q))

    def test_explicit_name_ok(self) -> None:
        self.assertEqual(extract_explicit_name("Меня зовут Иван Петров"), "Иван Петров")
        self.assertTrue(looks_like_name("Меня зовут Иван Петров"))

    def test_slots_do_not_mine_question_as_name(self) -> None:
        history = [{"role": "user", "content": ENERGY_Q}]
        slots = slots_from_history(history)
        self.assertNotIn("name", slots)
        self.assertFalse(lead_active_from_history(history, slots))


class IntentTests(unittest.TestCase):
    def test_photo_meter_not_pricing(self) -> None:
        u = fallback_understand(PHOTO_Q)
        self.assertEqual(u.route, "consult")
        self.assertEqual(u.question_kind, "fact")
        self.assertEqual(u.intent, "product_consult")
        self.assertEqual(u.aspects[0].attr, "наличие")

    def test_energytool_product(self) -> None:
        u = fallback_understand(ENERGY_Q)
        self.assertEqual(u.route, "consult")
        self.assertEqual(u.intent, "product_consult")

    def test_catalog_and_product_fact_intents(self) -> None:
        cat = fallback_understand("расскажи о продуктах")
        self.assertEqual(cat.consult_intent, "catalog")
        pf = fallback_understand("что такое облачная платформа учёта")
        self.assertEqual(pf.consult_intent, "product_fact")
        self.assertEqual(pf.explicit_product, "Облачная платформа учёта")


class MultiTurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cold_start()

    def test_q2_q3_not_phone_after_energytool(self) -> None:
        r1 = run(ENERGY_Q, [], skip_llm=True, skip_rerank=True)
        self.assertNotEqual(r1.get("behavior"), "lead")
        self.assertNotEqual((r1.get("answer") or "").strip(), PHONE_TMPL)
        history = [
            {"role": "user", "content": ENERGY_Q},
            {"role": "assistant", "content": r1["answer"]},
        ]
        r2 = run(PHOTO_Q, history, skip_llm=True, skip_rerank=True)
        self.assertNotEqual(r2.get("behavior"), "lead")
        self.assertNotEqual((r2.get("answer") or "").strip(), PHONE_TMPL)
        history2 = history + [
            {"role": "user", "content": PHOTO_Q},
            {"role": "assistant", "content": r2["answer"]},
        ]
        r3 = run(ENERGY_Q, history2, skip_llm=True, skip_rerank=True)
        self.assertNotEqual(r3.get("behavior"), "lead")
        self.assertNotEqual((r3.get("answer") or "").strip(), PHONE_TMPL)
        ans = (r3.get("answer") or "").lower()
        self.assertTrue(
            "energytool" in ans or "энерджи" in ans or "мтс" in ans or "учёт" in ans or "учет" in ans,
            msg=r3.get("answer"),
        )

    def test_explicit_lead_still_asks_name(self) -> None:
        out = run("свяжите с менеджером", [], skip_llm=True, skip_rerank=True)
        self.assertEqual(out.get("behavior"), "lead")
        self.assertIn("ФИО", out.get("answer") or "")


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = cold_start()

    def test_photo_query_hits_meter_reading(self) -> None:
        u = fallback_understand(PHOTO_Q)
        branches = retrieve(self.store, u)
        ranked = fuse(self.store, branches)
        ids = [c.chunk_id for c in ranked[:8]]
        self.assertTrue(
            any("25-meter-reading" in cid for cid in ids),
            msg=f"top ids={ids}",
        )
        # Must not be only the responsibility FAQ
        if ids:
            self.assertFalse(
                ids[0].startswith("17-faq.md#Кто отвечает"),
                msg=f"bad top1={ids[0]}",
            )

    def test_pipeline_photo_answer_relevant(self) -> None:
        out = run(PHOTO_Q, [], skip_llm=True, skip_rerank=True)
        answer = (out.get("answer") or "").lower().replace("ё", "е")
        cites = out.get("citations") or []
        for bad in FORBIDDEN:
            self.assertNotIn(bad, answer)
        self.assertFalse(
            len(cites) == 1 and str(cites[0]).startswith("17-faq.md#Кто отвечает"),
            msg=f"citations={cites} answer={out.get('answer')}",
        )
        ok = (
            "распозна" in answer
            or "фото" in answer
            or "тариф" in answer
            or "показан" in answer
            or out.get("behavior") == "refuse"
            or any("25-meter-reading" in str(c) for c in cites)
        )
        self.assertTrue(ok, msg=f"behavior={out.get('behavior')} cites={cites} answer={out.get('answer')}")


if __name__ == "__main__":
    unittest.main()
