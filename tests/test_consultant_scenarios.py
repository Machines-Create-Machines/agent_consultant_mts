"""Scenario suite for Consultant v2 (no LLM required)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.pipeline import run
from agent.store import cold_start

SCENARIOS = ROOT / "tests" / "consultant_scenarios.jsonl"


class ConsultantScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cold_start(force=True)
        rows = []
        for line in SCENARIOS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        cls.rows = rows

    def test_all_scenarios(self) -> None:
        failures: list[str] = []
        for row in self.rows:
            out = run(
                row["question"],
                row.get("history") or [],
                skip_llm=True,
                skip_rerank=True,
            )
            beh = out.get("behavior") or ""
            ans = (out.get("answer") or "").lower().replace("ё", "е")
            cites = [str(c) for c in (out.get("citations") or [])]
            expected = set(row.get("expect_behavior") or [])
            if expected and beh not in expected:
                failures.append(f"{row['id']}: behavior={beh} expected={expected}")
            must = row.get("must_include_any") or []
            if must and not any(m.lower().replace("ё", "е") in ans for m in must):
                failures.append(f"{row['id']}: missing any of {must}; answer={out.get('answer')!r}")
            for bad in row.get("forbid") or []:
                if bad.lower().replace("ё", "е") in ans:
                    failures.append(f"{row['id']}: forbidden {bad!r}")
            gold_any = row.get("gold_chunks_any") or []
            if gold_any and cites and not any(
                any(g in c for g in gold_any) for c in cites
            ):
                # soft: retrieval may still put gold in retrieved_ids
                retrieved = [str(c) for c in (out.get("retrieved_ids") or [])]
                if not any(any(g in c for g in gold_any) for c in retrieved[:8]):
                    failures.append(
                        f"{row['id']}: gold not in cites/retrieved; cites={cites} top={retrieved[:5]}"
                    )
            cover = row.get("must_cover_products") or []
            if cover:
                from eval.consultant_metrics import catalog_coverage

                cov = catalog_coverage(
                    out.get("answer") or "",
                    list(out.get("products") or []),
                    cover,
                )
                if cov < 0.8:
                    failures.append(
                        f"{row['id']}: catalog_coverage={cov:.2f} products={out.get('products')} answer={out.get('answer')!r}"
                    )
        self.assertFalse(failures, msg="\n".join(failures))


if __name__ == "__main__":
    unittest.main()
