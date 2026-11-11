import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app


ROOT = Path(__file__).resolve().parents[1]


class ComplianceAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads((ROOT / "sample_cases.json").read_text(encoding="utf-8"))
        cls.samples = payload["cases"] if isinstance(payload, dict) else payload
        cls.by_name = {item["name"]: app.normalize_case(item) for item in cls.samples}

    def test_rule_catalog_is_complete_and_bounded(self):
        self.assertEqual(17, len(app.RULES))
        self.assertEqual(100, sum(rule.points for rule in app.RULES))
        self.assertEqual(len(app.RULES), len({rule.rule_id for rule in app.RULES}))

    def test_high_risk_credit_case(self):
        result = app.evaluate_case(self.by_name["智能信贷预审助手"])
        self.assertEqual(100, result["score"])
        self.assertEqual("极高", result["level"])
        self.assertEqual(17, result["trigger_count"])

    def test_low_risk_internal_qa_case(self):
        result = app.evaluate_case(self.by_name["内部合规知识问答助手"])
        self.assertEqual(0, result["score"])
        self.assertEqual("低", result["level"])
        self.assertEqual(0, result["trigger_count"])

    def test_export_signatures_and_disclaimer(self):
        case = self.by_name["智能信贷预审助手"]
        result = app.evaluate_case(case)
        html_report = app.html_report(case, result)
        csv_report = app.csv_report(result)
        pdf_report = app.english_pdf_summary(case, result)
        self.assertTrue(html_report.startswith(b"<!doctype html>"))
        self.assertIn(app.DISCLAIMER_SHORT.encode("utf-8"), html_report)
        self.assertTrue(csv_report.startswith(b"\xef\xbb\xbf"))
        self.assertTrue(pdf_report.startswith(b"%PDF"))

    def test_deepseek_payload_is_minimized(self):
        case = self.by_name["智能信贷预审助手"]
        result = app.evaluate_case(case)
        payload = app.deepseek_payload(result)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(case["name"], serialized)
        self.assertNotIn(case["description"], serialized)
        self.assertNotIn("api_key", serialized.lower())

    def test_deepseek_openai_compatible_request(self):
        case = self.by_name["智能信贷预审助手"]
        result = app.evaluate_case(case)
        fake_response = Mock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {
            "choices": [{"message": {"content": "P0：先阻断可识别客户信息进入外部模型。"}}]
        }
        with patch("app.requests.post", return_value=fake_response) as mocked_post:
            content = app.call_deepseek("test-key", "deepseek-flash", result)
        self.assertIn("P0", content)
        _, kwargs = mocked_post.call_args
        self.assertEqual(f"{app.DEEPSEEK_BASE_URL}/chat/completions", mocked_post.call_args.args[0])
        self.assertEqual("deepseek-flash", kwargs["json"]["model"])
        self.assertNotIn(case["name"], json.dumps(kwargs["json"], ensure_ascii=False))
        self.assertNotIn(case["description"], json.dumps(kwargs["json"], ensure_ascii=False))
        self.assertEqual("Bearer test-key", kwargs["headers"]["Authorization"])

    def test_result_fingerprint_distinguishes_equal_scores(self):
        first = {"triggered": [{"规则ID": "R01", "分值": 4, "事实判断": "A"}]}
        second = {"triggered": [{"规则ID": "R10", "分值": 4, "事实判断": "B"}]}
        self.assertNotEqual(app.result_fingerprint(first), app.result_fingerprint(second))


if __name__ == "__main__":
    unittest.main()
