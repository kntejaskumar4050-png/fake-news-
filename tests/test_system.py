"""
End-to-end tests for the Fake News Detection System.

Run from the project root (models must be trained first):

    python -m unittest discover -s tests -v

The suite uses a temporary SQLite database so your real prediction history is
never touched.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Point the app at a throwaway database and skip model warm-up during tests.
_TMP_DIR = tempfile.mkdtemp(prefix="fnd_tests_")
os.environ["FND_DATABASE"] = str(Path(_TMP_DIR) / "test_predictions.db")
os.environ["FND_WARMUP"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config  # noqa: E402
import database  # noqa: E402
from app import create_app  # noqa: E402
from predict import Predictor  # noqa: E402
from preprocess import TextPreprocessor, clean_text  # noqa: E402

MODELS_READY = config.BILSTM_MODEL_PATH.exists() and config.TOKENIZER_PATH.exists()

REAL_SAMPLE = (
    "WASHINGTON (Reuters) - The U.S. Senate on Tuesday approved a bipartisan "
    "infrastructure package that would fund road, bridge and broadband projects "
    "over the next five years. The measure passed 69-30 after weeks of negotiation "
    "between lawmakers from both parties."
)
FAKE_SAMPLE = (
    "Doctors are completely STUNNED! A secret kitchen ingredient that Big Pharma "
    "has been hiding for decades reportedly cures every disease in just 24 hours. "
    "The mainstream media refuses to report this shocking truth. Share before it "
    "is removed forever!!!"
)


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_removes_html_and_urls(self):
        dirty = '<p>BREAKING: Visit <a href="https://example.com">https://example.com</a> &amp; read!</p>'
        cleaned = clean_text(dirty)
        self.assertNotIn("<", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("&amp", cleaned)
        self.assertTrue(all(token.isalpha() for token in cleaned.split()))

    def test_cleaning_lowercases_and_removes_stopwords(self):
        cleaned = clean_text("The QUICK brown fox is jumping over the lazy dog")
        self.assertEqual(cleaned, cleaned.lower())
        self.assertNotIn("the", cleaned.split())
        self.assertNotIn("over", cleaned.split())

    def test_negations_are_preserved(self):
        cleaned = clean_text("The report did not confirm the claim and never named a source")
        self.assertIn("not", cleaned.split())
        self.assertIn("never", cleaned.split())

    def test_nan_and_empty_input(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")
        self.assertEqual(clean_text(""), "")

    def test_cleaning_removes_digits_and_punctuation(self):
        self.assertEqual(clean_text("Prices rose 12% in 2019!!!"), "prices rose")

    @unittest.skipUnless(MODELS_READY, "tokenizer not trained yet")
    def test_tokenizer_roundtrip_and_shape(self):
        pre = TextPreprocessor.load()
        padded = pre.transform_text("A short test article about the economy and jobs")
        self.assertEqual(padded.shape, (1, pre.max_len))
        self.assertEqual(str(padded.dtype), "int32")

    def test_tokenizer_fit_is_deterministic(self):
        texts = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"]
        a = TextPreprocessor(max_len=8, max_words=50).fit(texts)
        b = TextPreprocessor(max_len=8, max_words=50).fit(texts)
        self.assertEqual(a.word_index, b.word_index)


@unittest.skipUnless(MODELS_READY, "trained models are required")
class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor().load()

    def test_models_loaded(self):
        self.assertTrue(self.predictor.loaded)
        self.assertIn("lstm", self.predictor.available_models)
        self.assertIn("bilstm", self.predictor.available_models)

    def test_prediction_shape(self):
        result = self.predictor.predict(REAL_SAMPLE, "bilstm")
        self.assertIn(result["prediction"], ("FAKE", "REAL"))
        self.assertGreaterEqual(result["confidence"], 0.5)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertAlmostEqual(
            result["probability_fake"] + result["probability_real"], 1.0, places=3
        )
        self.assertEqual(len(result["model_scores"]), 2)
        self.assertLessEqual(result["confidence_percent"], 99.9)
        self.assertEqual(result["text_stats"]["max_len"], self.predictor._preprocessor.max_len)
        self.assertIn("not a fact-check", result["disclaimer"])

    def test_display_confidence_never_reaches_100(self):
        # A saturated sigmoid must not be presented as absolute certainty.
        explanation = self.predictor._explain("REAL", 1.0, "bilstm", 1.0, tokens=200)
        self.assertIn("99.9%", explanation)
        self.assertNotIn("100.0%", explanation)

    def test_all_model_options(self):
        for model in ("lstm", "bilstm", "ensemble"):
            result = self.predictor.predict(REAL_SAMPLE, model)
            self.assertEqual(result["model_key"], model)

    def test_special_characters_and_long_input(self):
        symbols_only = "!!! @@@ ### $$$ %%% ^^^ &&& *** ((( ))) … “” — 12345 67890"
        with self.assertRaises(ValueError):
            self.predictor.predict(symbols_only, "bilstm")  # no usable words
        # Special characters mixed with real words must still classify cleanly.
        mixed = "“Breaking” — the @Senate $$$ approved the bill!!! (Reuters) 2024."
        mixed_result = self.predictor.predict(mixed, "bilstm")
        self.assertIn(mixed_result["prediction"], ("FAKE", "REAL"))
        result = self.predictor.predict(REAL_SAMPLE + " " + "and more context " * 800, "bilstm")
        self.assertTrue(result["text_stats"]["truncated"])

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict(REAL_SAMPLE, "transformer")

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("   ", "bilstm")

    def test_metrics_are_real(self):
        summary = self.predictor.model_summary()
        self.assertIn("bilstm", summary["models"])
        accuracy = summary["models"]["bilstm"]["accuracy"]
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy, 0.5)


@unittest.skipUnless(MODELS_READY, "trained models are required")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()
        database.init_db()

    def setUp(self):
        database.clear_history()

    # ---- pages -------------------------------------------------------- #
    def test_pages_render(self):
        for path, marker in [
            ("/", b"Know what you"),
            ("/detect", b"Analyse a news article"),
            ("/dashboard", b"Prediction dashboard"),
            ("/about", b"About the project"),
        ]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(marker, response.data, path)

    def test_unknown_page_returns_html_404(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Page not found", response.data)

    def test_api_unknown_returns_json_404(self):
        response = self.client.get("/api/nope")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["status"], 404)

    # ---- prediction --------------------------------------------------- #
    def test_predict_valid(self):
        response = self.client.post("/predict", json={"text": FAKE_SAMPLE, "model": "bilstm"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], ("FAKE", "REAL"))
        self.assertTrue(data["stored"])
        self.assertIsInstance(data["id"], int)
        self.assertEqual(len(data["model_scores"]), 2)
        self.assertLessEqual(data["confidence_percent"], 99.9)
        self.assertIn("explanation", data)
        self.assertNotIn("100.0%", data["explanation"])
        self.assertIn("disclaimer", data)

    def test_predict_persists_to_sqlite(self):
        response = self.client.post("/predict", json={"text": REAL_SAMPLE, "model": "lstm"})
        prediction_id = response.get_json()["id"]
        rows = database.fetch_history(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], prediction_id)
        detail = self.client.get(f"/api/predictions/{prediction_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["prediction"]["label"], rows[0]["label"])

    def test_predict_ensemble(self):
        response = self.client.post("/predict", json={"text": REAL_SAMPLE, "model": "ensemble"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model_key"], "ensemble")

    def test_empty_input_rejected(self):
        response = self.client.post("/predict", json={"text": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_short_input_rejected(self):
        response = self.client.post("/predict", json={"text": "Too short"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("too short", response.get_json()["error"].lower())

    def test_missing_text_rejected(self):
        response = self.client.post("/predict", json={"model": "lstm"})
        self.assertEqual(response.status_code, 400)

    def test_non_string_text_rejected(self):
        response = self.client.post("/predict", json={"text": 12345})
        self.assertEqual(response.status_code, 400)

    def test_invalid_model_rejected(self):
        response = self.client.post("/predict", json={"text": REAL_SAMPLE, "model": "svm"})
        self.assertEqual(response.status_code, 400)

    def test_oversized_input_rejected(self):
        response = self.client.post("/predict", json={"text": "word " * 6000})
        self.assertEqual(response.status_code, 413)

    def test_invalid_json_rejected(self):
        response = self.client.post("/predict", data="not json",
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_form_encoded_input_supported(self):
        response = self.client.post("/predict", data={"text": REAL_SAMPLE, "model": "bilstm"})
        self.assertEqual(response.status_code, 200)

    # ---- dashboard / history ------------------------------------------ #
    def test_dashboard_statistics_reflect_predictions(self):
        for text in (FAKE_SAMPLE, FAKE_SAMPLE, REAL_SAMPLE):
            self.client.post("/predict", json={"text": text, "model": "bilstm"})
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 200)
        stats = response.get_json()
        self.assertEqual(stats["total_predictions"], 3)
        self.assertEqual(stats["fake_predictions"] + stats["real_predictions"], 3)
        self.assertEqual(len(stats["trend"]), 7)
        self.assertIn("model_performance", stats)
        self.assertGreater(stats["average_confidence"], 0.5)

    def test_dashboard_accepts_json_negotiation(self):
        response = self.client.get("/dashboard", headers={"Accept": "application/json"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["total_predictions"], 0)

    def test_history_limit_and_filter(self):
        self.client.post("/predict", json={"text": FAKE_SAMPLE})
        self.client.post("/predict", json={"text": REAL_SAMPLE})
        response = self.client.get("/api/history?limit=1")
        self.assertEqual(len(response.get_json()["predictions"]), 1)
        response = self.client.get("/api/history?limit=10&label=FAKE")
        rows = response.get_json()["predictions"]
        self.assertTrue(all(row["label"] == "FAKE" for row in rows))

    def test_history_validation(self):
        self.assertEqual(self.client.get("/api/history?limit=0").status_code, 400)
        self.assertEqual(self.client.get("/api/history?limit=abc").status_code, 400)
        self.assertEqual(self.client.get("/api/history?label=MAYBE").status_code, 400)

    def test_clear_history(self):
        self.client.post("/predict", json={"text": REAL_SAMPLE})
        response = self.client.delete("/api/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["deleted"], 1)
        self.assertEqual(database.fetch_history(limit=5), [])

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["models_loaded"])
        self.assertEqual(data["database"]["ok"], True)

    def test_models_endpoint(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("lstm", data["models"])
        self.assertIn("bilstm", data["models"])
        self.assertIsNotNone(data["models"]["bilstm"]["accuracy"])


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        database.init_db()
        database.clear_history()

    def test_schema_and_stats_on_empty_db(self):
        stats = database.get_statistics()
        self.assertEqual(stats["total_predictions"], 0)
        self.assertEqual(stats["fake_percentage"], 0.0)
        self.assertEqual(len(stats["trend"]), 7)

    def test_insert_and_aggregate(self):
        base_result = {
            "prediction": "FAKE",
            "label_code": 0,
            "confidence": 0.97,
            "model_used": "BiLSTM",
            "probability_fake": 0.97,
            "probability_real": 0.03,
            "processing_ms": 42.0,
            "text_stats": {"words_raw": 50},
            "model_scores": [
                {"key": "lstm", "confidence": 0.95},
                {"key": "bilstm", "confidence": 0.97},
            ],
        }
        row_id = database.record_prediction(base_result, "Example fake article text")
        self.assertGreater(row_id, 0)
        stats = database.get_statistics()
        self.assertEqual(stats["total_predictions"], 1)
        self.assertEqual(stats["fake_predictions"], 1)
        self.assertAlmostEqual(stats["average_confidence"], 0.97, places=4)
        rows = database.fetch_history(limit=1)
        self.assertEqual(rows[0]["model"], "BiLSTM")
        self.assertEqual(rows[0]["text_preview"], "Example fake article text")

    def test_preview_is_truncated(self):
        long_text = "word " * 200
        result = {
            "prediction": "REAL", "label_code": 1, "confidence": 0.8,
            "model_used": "LSTM", "probability_fake": 0.2,
            "probability_real": 0.8, "processing_ms": 10,
            "text_stats": {"words_raw": 200}, "model_scores": [],
        }
        database.record_prediction(result, long_text)
        row = database.fetch_history(limit=1)[0]
        self.assertLessEqual(len(row["text_preview"]), 221)


if __name__ == "__main__":
    unittest.main(verbosity=2)
