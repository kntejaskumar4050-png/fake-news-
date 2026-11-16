"""
FAKE NEWS DETECTOR — Flask application.

Pages
-----
/            Landing page
/detect      Detection workspace
/dashboard   Analytics dashboard (HTML; JSON with Accept: application/json)
/about       Project documentation page

API
---
POST /predict              classify an article
GET  /dashboard            statistics (JSON when requested, HTML otherwise)
GET  /history              recent prediction history (?limit=&label=)
DELETE /history            clear prediction history
GET  /api/dashboard        statistics (JSON alias)
GET  /api/models           real test metrics for LSTM / BiLSTM
GET  /api/health           service + model health
GET  /api/predictions/<id> single stored prediction

Run:  python app.py    (or: flask --app app run --debug)
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

import config
import database
from predict import ModelNotReadyError, get_predictor, warm_up

APP_NAME = "FAKE NEWS DETECTOR"
APP_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Lightweight in-memory rate limiter (per IP, fixed window)
# --------------------------------------------------------------------------- #
_RATE_LIMIT = int(os.environ.get("FND_RATE_LIMIT", "120"))
_RATE_WINDOW = 60.0
_rate_state: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def _rate_limited(ip: str) -> bool:
    if _RATE_LIMIT <= 0:
        return False
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_state[ip]
        while bucket and now - bucket[0] > _RATE_WINDOW:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT:
            return True
        bucket.append(now)
        return False


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        APP_NAME=APP_NAME,
        APP_VERSION=APP_VERSION,
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,   # 2 MB request ceiling
        JSON_SORT_KEYS=False,
        SEND_FILE_MAX_AGE_DEFAULT=3600,
        TEMPLATES_AUTO_RELOAD=os.environ.get("FLASK_DEBUG") == "1",
        MAX_INPUT_CHARS=config.MAX_INPUT_CHARS,
        MIN_INPUT_CHARS=config.MIN_INPUT_CHARS,
        DEFAULT_MODEL=config.DEFAULT_MODEL,
    )
    if test_config:
        app.config.update(test_config)

    database.init_db()

    # ------------------------------------------------------------------ #
    # Context helpers
    # ------------------------------------------------------------------ #
    def predictor():
        return get_predictor()

    def safe_metrics() -> dict:
        try:
            return predictor().model_summary()
        except Exception:
            return {"models": {}, "best_model": None, "dataset": {}, "training": {}}

    def site_stats() -> dict:
        try:
            stats = database.get_statistics()
            stats.pop("trend", None)
            stats.pop("confidence_distribution", None)
            return stats
        except Exception:
            return {"total_predictions": 0, "fake_predictions": 0,
                    "real_predictions": 0, "average_confidence": 0.0}

    @app.context_processor
    def inject_globals():
        summary = safe_metrics()
        models = summary.get("models", {})
        best = summary.get("best_model")
        best_metrics = models.get(best, {}) if best else {}
        if not best_metrics and models:
            best = next(iter(models))
            best_metrics = models[best]
        dataset = summary.get("dataset", {})
        return {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "current_year": datetime.now().year,
            "best_model_key": best,
            "best_model_name": config.DISPLAY_NAMES.get(best, "—") if best else "—",
            "best_accuracy": best_metrics.get("accuracy"),
            "best_f1": best_metrics.get("f1_score"),
            "model_metrics": models,
            "dataset_info": dataset,
            "stats": site_stats(),
            "max_input_chars": config.MAX_INPUT_CHARS,
            "min_input_chars": config.MIN_INPUT_CHARS,
            "disclaimer": (
                "Predictions are pattern-based classifications, not fact-checks. "
                "Confidence reflects the model's certainty, not the truth of an article."
            ),
        }

    # ------------------------------------------------------------------ #
    # JSON helpers
    # ------------------------------------------------------------------ #
    def json_error(message: str, status: int = 400, **extra):
        payload = {"success": False, "error": message, "status": status}
        payload.update(extra)
        return jsonify(payload), status

    def wants_json() -> bool:
        if request.args.get("format", "").lower() == "json":
            return True
        accept = request.headers.get("Accept", "")
        return "application/json" in accept and "text/html" not in accept

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #
    @app.get("/")
    def index():
        return render_template("index.html", active="home")

    @app.get("/detect")
    def detect():
        return render_template(
            "detect.html",
            active="detect",
            model_options=[
                {"key": "bilstm", "name": "BiLSTM", "hint": "Best overall accuracy"},
                {"key": "lstm", "name": "LSTM", "hint": "Fast, single direction"},
                {"key": "ensemble", "name": "Ensemble", "hint": "Average of both models"},
            ],
            default_model=app.config["DEFAULT_MODEL"],
        )

    @app.get("/dashboard")
    def dashboard():
        if wants_json():
            try:
                payload = database.get_statistics()
            except Exception as exc:
                return json_error(f"Could not read statistics: {exc}", 500)
            payload["model_performance"] = safe_metrics().get("models", {})
            payload["best_model"] = safe_metrics().get("best_model")
            payload["success"] = True
            return jsonify(payload)
        return render_template("dashboard.html", active="dashboard")

    @app.get("/about")
    def about():
        return render_template("about.html", active="about")

    # ------------------------------------------------------------------ #
    # Prediction API
    # ------------------------------------------------------------------ #
    @app.post("/predict")
    def predict_route():
        if _rate_limited(request.remote_addr or "unknown"):
            return json_error(
                "Too many requests. Please wait a moment before trying again.",
                429,
            )

        payload = request.get_json(silent=True)
        if payload is None and request.form:
            payload = request.form.to_dict()
        if not isinstance(payload, dict):
            return json_error(
                "Request body must be JSON, e.g. {\"text\": \"...\", \"model\": \"bilstm\"}.",
                400,
            )

        text = payload.get("text") or payload.get("article") or payload.get("content")
        model_key = (payload.get("model") or app.config["DEFAULT_MODEL"]).strip().lower()
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)
        if not isinstance(text, str):
            return json_error("The 'text' field must be a string.", 400)

        text = text.strip()
        if not text:
            return json_error("Please enter a news headline or article to analyse.", 400)
        if len(text) < config.MIN_INPUT_CHARS:
            return json_error(
                f"Input is too short ({len(text)} characters). "
                f"Please provide at least {config.MIN_INPUT_CHARS} characters.",
                400,
                min_characters=config.MIN_INPUT_CHARS,
            )
        if len(text) > config.MAX_INPUT_CHARS:
            return json_error(
                f"Input is too long ({len(text):,} characters). "
                f"The limit is {config.MAX_INPUT_CHARS:,} characters.",
                413,
                max_characters=config.MAX_INPUT_CHARS,
            )
        if model_key not in ("lstm", "bilstm", "ensemble"):
            return json_error(
                "Unknown model. Choose 'lstm', 'bilstm' or 'ensemble'.", 400
            )

        try:
            result = predictor().predict(text, model_key)
        except ModelNotReadyError as exc:
            return json_error(str(exc), 503, code="MODEL_NOT_READY")
        except ValueError as exc:
            return json_error(str(exc), 422, code="INVALID_INPUT")
        except Exception as exc:  # pragma: no cover - defensive
            app.logger.exception("prediction failed")
            return json_error(f"Prediction failed: {exc}", 500, code="PREDICTION_ERROR")

        try:
            result["id"] = database.record_prediction(result, text)
            result["stored"] = True
        except Exception as exc:
            app.logger.warning("prediction could not be stored: %s", exc)
            result["id"] = None
            result["stored"] = False

        result["success"] = True
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(result)

    # ------------------------------------------------------------------ #
    # History / stats
    # ------------------------------------------------------------------ #
    @app.get("/history")
    @app.get("/api/history")
    def history():
        try:
            limit = int(request.args.get("limit", 20))
        except (TypeError, ValueError):
            return json_error("'limit' must be an integer.", 400)
        if not 1 <= limit <= config.MAX_HISTORY_ROWS:
            return json_error(
                f"'limit' must be between 1 and {config.MAX_HISTORY_ROWS}.", 400
            )
        label = (request.args.get("label") or "").upper() or None
        if label and label not in ("FAKE", "REAL"):
            return json_error("'label' must be FAKE or REAL.", 400)
        try:
            rows = database.fetch_history(limit=limit, label=label)
        except Exception as exc:
            return json_error(f"Could not read history: {exc}", 500)
        return jsonify({"success": True, "count": len(rows), "predictions": rows})

    @app.delete("/history")
    @app.delete("/api/history")
    def clear_history_route():
        try:
            deleted = database.clear_history()
        except Exception as exc:
            return json_error(f"Could not clear history: {exc}", 500)
        return jsonify({"success": True, "deleted": deleted,
                        "message": f"Deleted {deleted} prediction(s)."})

    @app.get("/dashboard/stats")
    @app.get("/api/dashboard")
    def dashboard_stats():
        try:
            stats = database.get_statistics()
        except Exception as exc:
            return json_error(f"Could not read statistics: {exc}", 500)
        stats["model_performance"] = safe_metrics().get("models", {})
        stats["best_model"] = safe_metrics().get("best_model")
        stats["success"] = True
        return jsonify(stats)

    @app.get("/api/models")
    def models_route():
        summary = safe_metrics()
        return jsonify({"success": True, **summary,
                        "input_limits": {
                            "min_characters": config.MIN_INPUT_CHARS,
                            "max_characters": config.MAX_INPUT_CHARS,
                            "max_tokens": config.MAX_LEN,
                        }})

    @app.get("/api/predictions/<int:prediction_id>")
    def prediction_detail(prediction_id: int):
        try:
            row = database.fetch_prediction(prediction_id)
        except Exception as exc:
            return json_error(f"Database error: {exc}", 500)
        if not row:
            return json_error("Prediction not found.", 404)
        return jsonify({"success": True, "prediction": row})

    @app.get("/api/health")
    def health():
        pred = predictor()
        summary = safe_metrics()
        db_ok = True
        db_error = None
        try:
            database.get_statistics()
        except Exception as exc:  # pragma: no cover
            db_ok = False
            db_error = str(exc)
        payload = {
            "success": True,
            "status": "ok" if (pred.loaded and db_ok) else "degraded",
            "app": APP_NAME,
            "version": APP_VERSION,
            "models_loaded": pred.loaded,
            "models": pred.available_models,
            "best_model": summary.get("best_model"),
            "database": {"ok": db_ok, "error": db_error},
            "tensorflow": _tf_version(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        return jsonify(payload), (200 if payload["status"] == "ok" else 503)

    # ------------------------------------------------------------------ #
    # Errors
    # ------------------------------------------------------------------ #
    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api") or wants_json():
            return json_error("Resource not found.", 404)
        return render_template("error.html", active="", code=404,
                               title="Page not found",
                               message="The page you are looking for does not exist."), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        if request.path.startswith("/api") or request.is_json or wants_json():
            return json_error("Method not allowed for this endpoint.", 405)
        return render_template("error.html", active="", code=405,
                               title="Method not allowed",
                               message="This endpoint does not support that request method."), 405

    @app.errorhandler(413)
    def payload_too_large(error):
        return json_error(
            "Request body is too large. The maximum upload size is 2 MB and the "
            f"text limit is {config.MAX_INPUT_CHARS:,} characters.",
            413,
        )

    @app.errorhandler(500)
    def server_error(error):  # pragma: no cover
        app.logger.exception("unhandled server error")
        if request.path.startswith("/api") or wants_json():
            return json_error("Internal server error.", 500)
        return render_template("error.html", active="", code=500,
                               title="Something went wrong",
                               message="An unexpected error occurred. Please try again."), 500

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if request.path.startswith("/api") or request.path in ("/predict", "/history", "/dashboard"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # Optional background warm-up for WSGI deployments. The __main__ entry point
    # loads the models synchronously *before* binding the port, which avoids a
    # slow first request and prevents TensorFlow from competing with server boot.
    if not app.config.get("TESTING") and os.environ.get("FND_WARMUP") == "background":
        warm_up(background=True)

    return app


def _tf_version() -> str | None:
    """Read the TensorFlow version from package metadata.

    Importing TensorFlow inside a request handler would race with the model
    loader thread, so metadata is used instead.
    """
    try:
        from importlib.metadata import version

        return version("tensorflow")
    except Exception:
        return None


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n  {APP_NAME} v{APP_VERSION}")
    if os.environ.get("FND_WARMUP", "1") != "0":
        print("  Loading trained models …", flush=True)
        warm_up(background=False)
    print(f"  Serving on http://127.0.0.1:{port}")
    print("  Press CTRL+C to stop\n")
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=False, threaded=True)
