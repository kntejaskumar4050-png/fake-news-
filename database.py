"""SQLite persistence for prediction history and dashboard statistics."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    text            TEXT    NOT NULL,
    text_preview    TEXT    NOT NULL,
    label           TEXT    NOT NULL CHECK (label IN ('FAKE', 'REAL')),
    label_code      INTEGER NOT NULL CHECK (label_code IN (0, 1)),
    confidence      REAL    NOT NULL,
    model           TEXT    NOT NULL,
    probability_fake REAL,
    probability_real REAL,
    lstm_confidence   REAL,
    bilstm_confidence REAL,
    processing_ms   REAL,
    word_count      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_label
    ON predictions (label);
"""


def _db_path(path: Path | str | None = None) -> Path:
    return Path(path or config.DATABASE_PATH)


@contextmanager
def get_connection(path: Path | str | None = None):
    """Open a short-lived connection with row access by column name."""
    db_path = _db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(path: Path | str | None = None) -> None:
    with get_connection(path) as connection:
        connection.executescript(SCHEMA)


def _preview(text: str, limit: int = 220) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def record_prediction(result: dict, text: str, path: Path | str | None = None) -> int:
    """Persist one prediction and return its row id."""
    scores = {item["key"]: item for item in result.get("model_scores", [])}
    with get_connection(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at, text, text_preview, label, label_code, confidence,
                model, probability_fake, probability_real,
                lstm_confidence, bilstm_confidence, processing_ms, word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(text)[: config.MAX_INPUT_CHARS],
                _preview(text),
                result["prediction"],
                int(result["label_code"]),
                float(result["confidence"]),
                result["model_used"],
                float(result.get("probability_fake", 0.0)),
                float(result.get("probability_real", 0.0)),
                scores.get("lstm", {}).get("confidence"),
                scores.get("bilstm", {}).get("confidence"),
                float(result.get("processing_ms") or 0.0),
                int(result.get("text_stats", {}).get("words_raw") or len(str(text).split())),
            ),
        )
        return int(cursor.lastrowid)


def fetch_history(limit: int = 20, label: str | None = None,
                  path: Path | str | None = None) -> list[dict]:
    limit = max(1, min(int(limit), config.MAX_HISTORY_ROWS))
    query = """
        SELECT id, created_at, text_preview, label, label_code, confidence,
               model, probability_fake, probability_real, processing_ms, word_count
        FROM predictions
    """
    params: list = []
    if label in ("FAKE", "REAL"):
        query += " WHERE label = ?"
        params.append(label)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_connection(path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def fetch_prediction(prediction_id: int, path: Path | str | None = None) -> dict | None:
    with get_connection(path) as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
    return dict(row) if row else None


def get_statistics(days: int = 7, path: Path | str | None = None) -> dict:
    """Aggregates for the dashboard (all values come from stored predictions)."""
    with get_connection(path) as connection:
        totals = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(label = 'FAKE') AS fake,
                   SUM(label = 'REAL') AS real,
                   AVG(confidence) AS avg_confidence,
                   AVG(processing_ms) AS avg_latency,
                   MAX(created_at) AS last_prediction
            FROM predictions
            """
        ).fetchone()

        by_model = connection.execute(
            """
            SELECT model, COUNT(*) AS count
            FROM predictions GROUP BY model ORDER BY count DESC
            """
        ).fetchall()

        daily = connection.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   SUM(label = 'FAKE') AS fake,
                   SUM(label = 'REAL') AS real,
                   COUNT(*) AS total
            FROM predictions
            WHERE date(created_at) >= date('now', ?)
            GROUP BY day ORDER BY day ASC
            """,
            (f"-{max(days - 1, 0)} days",),
        ).fetchall()

        confidence_buckets = connection.execute(
            """
            SELECT CASE
                     WHEN confidence >= 0.90 THEN '90-100%'
                     WHEN confidence >= 0.75 THEN '75-90%'
                     WHEN confidence >= 0.60 THEN '60-75%'
                     WHEN confidence >= 0.50 THEN '50-60%'
                     ELSE '<50%'
                   END AS bucket,
                   COUNT(*) AS count
            FROM predictions GROUP BY bucket
            """
        ).fetchall()

        today = connection.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE date(created_at) = date('now')"
        ).fetchone()

    total = int(totals["total"] or 0)
    fake = int(totals["fake"] or 0)
    real = int(totals["real"] or 0)

    # Build a continuous 7-day series so the chart has no gaps.
    series = []
    by_day = {row["day"]: dict(row) for row in daily}
    for offset in range(days - 1, -1, -1):
        day = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        record = by_day.get(day, {"day": day, "fake": 0, "real": 0, "total": 0})
        series.append(
            {
                "day": day,
                "label": datetime.strptime(day, "%Y-%m-%d").strftime("%a"),
                "fake": int(record.get("fake") or 0),
                "real": int(record.get("real") or 0),
                "total": int(record.get("total") or 0),
            }
        )

    return {
        "total_predictions": total,
        "fake_predictions": fake,
        "real_predictions": real,
        "fake_percentage": round(fake / total * 100, 1) if total else 0.0,
        "real_percentage": round(real / total * 100, 1) if total else 0.0,
        "average_confidence": round(float(totals["avg_confidence"] or 0), 4),
        "average_latency_ms": round(float(totals["avg_latency"] or 0), 1),
        "predictions_today": int(today["c"] or 0),
        "last_prediction": totals["last_prediction"],
        "by_model": [dict(row) for row in by_model],
        "trend": series,
        "confidence_distribution": [dict(row) for row in confidence_buckets],
    }


def clear_history(path: Path | str | None = None) -> int:
    with get_connection(path) as connection:
        cursor = connection.execute("DELETE FROM predictions")
        try:
            connection.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        except sqlite3.OperationalError:
            pass
        return int(cursor.rowcount or 0)


def seed_demo_if_empty(path: Path | str | None = None) -> None:  # pragma: no cover
    """No synthetic rows are ever inserted. Kept as an explicit no-op."""
    return None


if __name__ == "__main__":  # manual inspection helper
    import json

    init_db()
    print(json.dumps(get_statistics(), indent=2))
    print(json.dumps(fetch_history(5), indent=2))
