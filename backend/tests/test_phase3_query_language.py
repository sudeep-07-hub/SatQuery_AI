"""
Phase 3 — the query/answer language seam.

Covers the contract that matters for honesty: a non-English query is translated once, before MC2,
and the transformation is recorded; and when nothing can translate it, the job is refused rather
than run through English keyword rules that would answer a question the pipeline never understood.
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import job_manager
from job_manager import execute_agentic_pipeline, job_registry
from qwen.translation import (
    TranslationUnavailable,
    translate_query_to_english,
    translate_text_from_english,
    translate_texts_from_english,
)


class StubEngine:
    """Minimal stand-in for the Qwen3 engine's generate() contract."""

    model_name = "qwen3:4b"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0) if self.payloads else {"status": "error", "error": "exhausted"}
        if isinstance(payload, dict) and payload.get("status") == "error":
            return payload
        # "ok" is the success status both real engines return; a stub that said "success" would
        # have hidden a live failure, which is exactly what happened before this was corrected.
        return {"status": "ok", "response": json.dumps(payload), "metadata": {}}


MC1_PROFILE = {
    "task_executable": True,
    "spatial_overlap": 0.95,
    "coregistration_score": 0.92,
    "alignment": "aligned",
    "image_1": {"filename": "a.tif", "modality": "sar"},
    "image_2": {"filename": "b.tif", "modality": "sar"},
}

FIXTURES = Path(__file__).parent / "fixtures"


def real_files():
    """Real GeoTIFFs: MC1 runs before the translation seam and rejects unreadable bytes."""
    return [
        ("test_geo1.tif", (FIXTURES / "test_geo1.tif").read_bytes()),
        ("test_geo2.tif", (FIXTURES / "test_geo2.tif").read_bytes()),
    ]


# ── The translation module itself ──────────────────────────────────────────

def test_query_translation_returns_the_auditable_record():
    engine = StubEngine([{"translation": "Has a new airstrip been cleared?"}])
    record = translate_query_to_english(engine, "ಹೊಸ ವಿಮಾನ ಪಟ್ಟಿ ನಿರ್ಮಾಣವಾಗಿದೆಯೇ?", "kn")

    assert record["from"] == "kn" and record["to"] == "en"
    assert record["engine"] == "qwen3:4b"
    assert record["original"] == "ಹೊಸ ವಿಮಾನ ಪಟ್ಟಿ ನಿರ್ಮಾಣವಾಗಿದೆಯೇ?"
    assert record["translated"] == "Has a new airstrip been cleared?"


def test_query_translation_without_an_engine_is_refused():
    with pytest.raises(TranslationUnavailable):
        translate_query_to_english(None, "ಏನು ಬದಲಾಯಿತು?", "kn")


def test_query_translation_refuses_rather_than_inventing_on_model_error():
    engine = StubEngine([{"status": "error", "error": "connection refused"}])
    with pytest.raises(TranslationUnavailable):
        translate_query_to_english(engine, "ఏమి మారింది?", "te")


def test_query_translation_refuses_an_unrecognised_status():
    """Guards the engine contract: success is reported as "ok", not "success"."""
    class WrongStatusEngine:
        model_name = "qwen3:4b"

        def generate(self, prompt, **kwargs):
            return {"status": "success", "response": json.dumps({"translation": "hi"}), "metadata": {}}

    with pytest.raises(TranslationUnavailable):
        translate_query_to_english(WrongStatusEngine(), "ಏನು?", "kn")


def test_query_translation_refuses_an_empty_translation():
    engine = StubEngine([{"translation": "   "}])
    with pytest.raises(TranslationUnavailable):
        translate_query_to_english(engine, "என்ன மாறியது?", "ta")


def test_answer_translation_falls_back_to_english_rather_than_losing_the_answer():
    engine = StubEngine([{"status": "error", "error": "model gone"}])
    assert translate_text_from_english(engine, "No change was detected.", "hi") == "No change was detected."


def test_caveats_are_translated_one_passage_per_call():
    """Batching made the model return one invented sentence for every passage; see FIX_LOG."""
    engine = StubEngine([{"translation": "ಒಂದು"}, {"translation": "ಎರಡು"}])
    out = translate_texts_from_english(engine, ["One.", "Two."], "kn")
    assert out == ["ಒಂದು", "ಎರಡು"]
    assert len(engine.prompts) == 2


def test_a_caveat_the_model_cannot_translate_stays_english_on_its_own():
    """Degrades per passage: a failure must not take the other caveats down with it."""
    engine = StubEngine([{"translation": "ಒಂದು"}, {"status": "error", "error": "loop"}])
    out = translate_texts_from_english(engine, ["One.", "Two."], "kn")
    assert out == ["ಒಂದು", "Two."]


# ── The pipeline seam ──────────────────────────────────────────────────────

def test_non_english_query_is_refused_when_no_language_model_can_translate_it():
    """The Tool Registry matches English keywords; it must never see a non-English string."""
    job_id = job_registry.create_job()
    with patch("job_manager.RegistryQueryInterpreter") as registry_interpreter:
        asyncio.run(execute_agentic_pipeline(
            job_id, real_files(), "ಈ ಎರಡು ಚಿತ್ರಗಳ ನಡುವೆ ಏನು ಬದಲಾಯಿತು?",
            execution_mode="real", qwen_backend="none", query_language="kn",
        ))

    job = job_registry.get_job(job_id)
    assert job["status"] == "TRANSLATION_UNAVAILABLE"
    assert job["result"]["execution_status"] == "TRANSLATION_UNAVAILABLE"
    assert "Tool Registry rules" in job["result"]["final_answer"]
    assert job["result"]["query_language"] == "kn"
    # The honest part: no keyword matching was attempted on the Kannada string.
    registry_interpreter.assert_not_called()
    assert job["result"]["claims"] == []


def test_english_query_never_enters_the_translation_seam():
    job_id = job_registry.create_job()
    with patch("job_manager.run_mc1_pipeline", return_value=dict(MC1_PROFILE)), \
         patch("job_manager.translate_query_to_english") as translate:
        asyncio.run(execute_agentic_pipeline(
            job_id, real_files(), "What changed between these two images?", execution_mode="fixture",
        ))

    translate.assert_not_called()
    job = job_registry.get_job(job_id)
    assert "input_translation" not in job
    assert job["result"]["query_language"] == "en"


def test_translation_happens_before_mc2_and_is_recorded_in_the_trace():
    job_id = job_registry.create_job()
    record = {
        "from": "ta", "to": "en", "engine": "qwen3:4b",
        "original": "இந்த இரண்டு படங்களுக்கு இடையே என்ன மாறியது?",
        "translated": "What changed between these two images?",
    }
    seen = {}
    real_pipeline = job_manager.QueryIntelligencePipeline

    class CapturingPipeline(real_pipeline):
        def process_query(self, query):
            seen["query"] = query
            return super().process_query(query)

    with patch("job_manager.run_mc1_pipeline", return_value=dict(MC1_PROFILE)), \
         patch("job_manager.translate_query_to_english", return_value=record), \
         patch("job_manager.QueryIntelligencePipeline", CapturingPipeline):
        asyncio.run(execute_agentic_pipeline(
            job_id, real_files(), record["original"],
            execution_mode="fixture", query_language="ta",
        ))

    job = job_registry.get_job(job_id)
    # MC2 received the English string, not the Tamil one.
    assert seen["query"] == record["translated"]
    assert job["input_translation"] == record
    entries = [e for e in job["progress_trace"] if "input_translation" in e]
    assert len(entries) == 1
    assert entries[0]["input_translation"]["original"] == record["original"]
    assert entries[0]["input_translation"]["translated"] == record["translated"]


# ── The API contract ───────────────────────────────────────────────────────
# fastapi.testclient needs httpx, which is not installed and which this work order does not
# permit adding, so the endpoint coroutine is driven directly.

class _UploadStub:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


class _BackgroundTasksStub:
    def __init__(self):
        self.calls = []

    def add_task(self, func, *args, **kwargs):
        self.calls.append((func, args, kwargs))


def test_query_language_is_declared_optional_with_an_english_default():
    """
    Driving the coroutine directly bypasses FastAPI's Form-default resolution, so this asserts the
    *declared* default. That a request omitting the field behaves as before is verified over real
    HTTP in the Phase 3 gate.
    """
    import inspect

    import main

    default = inspect.signature(main.submit_query).parameters["query_language"].default
    assert getattr(default, "default", default) == "en"


def test_an_english_query_is_forwarded_without_translation():
    import main

    tasks = _BackgroundTasksStub()
    asyncio.run(main.submit_query(
        tasks, files=[_UploadStub("a.tif", b"x")], query="What changed?", query_language="en",
    ))
    _, _, kwargs = tasks.calls[0]
    assert kwargs["query_language"] == "en"


def test_query_language_is_forwarded_to_the_pipeline():
    import main

    tasks = _BackgroundTasksStub()
    asyncio.run(main.submit_query(
        tasks, files=[_UploadStub("a.tif", b"x")], query="ಏನು ಬದಲಾಯಿತು?", query_language="kn",
    ))
    func, args, kwargs = tasks.calls[0]
    assert func is main.run_agentic_pipeline_sync
    assert kwargs["query_language"] == "kn"
    assert args[2] == "ಏನು ಬದಲಾಯಿತು?"


def test_unsupported_query_language_is_rejected():
    from fastapi import HTTPException

    import main

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(main.submit_query(
            _BackgroundTasksStub(), files=[_UploadStub("a.tif", b"x")], query="Was hat sich geändert?",
            query_language="de",
        ))
    assert excinfo.value.status_code == 400
    assert "de" in excinfo.value.detail


# ── Exports stay English ───────────────────────────────────────────────────

def test_exports_use_the_english_original_not_the_translated_prose():
    """ReportLab's built-in faces have no Indic glyphs; the PDF must not be sent boxes."""
    from mc8_export.exporter import _audit_text

    translated = {
        "final_answer": "ಬದಲಾವಣೆ ಕಂಡುಬಂದಿದೆ.",
        "final_answer_en": "A change was detected.",
        "caveats": ["ದಿನಾಂಕಗಳು ಇಲ್ಲ."],
        "caveats_en": ["Acquisition dates missing."],
    }
    assert _audit_text(translated, "final_answer") == "A change was detected."
    assert _audit_text(translated, "caveats") == ["Acquisition dates missing."]


def test_exports_use_the_answer_directly_for_english_jobs():
    from mc8_export.exporter import _audit_text

    english = {"final_answer": "A change was detected.", "caveats": ["Dates missing."]}
    assert _audit_text(english, "final_answer") == "A change was detected."
    assert _audit_text(english, "caveats") == ["Dates missing."]
