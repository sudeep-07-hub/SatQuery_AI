"""Safeguards for the small free-tier deployment: job cap, disabled tools, upload size limit."""
import os

import pytest

import job_manager
from agent.adapters import build_adapters


def test_job_cap_evicts_oldest_finished_jobs_and_their_files(tmp_path, monkeypatch):
    registry = job_manager.JobRegistry(max_jobs=2)
    first = registry.create_job()
    export = tmp_path / "export_first.json"
    export.write_text("{}")
    registry.get_job(first)["exports"] = {"json": str(export)}
    registry.update_status(first, "DONE")
    second = registry.create_job()
    registry.update_status(second, "DONE")

    third = registry.create_job()

    assert registry.get_job(first) is None
    assert not export.exists()
    assert registry.get_job(second) is not None and registry.get_job(third) is not None


def test_job_cap_never_evicts_running_jobs():
    registry = job_manager.JobRegistry(max_jobs=1)
    running = registry.create_job()
    registry.update_status(running, "AGENTIC_EXECUTION")
    registry.create_job()
    assert registry.get_job(running) is not None


def test_job_cap_disabled_by_default():
    registry = job_manager.JobRegistry(max_jobs=0)
    ids = [registry.create_job() for _ in range(5)]
    for jid in ids:
        registry.update_status(jid, "DONE")
    registry.create_job()
    assert all(registry.get_job(jid) for jid in ids)


def test_disabled_tools_are_reported_unavailable_without_probing(monkeypatch):
    monkeypatch.setattr(job_manager, "DISABLED_TOOLS", {"single_image_vqa"})
    adapters = build_adapters("real")

    def must_not_probe():
        raise AssertionError("availability() must not run for a disabled tool")

    monkeypatch.setattr(adapters["single_image_vqa"], "availability", must_not_probe)
    tool_registry, report = job_manager.build_job_registry("real", adapters)

    vqa = next(r for r in report if r["tool_id"] == "single_image_vqa")
    assert vqa["available"] is False and vqa["reason"].startswith("DISABLED")
    assert tool_registry.get("single_image_vqa").enabled is False


def test_upload_limit_returns_413(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
    client = TestClient(main.app)
    response = client.post(
        "/api/query",
        files=[("files", ("big.tif", b"II*\x00" + b"0" * 2048, "image/tiff"))],
        data={"query": "What is here?"},
    )
    assert response.status_code == 413
    assert "Upload too large" in response.json()["detail"]
