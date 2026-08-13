from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from asr_sidecar.service import (
    LoadedModels,
    SidecarError,
    SidecarService,
    _resolve_local_model_path,
)


class CapturedOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self._condition = threading.Condition()
        self._responses: list[dict[str, Any]] = []

    def write(self, value: str) -> int:
        with self._condition:
            written = super().write(value)
            for line in value.splitlines():
                if line:
                    self._responses.append(json.loads(line))
            self._condition.notify_all()
            return written

    def wait_for(self, request_id: str | None, timeout: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for index, response in enumerate(self._responses):
                    if response.get("id") == request_id:
                        return self._responses.pop(index)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail_timeout(request_id)
                self._condition.wait(remaining)

    def fail_timeout(self, request_id: str | None) -> None:
        raise AssertionError(f"timed out waiting for response {request_id!r}")


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        audio = kwargs.get("audio")
        result_count = len(audio) if isinstance(audio, list) else 1
        return [
            SimpleNamespace(
                text="你好这是本地语音识别功能测试",
                language="Chinese",
                time_stamps=[
                    SimpleNamespace(text="你好", start_time=0.1, end_time=0.4),
                    SimpleNamespace(text="测试", start_time=0.5, end_time=0.9),
                ],
            )
            for _ in range(result_count)
        ]


class WrappedTimestampFakeModel(FakeModel):
    def transcribe(self, **kwargs: Any) -> list[Any]:
        results = super().transcribe(**kwargs)
        for result in results:
            result.time_stamps = SimpleNamespace(items=result.time_stamps)
        return results


class BlockingFakeModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, **kwargs: Any) -> list[Any]:
        self.started.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test did not release fake inference")
        return super().transcribe(**kwargs)


class FailingFakeModel(FakeModel):
    def transcribe(self, **kwargs: Any) -> list[Any]:
        raise RuntimeError("fake inference failed")


class FakeLoader:
    def __init__(self, model: FakeModel | None = None) -> None:
        self.model = model or FakeModel()
        self.load_count = 0

    def load(self) -> LoadedModels:
        self.load_count += 1
        return LoadedModels(
            model=self.model,
            asr_model_path="/models/Qwen3-ASR-1.7B",
            aligner_model_path="/models/Qwen3-ForcedAligner-0.6B",
            asr_device="mps:0",
            aligner_device="mps:0",
            asr_dtype="bfloat16",
            aligner_dtype="bfloat16",
        )


class MPSUnavailableLoader(FakeLoader):
    def load(self) -> LoadedModels:
        raise SidecarError("MPS_UNAVAILABLE", "MPS unavailable in fake loader")


class SidecarServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output = CapturedOutput()
        self.stderr = io.StringIO()
        self.loader = FakeLoader()
        self.service = SidecarService(
            model_loader=self.loader,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )

    def tearDown(self) -> None:
        self.service.close_input()
        self.service.join()

    def dispatch(
        self,
        request_id: str,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        should_stop = self.service.dispatch_line(
            json.dumps(
                {
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
        )
        self.assertFalse(should_stop)
        return self.output.wait_for(request_id)

    def test_asr_model_paths_come_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH": directory},
                clear=True,
            ):
                self.assertEqual(
                    _resolve_local_model_path(
                        None,
                        "ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH",
                        "Qwen ASR",
                    ),
                    Path(directory).resolve(),
                )

    def test_asr_model_path_environment_is_required(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SidecarError) as context:
                _resolve_local_model_path(
                    None,
                    "ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH",
                    "Qwen ASR",
                )
        self.assertEqual(context.exception.code, "INITIALIZATION_FAILED")

    def initialize(self) -> dict[str, Any]:
        return self.dispatch("initialize", "initialize")

    def test_health_and_status_do_not_load_models_before_initialize(self) -> None:
        health = self.dispatch("health", "health")
        status = self.dispatch("status", "status")

        self.assertTrue(health["result"]["healthy"])
        self.assertEqual(status["result"]["state"], "uninitialized")
        self.assertFalse(status["result"]["initialized"])
        self.assertEqual(self.loader.load_count, 0)

    def test_request_validation_and_unknown_method_are_structured(self) -> None:
        self.service.dispatch_line("not json")
        malformed = self.output.wait_for(None)
        unknown = self.dispatch("unknown", "does-not-exist")
        invalid = self.dispatch("invalid", "cancel", {"taskId": ""})
        self.service.dispatch_line(
            json.dumps(
                {
                    "id": "invalid-top-level",
                    "method": "status",
                    "params": {},
                    "extra": True,
                }
            )
        )
        invalid_top_level = self.output.wait_for("invalid-top-level")

        self.assertEqual(malformed["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(unknown["error"]["code"], "UNKNOWN_METHOD")
        self.assertEqual(unknown["error"]["details"], {"method": "does-not-exist"})
        self.assertEqual(invalid["error"]["code"], "INVALID_REQUEST")
        self.assertFalse(invalid["error"]["retryable"])
        self.assertEqual(invalid_top_level["error"]["details"], {"keys": ["extra"]})

    def test_mps_unavailable_is_structured_and_state_becomes_failed(self) -> None:
        self.service.close_input()
        self.service.join()
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=MPSUnavailableLoader(),
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )

        initialization = self.dispatch("mps-unavailable", "initialize")
        status = self.dispatch("status-after-mps-error", "status")

        self.assertEqual(initialization["error"]["code"], "MPS_UNAVAILABLE")
        self.assertEqual(status["result"]["state"], "failed")
        self.assertFalse(status["result"]["initialized"])

    def test_initialize_is_idempotent_and_loads_models_once(self) -> None:
        first = self.initialize()
        second = self.dispatch("initialize-again", "initialize")
        status = self.dispatch("status-after-init", "status")

        self.assertEqual(first["result"]["state"], "ready")
        self.assertEqual(first["result"]["asrDevice"], "mps:0")
        self.assertEqual(second["result"]["loadCount"], 1)
        self.assertEqual(status["result"]["state"], "ready")
        self.assertEqual(self.loader.load_count, 1)

    def test_transcribe_validates_initialization_and_normalizes_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "fixture.wav"
            audio_path.write_bytes(b"RIFF-fake")
            not_ready = self.dispatch(
                "before-init",
                "transcribe",
                {"taskId": "before-init", "audioPath": str(audio_path)},
            )
            self.assertEqual(not_ready["error"]["code"], "NOT_INITIALIZED")

            self.initialize()
            transcription = self.dispatch(
                "transcribe",
                "transcribe",
                {
                    "taskId": "transcribe-task",
                    "audioPath": str(audio_path),
                    "language": "Chinese",
                },
            )

        self.assertEqual(
            transcription["result"]["timestamps"],
            [
                {"text": "你好", "start": 0.1, "end": 0.4},
                {"text": "测试", "start": 0.5, "end": 0.9},
            ],
        )
        self.assertTrue(self.loader.model.calls[0]["return_time_stamps"])

    def test_transcribe_accepts_forced_align_result_items_wrapper(self) -> None:
        self.service.close_input()
        self.service.join()
        self.loader = FakeLoader(WrappedTimestampFakeModel())
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=self.loader,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "fixture.wav"
            audio_path.write_bytes(b"RIFF-fake")
            self.initialize()
            transcription = self.dispatch(
                "wrapped-timestamps",
                "transcribe",
                {"taskId": "wrapped", "audioPath": str(audio_path)},
            )

        self.assertEqual(transcription["result"]["timestamps"][0]["text"], "你好")

    def test_batch_transcribe_validates_maps_in_input_order_and_auto_detects_language(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_audio = Path(temporary_directory) / "first.wav"
            second_audio = Path(temporary_directory) / "second.mp3"
            first_audio.write_bytes(b"RIFF-fake")
            second_audio.write_bytes(b"ID3-fake")
            self.initialize()
            transcription = self.dispatch(
                "batch-transcribe",
                "transcribe",
                {
                    "taskId": "batch-task",
                    "items": [
                        {"itemId": "first", "audioPath": str(first_audio)},
                        {
                            "itemId": "second",
                            "audioPath": str(second_audio),
                            "language": "Chinese",
                        },
                    ],
                },
            )

        self.assertEqual(transcription["result"]["taskId"], "batch-task")
        self.assertEqual(
            [item["itemId"] for item in transcription["result"]["items"]],
            ["first", "second"],
        )
        self.assertEqual(
            [item["language"] for item in transcription["result"]["items"]],
            ["Chinese", "Chinese"],
        )
        self.assertEqual(
            self.loader.model.calls[0]["audio"],
            [str(first_audio), str(second_audio)],
        )
        self.assertEqual(self.loader.model.calls[0]["language"], [None, "Chinese"])
        self.assertTrue(self.loader.model.calls[0]["return_time_stamps"])

    def test_batch_transcribe_rejects_empty_items_duplicate_ids_and_missing_audio(self) -> None:
        self.initialize()
        empty = self.dispatch(
            "empty-batch",
            "transcribe",
            {"taskId": "empty-batch", "items": []},
        )
        duplicate = self.dispatch(
            "duplicate-batch",
            "transcribe",
            {
                "taskId": "duplicate-batch",
                "items": [
                    {"itemId": "same", "audioPath": "/missing-one.wav"},
                    {"itemId": "same", "audioPath": "/missing-two.wav"},
                ],
            },
        )
        missing = self.dispatch(
            "missing-batch",
            "transcribe",
            {
                "taskId": "missing-batch",
                "items": [
                    {"itemId": "missing", "audioPath": "/missing/audio.wav"},
                ],
            },
        )
        health = self.dispatch("health-after-batch-error", "health")

        self.assertEqual(empty["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(duplicate["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(missing["error"]["code"], "AUDIO_NOT_FOUND")
        self.assertTrue(health["result"]["healthy"])
        self.assertEqual(health["result"]["state"], "ready")

    def test_missing_audio_is_recoverable_and_sidecar_remains_healthy(self) -> None:
        self.initialize()
        missing = self.dispatch(
            "missing",
            "transcribe",
            {"taskId": "missing", "audioPath": "/missing/audio.wav"},
        )
        health = self.dispatch("health-after-error", "health")

        self.assertEqual(missing["error"]["code"], "AUDIO_NOT_FOUND")
        self.assertTrue(health["result"]["healthy"])
        self.assertEqual(health["result"]["state"], "ready")

    def test_unexpected_model_error_is_serialized_and_traceback_stays_on_stderr(self) -> None:
        self.service.close_input()
        self.service.join()
        loader = FakeLoader(FailingFakeModel())
        self.output = CapturedOutput()
        self.stderr = io.StringIO()
        self.service = SidecarService(
            model_loader=loader,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "fixture.wav"
            audio_path.write_bytes(b"RIFF-fake")
            self.initialize()
            failed = self.dispatch(
                "failed-transcription",
                "transcribe",
                {"taskId": "failed", "audioPath": str(audio_path)},
            )

        self.assertEqual(failed["error"]["code"], "TRANSCRIPTION_FAILED")
        self.assertNotIn("Traceback", self.output.getvalue())
        self.assertIn("Traceback", self.stderr.getvalue())

    def test_running_and_queued_tasks_support_cooperative_cancellation(self) -> None:
        self.service.close_input()
        self.service.join()
        blocking_model = BlockingFakeModel()
        self.loader = FakeLoader(blocking_model)
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=self.loader,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "fixture.wav"
            audio_path.write_bytes(b"RIFF-fake")
            self.initialize()
            self.service.dispatch_line(
                json.dumps(
                    {
                        "id": "running-request",
                        "method": "transcribe",
                        "params": {
                            "taskId": "running-task",
                            "audioPath": str(audio_path),
                        },
                    }
                )
            )
            self.assertTrue(blocking_model.started.wait(1.0))
            duplicate = self.dispatch(
                "duplicate-running-task",
                "transcribe",
                {
                    "taskId": "running-task",
                    "audioPath": str(audio_path),
                },
            )
            self.service.dispatch_line(
                json.dumps(
                    {
                        "id": "queued-request",
                        "method": "transcribe",
                        "params": {
                            "taskId": "queued-task",
                            "audioPath": str(audio_path),
                        },
                    }
                )
            )

            busy_status = self.dispatch("status-busy", "status")
            self.assertEqual(busy_status["result"]["state"], "busy")
            self.assertEqual(busy_status["result"]["currentTaskId"], "running-task")
            self.assertEqual(busy_status["result"]["queuedTaskCount"], 1)

            queued_cancel = self.dispatch(
                "cancel-queued",
                "cancel",
                {"taskId": "queued-task"},
            )
            queued_original = self.output.wait_for("queued-request")
            running_cancel = self.dispatch(
                "cancel-running",
                "cancel",
                {"taskId": "running-task"},
            )
            status = self.dispatch("status-cancelling", "status")
            blocking_model.release.set()
            running_original = self.output.wait_for("running-request")

        self.assertEqual(queued_cancel["result"]["status"], "cancelled")
        self.assertEqual(duplicate["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(queued_original["error"]["code"], "TASK_CANCELLED")
        self.assertEqual(running_cancel["result"]["status"], "cancel_requested")
        self.assertTrue(status["result"]["cancelRequested"])
        self.assertEqual(running_original["error"]["code"], "TASK_CANCELLED")
        self.assertEqual(len(blocking_model.calls), 1)

    def test_running_batch_is_cancelled_as_one_task(self) -> None:
        self.service.close_input()
        self.service.join()
        blocking_model = BlockingFakeModel()
        self.loader = FakeLoader(blocking_model)
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=self.loader,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            first_audio = Path(temporary_directory) / "first.wav"
            second_audio = Path(temporary_directory) / "second.wav"
            first_audio.write_bytes(b"RIFF-fake")
            second_audio.write_bytes(b"RIFF-fake")
            self.initialize()
            self.service.dispatch_line(
                json.dumps(
                    {
                        "id": "running-batch-request",
                        "method": "transcribe",
                        "params": {
                            "taskId": "running-batch",
                            "items": [
                                {"itemId": "first", "audioPath": str(first_audio)},
                                {"itemId": "second", "audioPath": str(second_audio)},
                            ],
                        },
                    }
                )
            )
            self.assertTrue(blocking_model.started.wait(1.0))
            cancelled = self.dispatch(
                "cancel-running-batch",
                "cancel",
                {"taskId": "running-batch"},
            )
            blocking_model.release.set()
            original = self.output.wait_for("running-batch-request")

        self.assertEqual(cancelled["result"]["status"], "cancel_requested")
        self.assertEqual(original["error"]["code"], "TASK_CANCELLED")
        self.assertEqual(len(blocking_model.calls), 1)
        self.assertEqual(len(blocking_model.calls[0]["audio"]), 2)

    def test_shutdown_responds_and_worker_exits(self) -> None:
        should_stop = self.service.dispatch_line(
            json.dumps({"id": "shutdown", "method": "shutdown", "params": {}})
        )
        response = self.output.wait_for("shutdown")

        self.assertTrue(should_stop)
        self.assertEqual(response["result"], {"state": "shutting_down"})
        self.service.join()


if __name__ == "__main__":
    unittest.main()
