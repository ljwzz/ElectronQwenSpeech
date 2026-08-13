from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tts_sidecar.service import (
    AudioMetadata,
    LoadedTTSModel,
    SidecarError,
    SidecarService,
    _resolve_local_model_path,
    _resolve_output_root,
)


class CapturedOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self._condition = threading.Condition()
        self._responses: list[dict[str, Any]] = []
        self._partial = ""

    def write(self, value: str) -> int:
        with self._condition:
            self._partial += value
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                if line:
                    self._responses.append(json.loads(line))
            self._condition.notify_all()
        return len(value)

    def flush(self) -> None:
        return None

    def wait_for(self, request_id: str | None, timeout: float = 2) -> dict[str, Any]:
        with self._condition:
            matched = self._condition.wait_for(
                lambda: any(response.get("id") == request_id for response in self._responses),
                timeout=timeout,
            )
            if not matched:
                raise AssertionError(f"未收到请求 {request_id!r} 的响应")
            index = next(
                index
                for index, response in enumerate(self._responses)
                if response.get("id") == request_id
            )
            return self._responses.pop(index)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_custom_voice(self, **kwargs: Any) -> tuple[list[object], int]:
        self.calls.append(kwargs)
        return [object()], 24000


class BlockingFakeModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_custom_voice(self, **kwargs: Any) -> tuple[list[object], int]:
        self.calls.append(kwargs)
        self.started.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("test release timeout")
        return [object()], 24000


class FakeLoader:
    def __init__(self, model: FakeModel | None = None) -> None:
        self.model = model or FakeModel()
        self.load_count = 0

    def load(self) -> LoadedTTSModel:
        self.load_count += 1
        return LoadedTTSModel(
            model=self.model,
            model_path="/models/Qwen3-TTS-CustomVoice",
            device="mps:0",
            dtype="bfloat16",
            supported_languages=("chinese", "english"),
            supported_voice_ids=("vivian", "ryan"),
        )


class FakeWriter:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def write(self, output_path: Path, waveform: Any, sample_rate: int) -> AudioMetadata:
        del waveform
        self.paths.append(output_path)
        output_path.write_bytes(b"RIFF-test-wave")
        return AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            frame_count=48000,
            duration_seconds=2,
        )


class SidecarServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary_directory.name)
        self.output = CapturedOutput()
        self.stderr = io.StringIO()
        self.loader = FakeLoader()
        self.writer = FakeWriter()
        self.service = SidecarService(
            model_loader=self.loader,
            audio_writer=self.writer,
            output_root=self.output_root,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )

    def tearDown(self) -> None:
        self.service.close_input()
        self.service.join()
        self.temporary_directory.cleanup()

    def dispatch(
        self,
        request_id: str,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        should_stop = self.service.dispatch_line(json.dumps({
            "id": request_id,
            "method": method,
            "params": params or {},
        }))
        self.assertFalse(should_stop)
        return self.output.wait_for(request_id)

    def initialize(self) -> dict[str, Any]:
        return self.dispatch("initialize", "initialize")

    def synthesis_params(self, task_id: str = "tts-task") -> dict[str, Any]:
        return {
            "taskId": task_id,
            "text": "银行行长走过人行道。",
            "language": "Chinese",
            "voiceId": "Vivian",
            "instruction": "自然朗读",
            "outputPath": str(self.output_root / f"{task_id}.wav"),
        }

    def test_health_and_status_do_not_load_model(self) -> None:
        health = self.dispatch("health", "health")
        status = self.dispatch("status", "status")

        self.assertTrue(health["result"]["healthy"])
        self.assertEqual(status["result"]["state"], "uninitialized")
        self.assertEqual(self.loader.load_count, 0)

    def test_output_root_uses_electron_qwen_speech_environment_variable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"ELECTRON_QWEN_SPEECH_TTS_OUTPUT_ROOT": directory},
                clear=True,
            ):
                self.assertEqual(_resolve_output_root(None), Path(directory).resolve())

    def test_tts_model_path_comes_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH": directory},
                clear=True,
            ):
                self.assertEqual(
                    _resolve_local_model_path(
                        None,
                        "ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH",
                        "Qwen TTS",
                    ),
                    Path(directory).resolve(),
                )

    def test_tts_model_path_environment_is_required(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SidecarError) as context:
                _resolve_local_model_path(
                    None,
                    "ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH",
                    "Qwen TTS",
                )
        self.assertEqual(context.exception.code, "INITIALIZATION_FAILED")

    def test_initialize_is_idempotent_and_reports_capabilities(self) -> None:
        first = self.initialize()
        second = self.dispatch("initialize-again", "initialize")

        self.assertEqual(first["result"]["device"], "mps:0")
        self.assertEqual(first["result"]["supportedVoiceIds"], ["vivian", "ryan"])
        self.assertEqual(second["result"]["loadCount"], 1)
        self.assertEqual(self.loader.load_count, 1)

    def test_synthesize_validates_and_writes_wav_metadata(self) -> None:
        self.initialize()
        response = self.dispatch("synthesize", "synthesize", self.synthesis_params())

        self.assertEqual(response["result"]["sampleRate"], 24000)
        self.assertEqual(response["result"]["durationSeconds"], 2)
        self.assertEqual(self.loader.model.calls[0]["speaker"], "Vivian")
        self.assertEqual(self.loader.model.calls[0]["instruct"], "自然朗读")
        self.assertTrue((self.output_root / "tts-task.wav").is_file())

    def test_synthesize_rejects_escape_existing_and_unsupported_values(self) -> None:
        self.initialize()
        escaped = self.synthesis_params("escape")
        escaped["outputPath"] = str(self.output_root.parent / "escape.wav")
        escaped_response = self.dispatch("escape", "synthesize", escaped)
        self.assertEqual(escaped_response["error"]["code"], "INVALID_REQUEST")

        existing_path = self.output_root / "existing.wav"
        existing_path.write_bytes(b"existing")
        existing = self.synthesis_params("existing")
        existing["outputPath"] = str(existing_path)
        existing_response = self.dispatch("existing", "synthesize", existing)
        self.assertEqual(existing_response["error"]["code"], "AUDIO_WRITE_FAILED")

        unsupported = self.synthesis_params("unsupported")
        unsupported["voiceId"] = "Unknown"
        unsupported_response = self.dispatch("unsupported", "synthesize", unsupported)
        self.assertEqual(unsupported_response["error"]["code"], "UNSUPPORTED_VOICE")

    def test_cancel_running_synthesis_discards_output(self) -> None:
        self.service.close_input()
        self.service.join()
        blocking_model = BlockingFakeModel()
        self.loader = FakeLoader(blocking_model)
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=self.loader,
            audio_writer=self.writer,
            output_root=self.output_root,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        self.initialize()
        self.service.dispatch_line(json.dumps({
            "id": "running-request",
            "method": "synthesize",
            "params": self.synthesis_params("running"),
        }))
        self.assertTrue(blocking_model.started.wait(timeout=2))

        cancellation = self.dispatch("cancel", "cancel", {"taskId": "running"})
        self.assertEqual(cancellation["result"]["status"], "cancel_requested")
        blocking_model.release.set()
        response = self.output.wait_for("running-request")

        self.assertEqual(response["error"]["code"], "TASK_CANCELLED")
        self.assertFalse((self.output_root / "running.wav").exists())

    def test_invalid_protocol_and_shutdown(self) -> None:
        self.service.dispatch_line("not-json")
        malformed = self.output.wait_for(None)
        unknown = self.dispatch("unknown", "does-not-exist")

        self.assertEqual(malformed["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(unknown["error"]["code"], "UNKNOWN_METHOD")
        should_stop = self.service.dispatch_line(json.dumps({
            "id": "shutdown",
            "method": "shutdown",
            "params": {},
        }))
        self.assertTrue(should_stop)
        self.assertEqual(self.output.wait_for("shutdown")["result"]["state"], "shutting_down")


if __name__ == "__main__":
    unittest.main()
