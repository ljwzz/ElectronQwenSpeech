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
    MLX_DEVICE,
    MLX_ASR_MODEL_REVISION,
    MLX_FORCED_ALIGNER_MODEL_REVISION,
    MLXModelLoader,
    LoadedModels,
    SidecarError,
    SidecarService,
    _first_valid_language,
    _language_is_supported,
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


class LanguageNormalizationTest(unittest.TestCase):
    def test_ignores_missing_language_sentinels(self) -> None:
        self.assertIsNone(_first_valid_language("None"))
        self.assertIsNone(_first_valid_language(["unknown", "auto"]))
        self.assertEqual(_first_valid_language(["None", "English"]), "English")

    def test_missing_aligner_language_capability_is_a_real_failure(self) -> None:
        with self.assertRaises(SidecarError) as context:
            _language_is_supported("Chinese", None)

        self.assertEqual(context.exception.code, "ALIGNMENT_FAILED")


class FakeMLXASRModel:
    def __init__(self, languages: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.languages = ["Chinese"] if languages is None else languages

    def parameters(self) -> dict[str, Any]:
        return {"weight": SimpleNamespace(dtype="bfloat16")}

    def generate(self, audio: str, **kwargs: Any) -> Any:
        self.calls.append({"audio": audio, **kwargs})
        return SimpleNamespace(
            text="你好这是 MLX 语音识别功能测试",
            language=self.languages,
            segments=[],
        )


class BlockingFakeMLXASRModel(FakeMLXASRModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, audio: str, **kwargs: Any) -> Any:
        self.started.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test did not release fake inference")
        return super().generate(audio, **kwargs)


class FailingFakeMLXASRModel(FakeMLXASRModel):
    def generate(self, audio: str, **kwargs: Any) -> Any:
        raise RuntimeError("fake inference failed")


class FailingSecondFakeMLXASRModel(FakeMLXASRModel):
    def generate(self, audio: str, **kwargs: Any) -> Any:
        if self.calls:
            self.calls.append({"audio": audio, **kwargs})
            raise RuntimeError("fake second inference failed")
        return super().generate(audio, **kwargs)


class FakeMLXAlignerModel:
    def __init__(self, supported_languages: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.supported_languages = (
            ["chinese", "english"]
            if supported_languages is None
            else supported_languages
        )

    def parameters(self) -> dict[str, Any]:
        return {"weight": SimpleNamespace(dtype="bfloat16")}

    def get_supported_languages(self) -> list[str]:
        return self.supported_languages

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            items=[
                SimpleNamespace(text="你好", start_time=0.1, end_time=0.4),
                SimpleNamespace(text="MLX", start_time=0.5, end_time=0.9),
            ]
        )


class FailingFakeMLXAlignerModel(FakeMLXAlignerModel):
    def generate(self, **kwargs: Any) -> Any:
        raise RuntimeError("fake MLX alignment failed")


class FakeMLXCore:
    gpu = "gpu"

    def __init__(self) -> None:
        self._default_device: Any = ("cpu", 0)
        self.metal = SimpleNamespace(is_available=lambda: True)

    def Device(self, device_type: Any, index: int) -> tuple[Any, int]:
        return device_type, index

    def device_count(self, device_type: Any) -> int:
        return 1 if device_type == self.gpu else 0

    def is_available(self, device: Any) -> bool:
        return device == (self.gpu, 0)

    def set_default_device(self, device: Any) -> None:
        self._default_device = device

    def default_device(self) -> Any:
        return self._default_device


class FakeMLXLoader:
    def __init__(
        self,
        asr_model: FakeMLXASRModel | None = None,
        aligner_model: FakeMLXAlignerModel | None = None,
    ) -> None:
        self.asr_model = asr_model or FakeMLXASRModel()
        self.aligner_model = aligner_model or FakeMLXAlignerModel()
        self.load_count = 0

    def load(self) -> LoadedModels:
        self.load_count += 1
        return LoadedModels(
            asr_model=self.asr_model,
            aligner_model=self.aligner_model,
            asr_model_path="/models/mlx/Qwen3-ASR-1.7B-bf16",
            aligner_model_path="/models/mlx/Qwen3-ForcedAligner-0.6B-bf16",
            asr_device="gpu:0",
            aligner_device="gpu:0",
            asr_dtype="bfloat16",
            aligner_dtype="bfloat16",
            runtime="mlx",
            accelerator="metal",
            device="gpu:0",
            model_format="mlx",
            asr_model_revision=MLX_ASR_MODEL_REVISION,
            aligner_model_revision=MLX_FORCED_ALIGNER_MODEL_REVISION,
        )


class AcceleratorUnavailableLoader(FakeMLXLoader):
    def load(self) -> LoadedModels:
        raise SidecarError(
            "ACCELERATOR_UNAVAILABLE",
            "accelerator unavailable in fake loader",
        )


class MLXModelLoaderTest(unittest.TestCase):
    def test_requires_model_path_environment_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH": "",
                "ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH": "",
            },
        ):
            with self.assertRaises(SidecarError) as context:
                MLXModelLoader().load()

        self.assertEqual(context.exception.code, "MODEL_INTEGRITY_FAILED")
        self.assertIn("ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH", str(context.exception))

    def test_loads_absolute_local_models_on_metal_gpu_zero_in_offline_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            asr_path = root / "asr"
            aligner_path = root / "aligner"
            asr_path.mkdir()
            aligner_path.mkdir()
            fake_mx = FakeMLXCore()
            asr_model = FakeMLXASRModel()
            aligner_model = FakeMLXAlignerModel()
            loaded_paths: list[str] = []
            validation_calls: list[tuple[Path, str]] = []

            def load_model(path: str) -> Any:
                loaded_paths.append(path)
                return asr_model if path == str(asr_path) else aligner_model

            def validate(path: Path, revision: str) -> None:
                validation_calls.append((path, revision))

            with patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "0",
                    "TRANSFORMERS_OFFLINE": "0",
                    "HF_DATASETS_OFFLINE": "0",
                },
            ):
                loaded = MLXModelLoader(
                    asr_model_path=asr_path,
                    aligner_model_path=aligner_path,
                    integrity_validator=validate,
                    mlx_core=fake_mx,
                    load_function=load_model,
                ).load()

                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")

        self.assertEqual(
            loaded_paths,
            [str(asr_path.resolve()), str(aligner_path.resolve())],
        )
        self.assertEqual(
            validation_calls,
            [
                (asr_path.resolve(), MLX_ASR_MODEL_REVISION),
                (aligner_path.resolve(), MLX_FORCED_ALIGNER_MODEL_REVISION),
            ],
        )
        self.assertEqual(fake_mx.default_device(), (fake_mx.gpu, 0))
        self.assertEqual(loaded.runtime, "mlx")
        self.assertEqual(loaded.accelerator, "metal")
        self.assertEqual(loaded.device, MLX_DEVICE)
        self.assertEqual(loaded.model_format, "mlx")
        self.assertEqual(loaded.asr_dtype, "bfloat16")
        self.assertEqual(loaded.aligner_dtype, "bfloat16")

    def test_default_integrity_preflight_runs_for_each_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            asr_path = root / "asr"
            aligner_path = root / "aligner"
            asr_path.mkdir()
            aligner_path.mkdir()
            models = iter([FakeMLXASRModel(), FakeMLXAlignerModel()])

            def fake_preflight(command: list[str], **kwargs: Any) -> Any:
                model_key = command[command.index("--model") + 1]
                path = Path(command[command.index(f"--{model_key}-path") + 1])
                revision = (
                    MLX_ASR_MODEL_REVISION
                    if model_key == "asr"
                    else MLX_FORCED_ALIGNER_MODEL_REVISION
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "ok": True,
                            "models": [
                                {
                                    "model": model_key,
                                    "path": str(path),
                                    "revision": revision,
                                }
                            ],
                        }
                    ),
                )

            with patch(
                "asr_sidecar.service.subprocess.run",
                side_effect=fake_preflight,
            ) as run:
                loaded = MLXModelLoader(
                    asr_model_path=asr_path,
                    aligner_model_path=aligner_path,
                    mlx_core=FakeMLXCore(),
                    load_function=lambda _path: next(models),
                ).load()

        self.assertEqual(run.call_count, 2)
        self.assertEqual(loaded.asr_model_revision, MLX_ASR_MODEL_REVISION)
        self.assertEqual(
            loaded.aligner_model_revision,
            MLX_FORCED_ALIGNER_MODEL_REVISION,
        )

    def test_preflight_failure_stops_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            load_calls: list[str] = []
            with patch(
                "asr_sidecar.service.subprocess.run",
                return_value=SimpleNamespace(returncode=1, stdout=""),
            ):
                with self.assertRaises(SidecarError) as captured:
                    MLXModelLoader(
                        asr_model_path=model_path,
                        aligner_model_path=model_path,
                        mlx_core=FakeMLXCore(),
                        load_function=lambda path: load_calls.append(path),
                    ).load()

        self.assertEqual(captured.exception.code, "MODEL_INTEGRITY_FAILED")
        self.assertEqual(load_calls, [])



class SidecarServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output = CapturedOutput()
        self.stderr = io.StringIO()
        self.loader = FakeMLXLoader()
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

    def test_accelerator_unavailable_is_structured_and_state_becomes_failed(self) -> None:
        self.service.close_input()
        self.service.join()
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=AcceleratorUnavailableLoader(),
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )

        initialization = self.dispatch("accelerator-unavailable", "initialize")
        status = self.dispatch("status-after-accelerator-error", "status")

        self.assertEqual(initialization["error"]["code"], "ACCELERATOR_UNAVAILABLE")
        self.assertEqual(status["result"]["state"], "failed")
        self.assertFalse(status["result"]["initialized"])

    def test_initialize_is_idempotent_and_loads_models_once(self) -> None:
        first = self.initialize()
        second = self.dispatch("initialize-again", "initialize")
        status = self.dispatch("status-after-init", "status")

        self.assertEqual(first["result"]["state"], "ready")
        self.assertEqual(first["result"]["runtime"], "mlx")
        self.assertEqual(first["result"]["accelerator"], "metal")
        self.assertEqual(first["result"]["device"], "gpu:0")
        self.assertEqual(first["result"]["modelFormat"], "mlx")
        self.assertEqual(first["result"]["asrModelPrecision"], "bfloat16")
        self.assertEqual(first["result"]["asrDevice"], "gpu:0")
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
                {"text": "MLX", "start": 0.5, "end": 0.9},
            ],
        )
        self.assertEqual(self.loader.asr_model.calls[0]["language"], "Chinese")
        self.assertEqual(
            self.loader.aligner_model.calls[0],
            {
                "audio": str(audio_path),
                "text": "你好这是 MLX 语音识别功能测试",
                "language": "Chinese",
            },
        )
        self.assertEqual(transcription["result"]["alignmentStatus"], "aligned")

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
            [call["audio"] for call in self.loader.asr_model.calls],
            [str(first_audio), str(second_audio)],
        )
        self.assertNotIn("language", self.loader.asr_model.calls[0])
        self.assertEqual(self.loader.asr_model.calls[1]["language"], "Chinese")
        self.assertEqual(
            [call["audio"] for call in self.loader.aligner_model.calls],
            [str(first_audio), str(second_audio)],
        )
        self.assertEqual(
            [item["alignmentStatus"] for item in transcription["result"]["items"]],
            ["aligned", "aligned"],
        )

    def test_mlx_transcribes_then_aligns_with_first_detected_language(self) -> None:
        self.service.close_input()
        self.service.join()
        mlx_loader = FakeMLXLoader(FakeMLXASRModel(["", "Chinese"]))
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=mlx_loader,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "fixture.wav"
            audio_path.write_bytes(b"RIFF-fake")
            initialized = self.initialize()
            transcription = self.dispatch(
                "mlx-transcribe",
                "transcribe",
                {"taskId": "mlx-task", "audioPath": str(audio_path)},
            )

        self.assertEqual(initialized["result"]["runtime"], "mlx")
        self.assertEqual(initialized["result"]["accelerator"], "metal")
        self.assertEqual(initialized["result"]["device"], "gpu:0")
        self.assertEqual(initialized["result"]["modelFormat"], "mlx")
        self.assertEqual(
            initialized["result"]["asrModelRevision"],
            MLX_ASR_MODEL_REVISION,
        )
        self.assertEqual(transcription["result"]["language"], "Chinese")
        self.assertEqual(transcription["result"]["alignmentStatus"], "aligned")
        self.assertEqual(
            mlx_loader.aligner_model.calls[0],
            {
                "audio": str(audio_path),
                "text": "你好这是 MLX 语音识别功能测试",
                "language": "Chinese",
            },
        )

    def test_requested_language_precedes_detection_and_unsupported_skips_aligner(self) -> None:
        self.service.close_input()
        self.service.join()
        mlx_loader = FakeMLXLoader(
            FakeMLXASRModel(["Chinese"]),
            FakeMLXAlignerModel(["chinese"]),
        )
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=mlx_loader,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "fixture.wav"
            audio_path.write_bytes(b"RIFF-fake")
            self.initialize()
            transcription = self.dispatch(
                "unsupported-language",
                "transcribe",
                {
                    "taskId": "unsupported-task",
                    "audioPath": str(audio_path),
                    "language": "German",
                },
            )

        self.assertEqual(mlx_loader.asr_model.calls[0]["language"], "German")
        self.assertEqual(transcription["result"]["language"], "German")
        self.assertEqual(transcription["result"]["timestamps"], [])
        self.assertEqual(
            transcription["result"]["alignmentStatus"],
            "unsupported_language",
        )
        self.assertEqual(mlx_loader.aligner_model.calls, [])

    def test_alignment_failure_is_distinct_from_transcription_failure(self) -> None:
        self.service.close_input()
        self.service.join()
        loader = FakeMLXLoader(aligner_model=FailingFakeMLXAlignerModel())
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
                "failed-alignment",
                "transcribe",
                {"taskId": "failed-alignment", "audioPath": str(audio_path)},
            )

        self.assertEqual(failed["error"]["code"], "ALIGNMENT_FAILED")
        self.assertIn("Traceback", self.stderr.getvalue())

    def test_batch_runs_each_asr_then_aligner_and_fails_as_one_request(self) -> None:
        self.service.close_input()
        self.service.join()
        asr_model = FailingSecondFakeMLXASRModel()
        aligner_model = FakeMLXAlignerModel()
        loader = FakeMLXLoader(asr_model, aligner_model)
        self.output = CapturedOutput()
        self.stderr = io.StringIO()
        self.service = SidecarService(
            model_loader=loader,
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
            failed = self.dispatch(
                "failed-batch",
                "transcribe",
                {
                    "taskId": "failed-batch",
                    "items": [
                        {"itemId": "first", "audioPath": str(first_audio)},
                        {"itemId": "second", "audioPath": str(second_audio)},
                    ],
                },
            )

        self.assertEqual(failed["error"]["code"], "TRANSCRIPTION_FAILED")
        self.assertEqual(len(asr_model.calls), 2)
        self.assertEqual(len(aligner_model.calls), 1)

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
        loader = FakeMLXLoader(FailingFakeMLXASRModel())
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
        blocking_model = BlockingFakeMLXASRModel()
        self.loader = FakeMLXLoader(blocking_model)
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
        blocking_model = BlockingFakeMLXASRModel()
        self.loader = FakeMLXLoader(blocking_model)
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
        self.assertIsInstance(blocking_model.calls[0]["audio"], str)

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
