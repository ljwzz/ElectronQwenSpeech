from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from tts_sidecar.service import (
    AudioMetadata,
    LoadedTTSModel,
    MLXQwenTTSModelLoader,
    MLX_TTS_MODEL_REVISION,
    PCM16WaveAudioWriter,
    SidecarError,
    SidecarService,
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


class FakeMLXGenerationResult:
    def __init__(self, audio: Any, sample_rate: int = 24000) -> None:
        self.audio = audio
        self.sample_rate = sample_rate


class FakeMLXModel:
    def __init__(self, audio: Any | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.audio = np.asarray(audio if audio is not None else [-0.5, 0.5], dtype=np.float32)

    def generate_custom_voice(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        yield FakeMLXGenerationResult(self.audio)


class FakeMLXLibraryModel:
    def parameters(self) -> dict[str, Any]:
        return {}

    def get_supported_languages(self) -> list[str]:
        return ["auto", "chinese", "english"]

    def get_supported_speakers(self) -> list[str]:
        return ["vivian", "ryan"]


class BlockingFakeMLXModel(FakeMLXModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_custom_voice(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.started.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("test release timeout")
        yield FakeMLXGenerationResult(self.audio)


class FakeMLXLoader:
    def __init__(
        self,
        model: FakeMLXModel | None = None,
        *,
        device: str = "gpu:0",
    ) -> None:
        self.model = model or FakeMLXModel()
        self.device = device
        self.load_count = 0

    def load(self) -> LoadedTTSModel:
        self.load_count += 1
        return LoadedTTSModel(
            model=self.model,
            model_path="/models/mlx/Qwen3-TTS-CustomVoice-bf16",
            device=self.device,
            dtype="bfloat16",
            supported_languages=("chinese", "english"),
            supported_voice_ids=("vivian", "ryan"),
            runtime="mlx",
            accelerator="metal",
            model_format="mlx",
            model_revision=MLX_TTS_MODEL_REVISION,
            model_precision="bfloat16",
            speech_tokenizer_precision="float32",
        )


class FakeWriter:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.waveforms: list[Any] = []

    def write(self, output_path: Path, waveform: Any, sample_rate: int) -> AudioMetadata:
        self.paths.append(output_path)
        self.waveforms.append(waveform)
        output_path.write_bytes(b"RIFF-test-wave")
        return AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            frame_count=48000,
            duration_seconds=2,
        )


@contextmanager
def fake_mlx_modules(
    *,
    model: Any | None = None,
    metal_available: bool = True,
    loaded_paths: list[Path] | None = None,
) -> Any:
    gpu = object()
    core = types.ModuleType("mlx.core")
    core.float32 = np.float32
    core.gpu = gpu
    core.metal = types.SimpleNamespace(is_available=lambda: metal_available)
    core.Device = lambda device_type, index: gpu
    core.device_count = lambda device_type: 1
    core.set_default_device = lambda device: None
    core.default_device = lambda: gpu
    core.array = np.asarray
    core.add = lambda left, right, *, stream: left + right
    core.eval = lambda *values: None

    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    mlx.core = core

    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda parameters: [
        ("talker.weight", types.SimpleNamespace(dtype="bfloat16")),
        ("speech_tokenizer.decoder.weight", types.SimpleNamespace(dtype="float32")),
    ]

    mlx_audio = types.ModuleType("mlx_audio")
    mlx_audio.__path__ = []
    tts = types.ModuleType("mlx_audio.tts")
    tts.__path__ = []
    tts_utils = types.ModuleType("mlx_audio.tts.utils")
    def load_model(path: Path) -> Any:
        if loaded_paths is not None:
            loaded_paths.append(path)
        return model

    tts_utils.load_model = load_model
    mlx_audio.tts = tts
    tts.utils = tts_utils

    with patch.dict(sys.modules, {
        "mlx": mlx,
        "mlx.core": core,
        "mlx.utils": utils,
        "mlx_audio": mlx_audio,
        "mlx_audio.tts": tts,
        "mlx_audio.tts.utils": tts_utils,
    }):
        yield


class SidecarServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary_directory.name)
        self.output = CapturedOutput()
        self.stderr = io.StringIO()
        self.loader = FakeMLXLoader()
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
        self.assertEqual(status["result"]["runtime"], "mlx")
        self.assertEqual(self.loader.load_count, 0)

    def test_output_root_uses_electron_qwen_speech_environment_variable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"ELECTRON_QWEN_SPEECH_TTS_OUTPUT_ROOT": directory},
                clear=True,
            ):
                self.assertEqual(_resolve_output_root(None), Path(directory).resolve())

    def test_initialize_is_idempotent_and_reports_capabilities(self) -> None:
        first = self.initialize()
        second = self.dispatch("initialize-again", "initialize")

        self.assertEqual(first["result"]["device"], "gpu:0")
        self.assertEqual(first["result"]["runtime"], "mlx")
        self.assertEqual(first["result"]["accelerator"], "metal")
        self.assertEqual(first["result"]["modelFormat"], "mlx")
        self.assertEqual(first["result"]["modelRevision"], MLX_TTS_MODEL_REVISION)
        self.assertEqual(first["result"]["modelPrecision"], "bfloat16")
        self.assertEqual(first["result"]["speechTokenizerPrecision"], "float32")
        self.assertEqual(first["result"]["supportedVoiceIds"], ["vivian", "ryan"])
        self.assertEqual(second["result"]["loadCount"], 1)
        self.assertEqual(self.loader.load_count, 1)

    def test_synthesize_validates_and_writes_wav_metadata(self) -> None:
        self.initialize()
        with fake_mlx_modules():
            response = self.dispatch("synthesize", "synthesize", self.synthesis_params())

        self.assertEqual(response["result"]["sampleRate"], 24000)
        self.assertEqual(response["result"]["durationSeconds"], 2)
        self.assertEqual(self.loader.model.calls[0]["speaker"], "Vivian")
        self.assertEqual(self.loader.model.calls[0]["instruct"], "自然朗读")
        self.assertTrue((self.output_root / "tts-task.wav").is_file())

    def test_mlx_runtime_reports_metadata_and_converts_pcm16(self) -> None:
        self.service.close_input()
        self.service.join()
        model = FakeMLXModel(audio=[-2.0, -0.25, 0.25, 2.0])
        self.loader = FakeMLXLoader(model)
        self.output = CapturedOutput()
        self.writer = FakeWriter()
        self.service = SidecarService(
            model_loader=self.loader,
            audio_writer=self.writer,
            output_root=self.output_root,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )

        initialized = self.initialize()
        self.assertEqual(initialized["result"]["runtime"], "mlx")
        self.assertEqual(initialized["result"]["device"], "gpu:0")
        self.assertEqual(initialized["result"]["modelFormat"], "mlx")
        self.assertEqual(
            initialized["result"]["modelRevision"],
            MLX_TTS_MODEL_REVISION,
        )

        with fake_mlx_modules():
            response = self.dispatch("mlx-synthesize", "synthesize", self.synthesis_params("mlx"))

        self.assertEqual(response["result"]["sampleRate"], 24000)
        self.assertEqual(self.writer.waveforms[0].dtype, np.int16)
        self.assertEqual(self.writer.waveforms[0].tolist(), [-32768, -8192, 8192, 32767])

    def test_mlx_runtime_rejects_silent_and_wrong_device_results(self) -> None:
        self.service.close_input()
        self.service.join()
        self.output = CapturedOutput()
        silent_loader = FakeMLXLoader(FakeMLXModel(audio=[0.0, 0.0]))
        self.service = SidecarService(
            model_loader=silent_loader,
            audio_writer=self.writer,
            output_root=self.output_root,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        self.initialize()
        with fake_mlx_modules():
            silent = self.dispatch("silent", "synthesize", self.synthesis_params("silent"))
        self.assertEqual(silent["error"]["code"], "AUDIO_OUTPUT_INVALID")

        self.service.close_input()
        self.service.join()
        self.output = CapturedOutput()
        wrong_device_loader = FakeMLXLoader(device="cpu:0")
        self.service = SidecarService(
            model_loader=wrong_device_loader,
            audio_writer=self.writer,
            output_root=self.output_root,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        mismatch = self.initialize()
        self.assertEqual(mismatch["error"]["code"], "RUNTIME_DEVICE_MISMATCH")

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
        blocking_model = BlockingFakeMLXModel()
        self.loader = FakeMLXLoader(blocking_model)
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
        with fake_mlx_modules():
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

    def test_duplicate_queued_output_path_does_not_delete_first_task_output(self) -> None:
        self.service.close_input()
        self.service.join()
        blocking_model = BlockingFakeMLXModel()
        self.loader = FakeMLXLoader(blocking_model)
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=self.loader,
            audio_writer=PCM16WaveAudioWriter(),
            output_root=self.output_root,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        self.initialize()
        first_params = self.synthesis_params("first-owner")
        second_params = self.synthesis_params("second-owner")
        second_params["outputPath"] = first_params["outputPath"]

        with fake_mlx_modules():
            self.service.dispatch_line(json.dumps({
                "id": "first-owner-request",
                "method": "synthesize",
                "params": first_params,
            }))
            self.assertTrue(blocking_model.started.wait(timeout=2))
            self.service.dispatch_line(json.dumps({
                "id": "second-owner-request",
                "method": "synthesize",
                "params": second_params,
            }))
            blocking_model.release.set()
            first = self.output.wait_for("first-owner-request")
            second = self.output.wait_for("second-owner-request")

        output_path = Path(first_params["outputPath"])
        self.assertEqual(first["result"]["audioPath"], str(output_path.resolve()))
        self.assertEqual(second["error"]["code"], "AUDIO_WRITE_FAILED")
        self.assertTrue(output_path.is_file())
        with wave.open(str(output_path), "rb") as wav_file:
            self.assertGreater(wav_file.getnframes(), 0)

    def test_raced_external_output_file_is_not_deleted(self) -> None:
        self.service.close_input()
        self.service.join()
        blocking_model = BlockingFakeMLXModel()
        self.loader = FakeMLXLoader(blocking_model)
        self.output = CapturedOutput()
        self.service = SidecarService(
            model_loader=self.loader,
            audio_writer=PCM16WaveAudioWriter(),
            output_root=self.output_root,
            stdout=self.output,
            stderr=self.stderr,
            require_python_312=False,
        )
        self.initialize()
        params = self.synthesis_params("external-owner")
        output_path = Path(params["outputPath"])

        with fake_mlx_modules():
            self.service.dispatch_line(json.dumps({
                "id": "external-owner-request",
                "method": "synthesize",
                "params": params,
            }))
            self.assertTrue(blocking_model.started.wait(timeout=2))
            output_path.write_bytes(b"external-file")
            blocking_model.release.set()
            response = self.output.wait_for("external-owner-request")

        self.assertEqual(response["error"]["code"], "AUDIO_WRITE_FAILED")
        self.assertEqual(output_path.read_bytes(), b"external-file")

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


class MLXModelLoaderTest(unittest.TestCase):
    def test_requires_model_path_environment_variable(self) -> None:
        with patch.dict(
            os.environ,
            {"ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH": ""},
        ):
            with self.assertRaises(SidecarError) as context:
                MLXQwenTTSModelLoader()

        self.assertEqual(context.exception.code, "MODEL_INTEGRITY_FAILED")
        self.assertIn("ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH", str(context.exception))

    def test_load_uses_absolute_local_path_and_reports_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory).resolve()
            validated: list[tuple[Path, str]] = []
            loaded_paths: list[Path] = []
            model = FakeMLXLibraryModel()
            loader = MLXQwenTTSModelLoader(
                model_path=model_path,
                integrity_validator=lambda path, revision: validated.append((path, revision)),
            )

            with fake_mlx_modules(model=model, loaded_paths=loaded_paths):
                loaded = loader.load()

        self.assertEqual(validated, [(model_path, MLX_TTS_MODEL_REVISION)])
        self.assertEqual(loaded_paths, [model_path])
        self.assertEqual(loaded.runtime, "mlx")
        self.assertEqual(loaded.accelerator, "metal")
        self.assertEqual(loaded.device, "gpu:0")
        self.assertEqual(loaded.model_precision, "bfloat16")
        self.assertEqual(loaded.speech_tokenizer_precision, "float32")

    def test_integrity_and_accelerator_failures_use_generic_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory).resolve()

            def reject_integrity(path: Path, revision: str) -> None:
                del path, revision
                raise RuntimeError("invalid model")

            integrity_loader = MLXQwenTTSModelLoader(
                model_path=model_path,
                integrity_validator=reject_integrity,
            )
            with self.assertRaises(SidecarError) as integrity_context:
                integrity_loader.load()
            self.assertEqual(integrity_context.exception.code, "MODEL_INTEGRITY_FAILED")

            accelerator_loader = MLXQwenTTSModelLoader(
                model_path=model_path,
                integrity_validator=lambda path, revision: None,
            )
            with fake_mlx_modules(metal_available=False):
                with self.assertRaises(SidecarError) as accelerator_context:
                    accelerator_loader.load()
            self.assertEqual(accelerator_context.exception.code, "ACCELERATOR_UNAVAILABLE")

    def test_offline_mode_is_forced(self) -> None:
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")


class PCM16WaveAudioWriterTest(unittest.TestCase):
    def test_writes_clipped_mono_pcm16_without_soundfile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "audio.wav"
            metadata = PCM16WaveAudioWriter().write(
                output_path,
                np.asarray([-2.0, -0.25, 0.25, 2.0], dtype=np.float32),
                24000,
            )
            with wave.open(str(output_path), "rb") as wav_file:
                frames = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")

        self.assertEqual(metadata.sample_rate, 24000)
        self.assertEqual(metadata.channels, 1)
        self.assertEqual(metadata.frame_count, 4)
        self.assertEqual(frames.tolist(), [-32768, -8192, 8192, 32767])

    def test_rejects_silent_nonfinite_and_wrong_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            writer = PCM16WaveAudioWriter()
            invalid_waveforms = (
                np.zeros(2, dtype=np.float32),
                np.asarray([0.0, np.nan], dtype=np.float32),
            )
            for index, waveform in enumerate(invalid_waveforms):
                with self.subTest(index=index):
                    with self.assertRaises(SidecarError) as context:
                        writer.write(root / f"invalid-{index}.wav", waveform, 24000)
                    self.assertEqual(context.exception.code, "AUDIO_OUTPUT_INVALID")

            with self.assertRaises(SidecarError) as sample_rate_context:
                writer.write(root / "wrong-rate.wav", np.asarray([0.5]), 16000)
            self.assertEqual(sample_rate_context.exception.code, "AUDIO_OUTPUT_INVALID")

    def test_removes_only_partial_output_created_by_this_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "partial.wav"
            with patch(
                "tts_sidecar.service.wave.open",
                side_effect=wave.Error("write failed"),
            ):
                with self.assertRaises(SidecarError) as context:
                    PCM16WaveAudioWriter().write(
                        output_path,
                        np.asarray([0.5, 0.25], dtype=np.float32),
                        24000,
                    )

            self.assertEqual(context.exception.code, "AUDIO_WRITE_FAILED")
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
