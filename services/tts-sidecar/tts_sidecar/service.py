from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import wave
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

SpeechRuntime = Literal["mlx"]

MLX_TTS_MODEL_REVISION = "52f4770fd9726457eae3d3b6aa92047a25a10776"
REQUIRED_PYTHON_VERSION = (3, 12)
MLX_MODEL_PATH_ENV = "ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH"
MLX_ACCELERATOR = "metal"
MLX_DEVICE = "gpu:0"
MODEL_PRECISION = "bfloat16"
SPEECH_TOKENIZER_PRECISION = "float32"
EXPECTED_SAMPLE_RATE = 24_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_VALIDATOR_PATH = REPOSITORY_ROOT / "scripts" / "validate_mlx_speech_models.py"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class SidecarError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details


@dataclass(frozen=True)
class LoadedTTSModel:
    model: Any
    model_path: str
    device: str
    dtype: str
    supported_languages: tuple[str, ...]
    supported_voice_ids: tuple[str, ...]
    runtime: SpeechRuntime
    accelerator: str
    model_format: str
    model_revision: str | None
    model_precision: str = MODEL_PRECISION
    speech_tokenizer_precision: str = SPEECH_TOKENIZER_PRECISION


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    frame_count: int
    duration_seconds: float


class ModelLoader(Protocol):
    def load(self) -> LoadedTTSModel: ...


class ModelIntegrityValidator(Protocol):
    def __call__(self, model_path: Path, expected_revision: str) -> None: ...


class AudioWriter(Protocol):
    def write(self, output_path: Path, waveform: Any, sample_rate: int) -> AudioMetadata: ...


class MLXQwenTTSModelLoader:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        integrity_validator: ModelIntegrityValidator | None = None,
    ) -> None:
        configured_path = model_path
        if configured_path is None:
            configured_value = os.environ.get(MLX_MODEL_PATH_ENV, "").strip()
            if not configured_value:
                raise SidecarError(
                    "MODEL_INTEGRITY_FAILED",
                    f"{MLX_MODEL_PATH_ENV} 未配置。",
                    details={"environmentVariable": MLX_MODEL_PATH_ENV},
                )
            configured_path = Path(configured_value)
        expanded_path = configured_path.expanduser()
        self._model_path_is_absolute = expanded_path.is_absolute()
        self._model_path = expanded_path.resolve(strict=False)
        self._integrity_validator = integrity_validator or _validate_mlx_model_integrity

    def load(self) -> LoadedTTSModel:
        if not self._model_path_is_absolute or not self._model_path.is_dir():
            raise SidecarError(
                "MODEL_INTEGRITY_FAILED",
                "本地 MLX Qwen TTS 模型目录不存在或不是绝对路径。",
                details={"path": str(self._model_path)},
            )
        self._validate_integrity()

        try:
            import mlx.core as mx
        except Exception as error:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "无法导入 MLX。",
                details={"exceptionType": type(error).__name__},
            ) from error

        try:
            metal_available = bool(mx.metal.is_available())
        except Exception as error:
            raise SidecarError(
                "ACCELERATOR_UNAVAILABLE",
                "无法探测 MLX Metal 后端。",
                details={"exceptionType": type(error).__name__},
            ) from error
        if not metal_available:
            raise SidecarError(
                "ACCELERATOR_UNAVAILABLE",
                "当前 MLX 环境不可使用 Metal。",
            )

        try:
            if mx.device_count(mx.gpu) < 1:
                raise RuntimeError("MLX did not report a gpu:0 device")
            gpu_device = mx.Device(mx.gpu, 0)
            mx.set_default_device(gpu_device)
            default_device = mx.default_device()
            probe = mx.add(mx.array([1.0]), mx.array([1.0]), stream=gpu_device)
            mx.eval(probe)
            if float(probe[0].item()) != 2.0:
                raise RuntimeError("MLX gpu:0 probe returned an unexpected value")
        except Exception as error:
            raise SidecarError(
                "ACCELERATOR_UNAVAILABLE",
                "无法选择 MLX gpu:0。",
                details={"exceptionType": type(error).__name__},
            ) from error
        if default_device != gpu_device:
            raise SidecarError(
                "RUNTIME_DEVICE_MISMATCH",
                "MLX 默认设备必须为 gpu:0。",
                details={"defaultDevice": str(default_device)},
            )

        try:
            from mlx_audio.tts.utils import load_model

            model = load_model(self._model_path)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise SidecarError(
                "MODEL_INTEGRITY_FAILED",
                "MLX Qwen TTS 模型文件无效。",
                details={"exceptionType": type(error).__name__},
            ) from error
        except Exception as error:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "MLX Qwen TTS 模型初始化失败。",
                details={"exceptionType": type(error).__name__},
            ) from error

        model_precision, speech_tokenizer_precision = _mlx_model_precisions(model)
        supported_languages = model.get_supported_languages()
        supported_voice_ids = model.get_supported_speakers()
        if not supported_languages or not supported_voice_ids:
            raise SidecarError(
                "MODEL_INTEGRITY_FAILED",
                "MLX CustomVoice 模型未返回语言或音色 capability。",
            )
        return LoadedTTSModel(
            model=model,
            model_path=str(self._model_path),
            device=MLX_DEVICE,
            dtype=MODEL_PRECISION,
            supported_languages=tuple(str(value) for value in supported_languages),
            supported_voice_ids=tuple(str(value) for value in supported_voice_ids),
            runtime="mlx",
            accelerator=MLX_ACCELERATOR,
            model_format="mlx",
            model_revision=MLX_TTS_MODEL_REVISION,
            model_precision=model_precision,
            speech_tokenizer_precision=speech_tokenizer_precision,
        )

    def _validate_integrity(self) -> None:
        try:
            self._integrity_validator(self._model_path, MLX_TTS_MODEL_REVISION)
        except SidecarError:
            raise
        except Exception as error:
            raise SidecarError(
                "MODEL_INTEGRITY_FAILED",
                "MLX Qwen TTS 模型预检失败。",
                details={"exceptionType": type(error).__name__},
            ) from error


class PCM16WaveAudioWriter:
    def write(self, output_path: Path, waveform: Any, sample_rate: int) -> AudioMetadata:
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise SidecarError(
                "AUDIO_OUTPUT_INVALID",
                "TTS 输出采样率必须为 24000 Hz。",
                details={"sampleRate": sample_rate},
            )
        pcm16 = _waveform_to_pcm16(waveform)
        created_output = False
        try:
            with output_path.open("xb") as output_file:
                created_output = True
                with wave.open(output_file, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm16.astype("<i2", copy=False).tobytes())
        except FileExistsError as error:
            raise SidecarError(
                "AUDIO_WRITE_FAILED",
                "TTS 输出路径已存在。",
                details={"path": str(output_path)},
            ) from error
        except Exception as error:
            if created_output:
                _safe_unlink(output_path)
            raise SidecarError(
                "AUDIO_WRITE_FAILED",
                "无法写入 TTS WAV。",
                details={"exceptionType": type(error).__name__},
            ) from error

        try:
            with wave.open(str(output_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                written_sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                compression_type = wav_file.getcomptype()
        except (OSError, wave.Error) as error:
            if created_output:
                _safe_unlink(output_path)
            raise SidecarError(
                "AUDIO_OUTPUT_INVALID",
                "无法读取 TTS 输出 WAV 元数据。",
                details={"exceptionType": type(error).__name__},
            ) from error
        duration_seconds = frame_count / written_sample_rate if written_sample_rate else 0.0
        if (
            channels != 1
            or sample_width != 2
            or written_sample_rate != EXPECTED_SAMPLE_RATE
            or compression_type != "NONE"
            or frame_count <= 0
            or not math.isfinite(duration_seconds)
            or duration_seconds <= 0
        ):
            if created_output:
                _safe_unlink(output_path)
            raise SidecarError("AUDIO_OUTPUT_INVALID", "TTS 输出 WAV 元数据无效。")
        return AudioMetadata(
            sample_rate=written_sample_rate,
            channels=channels,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
        )


SoundFileAudioWriter = PCM16WaveAudioWriter


@dataclass
class WorkItem:
    request_id: str
    method: str
    params: dict[str, Any]
    task_id: str | None = None
    cancelled: bool = False
    response_sent: bool = False


_STOP = object()


class SidecarService:
    def __init__(
        self,
        *,
        model_loader: ModelLoader | None = None,
        model_integrity_validator: ModelIntegrityValidator | None = None,
        audio_writer: AudioWriter | None = None,
        output_root: Path | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        require_python_312: bool = True,
    ) -> None:
        self._model_loader = model_loader or MLXQwenTTSModelLoader(
            integrity_validator=model_integrity_validator,
        )
        self._audio_writer = audio_writer or PCM16WaveAudioWriter()
        self._output_root = _resolve_output_root(output_root)
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._require_python_312 = require_python_312
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._queue: queue.Queue[WorkItem | object] = queue.Queue()
        self._queued_items: dict[str, WorkItem] = {}
        self._pending_items: list[WorkItem] = []
        self._loaded_model: LoadedTTSModel | None = None
        self._load_count = 0
        self._state = "uninitialized"
        self._current_task_id: str | None = None
        self._cancel_requested = False
        self._stopping = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="qwen-tts-worker",
            daemon=False,
        )
        self._worker.start()

    def dispatch_line(self, line: str) -> bool:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            self._emit_error(
                None,
                SidecarError(
                    "INVALID_REQUEST",
                    "请求不是有效的 JSON。",
                    details={"line": error.lineno, "column": error.colno},
                ),
            )
            return False

        request_id: str | None = None
        try:
            if isinstance(request, dict):
                candidate = request.get("id")
                if isinstance(candidate, str) and candidate.strip():
                    request_id = candidate
            request_id, method, params = self._validate_request(request)
            self._check_python_version()
            return self._dispatch(request_id, method, params)
        except SidecarError as error:
            self._emit_error(request_id, error)
            return False

    def close_input(self) -> None:
        self._begin_shutdown(emit_response=False)

    def join(self) -> None:
        self._worker.join()

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return self._snapshot_locked()

    def _validate_request(self, request: Any) -> tuple[str, str, dict[str, Any]]:
        if not isinstance(request, dict):
            raise SidecarError("INVALID_REQUEST", "请求必须是 JSON 对象。")
        unknown_keys = set(request) - {"id", "method", "params"}
        if unknown_keys:
            raise SidecarError(
                "INVALID_REQUEST",
                "请求包含未知字段。",
                details={"keys": sorted(unknown_keys)},
            )
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(request_id, str) or not request_id.strip():
            raise SidecarError("INVALID_REQUEST", "请求 id 必须是非空字符串。")
        if not isinstance(method, str) or not method.strip():
            raise SidecarError("INVALID_REQUEST", "请求 method 必须是非空字符串。")
        if not isinstance(params, dict):
            raise SidecarError("INVALID_REQUEST", "请求 params 必须是 JSON 对象。")
        return request_id, method, params

    def _check_python_version(self) -> None:
        if self._require_python_312 and sys.version_info[:2] != REQUIRED_PYTHON_VERSION:
            raise SidecarError(
                "PYTHON_VERSION_UNSUPPORTED",
                "Qwen TTS Sidecar 必须使用 Python 3.12。",
                details={
                    "required": "3.12",
                    "actual": f"{sys.version_info.major}.{sys.version_info.minor}",
                },
            )

    def _dispatch(self, request_id: str, method: str, params: dict[str, Any]) -> bool:
        if method == "health":
            self._require_empty_params(params)
            self._emit_result(request_id, {"healthy": not self._stopping, **self.snapshot()})
            return False
        if method == "status":
            self._require_empty_params(params)
            self._emit_result(request_id, self.snapshot())
            return False
        if method == "cancel":
            self._handle_cancel(request_id, params)
            return False
        if method == "shutdown":
            self._require_empty_params(params)
            self._begin_shutdown(request_id=request_id, emit_response=True)
            return True
        if method == "initialize":
            self._require_empty_params(params)
            with self._state_lock:
                if self._stopping:
                    raise SidecarError("SIDECAR_UNAVAILABLE", "Sidecar 正在退出。")
                if self._loaded_model is None:
                    self._state = "initializing"
            self._enqueue(WorkItem(request_id=request_id, method=method, params=params))
            return False
        if method == "synthesize":
            normalized = self._normalize_synthesis_params(params)
            task_id = normalized["taskId"]
            item = WorkItem(
                request_id=request_id,
                method=method,
                params=normalized,
                task_id=task_id,
            )
            with self._state_lock:
                if self._stopping:
                    raise SidecarError("SIDECAR_UNAVAILABLE", "Sidecar 正在退出。")
                if task_id == self._current_task_id or task_id in self._queued_items:
                    raise SidecarError("INVALID_REQUEST", "taskId 已存在。")
                self._queued_items[task_id] = item
            self._enqueue(item)
            return False
        raise SidecarError(
            "UNKNOWN_METHOD",
            f"未知方法：{method}",
            details={"method": method},
        )

    def _enqueue(self, item: WorkItem) -> None:
        with self._state_lock:
            self._pending_items.append(item)
        self._queue.put(item)

    def _require_empty_params(self, params: dict[str, Any]) -> None:
        if params:
            raise SidecarError("INVALID_REQUEST", "该方法不接受参数。")

    def _handle_cancel(self, request_id: str, params: dict[str, Any]) -> None:
        if set(params) != {"taskId"}:
            raise SidecarError("INVALID_REQUEST", "cancel 只接受 taskId。")
        task_id = _required_string(params, "taskId")
        cancelled_item: WorkItem | None = None
        with self._state_lock:
            if self._current_task_id == task_id:
                self._cancel_requested = True
                status = "cancel_requested"
            else:
                cancelled_item = self._queued_items.pop(task_id, None)
                if cancelled_item is None:
                    status = "not_found"
                else:
                    cancelled_item.cancelled = True
                    cancelled_item.response_sent = True
                    status = "cancelled"
        if cancelled_item is not None:
            self._emit_error(
                cancelled_item.request_id,
                SidecarError(
                    "TASK_CANCELLED",
                    "排队中的合成任务已取消。",
                    details={"taskId": task_id},
                ),
            )
        self._emit_result(request_id, {"taskId": task_id, "status": status})

    def _begin_shutdown(
        self,
        request_id: str | None = None,
        *,
        emit_response: bool,
    ) -> None:
        cancelled_items: list[WorkItem] = []
        with self._state_lock:
            if not self._stopping:
                self._stopping = True
                self._state = "shutting_down"
                self._cancel_requested = self._current_task_id is not None
                for item in self._pending_items:
                    if item.cancelled or item.response_sent:
                        continue
                    item.cancelled = True
                    item.response_sent = True
                    if item.task_id is not None:
                        self._queued_items.pop(item.task_id, None)
                    cancelled_items.append(item)
                self._queue.put(_STOP)
        for item in cancelled_items:
            code = "TASK_CANCELLED" if item.task_id is not None else "SIDECAR_UNAVAILABLE"
            self._emit_error(item.request_id, SidecarError(code, "Sidecar 退出时取消了排队请求。"))
        if emit_response and request_id is not None:
            self._emit_result(request_id, {"state": "shutting_down"})

    def _worker_loop(self) -> None:
        while True:
            queued = self._queue.get()
            if queued is _STOP:
                return
            item = queued
            assert isinstance(item, WorkItem)
            with self._state_lock:
                if item in self._pending_items:
                    self._pending_items.remove(item)
                if item.cancelled:
                    continue
                if item.task_id is not None:
                    self._queued_items.pop(item.task_id, None)
                    self._current_task_id = item.task_id
                    self._cancel_requested = False
                    self._state = "busy"

            result: Any = None
            failure: SidecarError | None = None
            try:
                result = self._initialize() if item.method == "initialize" else self._synthesize(item)
            except SidecarError as error:
                if error.__cause__ is not None:
                    traceback.print_exception(error, file=self._stderr)
                failure = error
            except Exception as error:
                traceback.print_exc(file=self._stderr)
                code = "INITIALIZATION_FAILED" if item.method == "initialize" else "SYNTHESIS_FAILED"
                failure = SidecarError(
                    code,
                    "Sidecar 工作线程执行失败。",
                    details={"exceptionType": type(error).__name__},
                )
            finally:
                with self._state_lock:
                    if item.task_id is not None:
                        if self._cancel_requested and failure is None:
                            failure = SidecarError(
                                "TASK_CANCELLED",
                                "合成任务已取消，推理结果已丢弃。",
                                details={"taskId": item.task_id},
                            )
                            _safe_unlink(Path(item.params["outputPath"]))
                        self._current_task_id = None
                        self._cancel_requested = False
                    if self._stopping:
                        self._state = "shutting_down"
                    elif self._loaded_model is not None:
                        self._state = "ready"
                    elif item.method == "initialize":
                        self._state = "failed"
                    else:
                        self._state = "uninitialized"

            if not item.response_sent:
                if failure is None:
                    self._emit_result(item.request_id, result)
                else:
                    self._emit_error(item.request_id, failure)
                item.response_sent = True

    def _initialize(self) -> dict[str, Any]:
        with self._state_lock:
            loaded_model = self._loaded_model
        if loaded_model is None:
            with redirect_stdout(self._stderr):
                candidate = self._model_loader.load()
            _validate_loaded_model(candidate)
            with self._state_lock:
                if self._loaded_model is None:
                    self._loaded_model = candidate
                    self._load_count += 1
                loaded_model = self._loaded_model
        assert loaded_model is not None
        with self._state_lock:
            self._state = "ready"
            load_count = self._load_count
        return {
            "state": "ready",
            "runtime": loaded_model.runtime,
            "accelerator": loaded_model.accelerator,
            "modelPath": loaded_model.model_path,
            "device": loaded_model.device,
            "dtype": loaded_model.dtype,
            "modelFormat": loaded_model.model_format,
            "modelRevision": loaded_model.model_revision,
            "modelPrecision": loaded_model.model_precision,
            "speechTokenizerPrecision": loaded_model.speech_tokenizer_precision,
            "supportedLanguages": list(loaded_model.supported_languages),
            "supportedVoiceIds": list(loaded_model.supported_voice_ids),
            "loadCount": load_count,
        }

    def _synthesize(self, item: WorkItem) -> dict[str, Any]:
        with self._state_lock:
            loaded_model = self._loaded_model
            cancelled_before_start = self._cancel_requested
        if loaded_model is None:
            raise SidecarError("NOT_INITIALIZED", "TTS Provider 尚未初始化。")
        if cancelled_before_start:
            raise SidecarError("TASK_CANCELLED", "合成任务已取消。")

        language = item.params["language"]
        voice_id = item.params["voiceId"]
        if language.casefold() not in {value.casefold() for value in loaded_model.supported_languages}:
            raise SidecarError(
                "UNSUPPORTED_LANGUAGE",
                "CustomVoice 不支持请求语言。",
                details={"language": language},
            )
        if voice_id.casefold() not in {value.casefold() for value in loaded_model.supported_voice_ids}:
            raise SidecarError(
                "UNSUPPORTED_VOICE",
                "CustomVoice 不支持请求音色。",
                details={"voiceId": voice_id},
            )

        output_path = Path(item.params["outputPath"])
        started_at = time.perf_counter()
        output_created = False
        try:
            kwargs: dict[str, Any] = {
                "text": item.params["text"],
                "language": language,
                "speaker": voice_id,
            }
            if "instruction" in item.params:
                kwargs["instruct"] = item.params["instruction"]
            with redirect_stdout(self._stderr):
                waveform, sample_rate = _generate_custom_voice(loaded_model, kwargs)
            generation_seconds = time.perf_counter() - started_at
            with self._state_lock:
                if self._cancel_requested:
                    raise SidecarError(
                        "TASK_CANCELLED",
                        "合成任务已取消，推理结果已丢弃。",
                        details={"taskId": item.task_id},
                    )
            if not isinstance(sample_rate, int) or sample_rate != EXPECTED_SAMPLE_RATE:
                raise SidecarError(
                    "AUDIO_OUTPUT_INVALID",
                    "CustomVoice 必须返回 24000 Hz 音频。",
                    details={"sampleRate": sample_rate},
                )
            metadata = self._audio_writer.write(output_path, waveform, sample_rate)
            output_created = True
            with self._state_lock:
                if self._cancel_requested:
                    _safe_unlink(output_path)
                    output_created = False
                    raise SidecarError(
                        "TASK_CANCELLED",
                        "合成任务已取消，WAV 已删除。",
                        details={"taskId": item.task_id},
                    )
        except SidecarError:
            if output_created:
                _safe_unlink(output_path)
            raise
        except Exception as error:
            if output_created:
                _safe_unlink(output_path)
            raise SidecarError(
                "SYNTHESIS_FAILED",
                "Qwen TTS 合成失败。",
                details={"exceptionType": type(error).__name__},
            ) from error

        return {
            "taskId": item.task_id,
            "audioPath": str(output_path),
            "mimeType": "audio/wav",
            "sampleRate": metadata.sample_rate,
            "channels": metadata.channels,
            "frameCount": metadata.frame_count,
            "durationSeconds": metadata.duration_seconds,
            "generationSeconds": generation_seconds,
        }

    def _normalize_synthesis_params(self, params: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"taskId", "text", "language", "voiceId", "instruction", "outputPath"}
        unknown_keys = set(params) - allowed_keys
        if unknown_keys:
            raise SidecarError(
                "INVALID_REQUEST",
                "synthesize 包含未知参数。",
                details={"keys": sorted(unknown_keys)},
            )
        normalized: dict[str, Any] = {
            "taskId": _required_string(params, "taskId"),
            "text": _required_string(params, "text"),
            "language": _required_string(params, "language"),
            "voiceId": _required_string(params, "voiceId"),
            "outputPath": str(self._require_output_path(_required_string(params, "outputPath"))),
        }
        if "instruction" in params:
            normalized["instruction"] = _required_string(params, "instruction")
        return normalized

    def _require_output_path(self, value: str) -> Path:
        output_path = Path(value)
        resolved = output_path.resolve(strict=False)
        if resolved.parent != self._output_root or resolved.suffix.casefold() != ".wav":
            raise SidecarError(
                "INVALID_REQUEST",
                "TTS 输出路径必须是会话输出目录中的 WAV 文件。",
                details={"path": str(output_path)},
            )
        if resolved.exists():
            raise SidecarError(
                "AUDIO_WRITE_FAILED",
                "TTS 输出路径已存在。",
                details={"path": str(resolved)},
            )
        return resolved

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "runtime": "mlx",
            "initialized": self._loaded_model is not None,
            "currentTaskId": self._current_task_id,
            "queuedTaskCount": len(self._queued_items),
            "cancelRequested": self._cancel_requested,
            "loadCount": self._load_count,
        }

    def _emit_result(self, request_id: str, result: Any) -> None:
        self._emit({"id": request_id, "result": result})

    def _emit_error(self, request_id: str | None, error: SidecarError) -> None:
        serialized: dict[str, Any] = {
            "code": error.code,
            "message": str(error),
            "retryable": error.retryable,
        }
        if error.details is not None:
            serialized["details"] = error.details
        self._emit({"id": request_id, "error": serialized})

    def _emit(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            self._stdout.write(f"{line}\n")
            self._stdout.flush()


def _validate_loaded_model(model: LoadedTTSModel) -> None:
    if (
        model.runtime != "mlx"
        or model.accelerator != MLX_ACCELERATOR
        or model.device != MLX_DEVICE
    ):
        raise SidecarError(
            "RUNTIME_DEVICE_MISMATCH",
            "模型加载结果与 MLX gpu:0 不一致。",
            details={
                "runtime": model.runtime,
                "accelerator": model.accelerator,
                "device": model.device,
            },
        )
    if (
        model.model_format != "mlx"
        or model.model_revision != MLX_TTS_MODEL_REVISION
        or model.dtype != MODEL_PRECISION
        or model.model_precision != MODEL_PRECISION
        or model.speech_tokenizer_precision != SPEECH_TOKENIZER_PRECISION
    ):
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "模型格式、revision 或精度不符合 MLX 约束。",
            details={
                "modelFormat": model.model_format,
                "modelRevision": model.model_revision,
                "modelPrecision": model.model_precision,
                "speechTokenizerPrecision": model.speech_tokenizer_precision,
            },
        )


def _validate_mlx_model_integrity(model_path: Path, expected_revision: str) -> None:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(MODEL_VALIDATOR_PATH),
                "--model",
                "tts",
                "--tts-path",
                str(model_path),
                "--compact",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("validator output must be a JSON object")
        reports = payload.get("models")
        report = reports[0] if isinstance(reports, list) and len(reports) == 1 else None
        if (
            payload.get("ok") is not True
            or not isinstance(report, dict)
            or report.get("model") != "tts"
            or report.get("path") != str(model_path)
            or report.get("revision") != expected_revision
        ):
            raise ValueError("validator output does not match the requested TTS model")
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as error:
        details: dict[str, Any] = {"exceptionType": type(error).__name__}
        if isinstance(error, subprocess.CalledProcessError):
            details["returnCode"] = error.returncode
            if error.stderr:
                details["stderr"] = error.stderr.strip()[-2_000:]
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "MLX Qwen TTS 模型离线完整性校验失败。",
            details=details,
        ) from error


def _mlx_model_precisions(model: Any) -> tuple[str, str]:
    try:
        from mlx.utils import tree_flatten

        flattened = tree_flatten(model.parameters())
        model_dtypes = {
            _normalize_mlx_dtype(value.dtype)
            for name, value in flattened
            if "speech_tokenizer" not in name.split(".")
        }
        speech_tokenizer_dtypes = {
            _normalize_mlx_dtype(value.dtype)
            for name, value in flattened
            if "speech_tokenizer" in name.split(".")
        }
    except Exception as error:
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "无法确认 MLX TTS 模型实际精度。",
            details={"exceptionType": type(error).__name__},
        ) from error
    if model_dtypes != {MODEL_PRECISION} or speech_tokenizer_dtypes != {
        SPEECH_TOKENIZER_PRECISION
    }:
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "MLX TTS 主模型或 speech tokenizer 实际精度不符合约束。",
            details={
                "modelDtypes": sorted(model_dtypes),
                "speechTokenizerDtypes": sorted(speech_tokenizer_dtypes),
            },
        )
    return MODEL_PRECISION, SPEECH_TOKENIZER_PRECISION


def _normalize_mlx_dtype(value: Any) -> str:
    return str(value).removeprefix("mlx.core.")


def _generate_custom_voice(
    loaded_model: LoadedTTSModel,
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    results = list(loaded_model.model.generate_custom_voice(**kwargs))
    if len(results) != 1:
        raise SidecarError("SYNTHESIS_FAILED", "MLX CustomVoice 必须返回一个完整波形。")
    result = results[0]
    waveform = getattr(result, "audio", None)
    sample_rate = getattr(result, "sample_rate", None)
    if waveform is None:
        raise SidecarError("SYNTHESIS_FAILED", "MLX CustomVoice 未返回音频。")
    return _mlx_waveform_to_pcm16(waveform), sample_rate


def _mlx_waveform_to_pcm16(waveform: Any) -> Any:
    try:
        import mlx.core as mx
        import numpy as np

        mx.eval(waveform)
        array = np.asarray(waveform.astype(mx.float32))
    except Exception as error:
        raise SidecarError(
            "AUDIO_OUTPUT_INVALID",
            "无法将 MLX 音频转换为 float32。",
            details={"exceptionType": type(error).__name__},
        ) from error
    return _waveform_to_pcm16(array)


def _waveform_to_pcm16(waveform: Any) -> Any:
    try:
        import numpy as np

        array = np.asarray(waveform)
    except Exception as error:
        raise SidecarError(
            "AUDIO_OUTPUT_INVALID",
            "无法读取 TTS 音频波形。",
            details={"exceptionType": type(error).__name__},
        ) from error
    array = np.squeeze(array)
    if array.ndim != 1 or array.size == 0:
        raise SidecarError("AUDIO_OUTPUT_INVALID", "TTS 音频必须是非空单声道波形。")
    if array.dtype == np.int16:
        pcm16 = np.ascontiguousarray(array)
        if not bool(np.any(pcm16 != 0)):
            raise SidecarError("AUDIO_OUTPUT_INVALID", "TTS 音频为空白或静音。")
        return pcm16
    if not np.issubdtype(array.dtype, np.floating):
        raise SidecarError("AUDIO_OUTPUT_INVALID", "TTS 音频必须为浮点波形或 int16 PCM。")
    array = array.astype(np.float32, copy=False)
    if not bool(np.isfinite(array).all()):
        raise SidecarError("AUDIO_OUTPUT_INVALID", "TTS 音频包含非有限值。")
    clipped = np.clip(array, -1.0, 1.0)
    pcm16 = np.where(clipped < 0, clipped * 32_768.0, clipped * 32_767.0)
    pcm16 = np.rint(pcm16).astype(np.int16)
    if not bool(np.any(pcm16 != 0)):
        raise SidecarError("AUDIO_OUTPUT_INVALID", "TTS 音频为空白或静音。")
    return pcm16


def _required_string(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SidecarError("INVALID_REQUEST", f"{key} 必须是非空字符串。")
    return value


def _resolve_output_root(output_root: Path | None) -> Path:
    configured = output_root
    if configured is None:
        raw_value = os.environ.get("ELECTRON_QWEN_SPEECH_TTS_OUTPUT_ROOT")
        if not raw_value:
            raise SidecarError("INVALID_REQUEST", "缺少 ELECTRON_QWEN_SPEECH_TTS_OUTPUT_ROOT。")
        configured = Path(raw_value)
    configured.mkdir(parents=True, exist_ok=True)
    resolved = configured.resolve(strict=True)
    if not resolved.is_dir():
        raise SidecarError("INVALID_REQUEST", "TTS 输出根路径不是目录。")
    return resolved


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
