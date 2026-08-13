from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

TTS_MODEL_PATH_ENV = "ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH"
REQUIRED_PYTHON_VERSION = (3, 12)
MPS_DEVICE = "mps:0"
MODEL_DTYPE = "bfloat16"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


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


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    frame_count: int
    duration_seconds: float


class ModelLoader(Protocol):
    def load(self) -> LoadedTTSModel: ...


class AudioWriter(Protocol):
    def write(self, output_path: Path, waveform: Any, sample_rate: int) -> AudioMetadata: ...


class QwenTTSModelLoader:
    def __init__(self, *, model_path: Path | None = None) -> None:
        self._configured_model_path = model_path

    def load(self) -> LoadedTTSModel:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
            raise SidecarError(
                "MPS_UNAVAILABLE",
                "PYTORCH_ENABLE_MPS_FALLBACK 必须为 0，禁止 CPU fallback。",
            )

        model_path = _resolve_local_model_path(
            self._configured_model_path,
            TTS_MODEL_PATH_ENV,
            "Qwen TTS",
        )

        try:
            import torch
        except Exception as error:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "无法导入 PyTorch。",
                details={"exceptionType": type(error).__name__},
            ) from error

        if not torch.backends.mps.is_available():
            raise SidecarError(
                "MPS_UNAVAILABLE",
                "当前 PyTorch 或 macOS 环境不可使用 MPS。",
                details={"mpsBuilt": bool(torch.backends.mps.is_built())},
            )
        try:
            from qwen_tts import Qwen3TTSModel

            model = Qwen3TTSModel.from_pretrained(
                str(model_path),
                device_map=MPS_DEVICE,
                dtype=torch.bfloat16,
            )
        except SidecarError:
            raise
        except Exception as error:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "Qwen TTS 模型初始化失败。",
                details={"exceptionType": type(error).__name__},
            ) from error

        model_device = _model_device(getattr(model, "model", model))
        model_dtype = _model_dtype(getattr(model, "model", model))
        if model_device != MPS_DEVICE or model_dtype != MODEL_DTYPE:
            raise SidecarError(
                "MPS_DEVICE_MISMATCH",
                "TTS 模型必须位于 mps:0 并使用 bfloat16。",
                details={"device": model_device, "dtype": model_dtype},
            )

        supported_languages = model.get_supported_languages()
        supported_voice_ids = model.get_supported_speakers()
        if not supported_languages or not supported_voice_ids:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "CustomVoice 模型未返回语言或音色 capability。",
            )
        return LoadedTTSModel(
            model=model,
            model_path=str(model_path),
            device=model_device,
            dtype=model_dtype,
            supported_languages=tuple(str(value) for value in supported_languages),
            supported_voice_ids=tuple(str(value) for value in supported_voice_ids),
        )


def _resolve_local_model_path(
    configured_path: Path | None,
    environment_variable: str,
    model_label: str,
) -> Path:
    candidate = configured_path
    if candidate is None:
        configured_value = os.environ.get(environment_variable, "").strip()
        if not configured_value:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                f"{environment_variable} 未配置。",
                details={"environmentVariable": environment_variable},
            )
        candidate = Path(configured_value)

    expanded_path = candidate.expanduser()
    if not expanded_path.is_absolute():
        raise SidecarError(
            "INITIALIZATION_FAILED",
            f"{model_label} 模型路径必须是绝对路径。",
            details={"path": str(candidate)},
        )
    resolved_path = expanded_path.resolve(strict=False)
    if not resolved_path.is_dir():
        raise SidecarError(
            "INITIALIZATION_FAILED",
            f"本地 {model_label} 模型目录不存在。",
            details={"path": str(resolved_path)},
        )
    return resolved_path


class SoundFileAudioWriter:
    def write(self, output_path: Path, waveform: Any, sample_rate: int) -> AudioMetadata:
        try:
            import soundfile as sf

            with output_path.open("xb") as output_file:
                sf.write(output_file, waveform, sample_rate, format="WAV", subtype="PCM_16")
            info = sf.info(str(output_path))
        except FileExistsError as error:
            raise SidecarError(
                "AUDIO_WRITE_FAILED",
                "TTS 输出路径已存在。",
                details={"path": str(output_path)},
            ) from error
        except Exception as error:
            _safe_unlink(output_path)
            raise SidecarError(
                "AUDIO_WRITE_FAILED",
                "无法写入 TTS WAV。",
                details={"exceptionType": type(error).__name__},
            ) from error

        if (
            info.format != "WAV"
            or info.subtype != "PCM_16"
            or info.samplerate <= 0
            or info.channels <= 0
            or info.frames <= 0
            or not math.isfinite(info.duration)
            or info.duration <= 0
        ):
            _safe_unlink(output_path)
            raise SidecarError("AUDIO_OUTPUT_INVALID", "TTS 输出 WAV 元数据无效。")
        return AudioMetadata(
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            frame_count=int(info.frames),
            duration_seconds=float(info.duration),
        )


def _model_device(model: Any) -> str:
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration) as error:
            raise SidecarError("MPS_DEVICE_MISMATCH", "无法确认 TTS 模型设备。") from error
    normalized = str(device)
    return MPS_DEVICE if normalized == "mps" else normalized


def _model_dtype(model: Any) -> str:
    dtype = getattr(model, "dtype", None)
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration) as error:
            raise SidecarError("MPS_DEVICE_MISMATCH", "无法确认 TTS 模型精度。") from error
    return str(dtype).removeprefix("torch.")


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
        audio_writer: AudioWriter | None = None,
        output_root: Path | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        require_python_312: bool = True,
    ) -> None:
        self._model_loader = model_loader or QwenTTSModelLoader()
        self._audio_writer = audio_writer or SoundFileAudioWriter()
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
            if candidate.device != MPS_DEVICE or candidate.dtype != MODEL_DTYPE:
                raise SidecarError(
                    "MPS_DEVICE_MISMATCH",
                    "模型加载结果不满足 mps:0 与 bfloat16 约束。",
                )
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
            "modelPath": loaded_model.model_path,
            "device": loaded_model.device,
            "dtype": loaded_model.dtype,
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
        try:
            kwargs: dict[str, Any] = {
                "text": item.params["text"],
                "language": language,
                "speaker": voice_id,
            }
            if "instruction" in item.params:
                kwargs["instruct"] = item.params["instruction"]
            with redirect_stdout(self._stderr):
                waveforms, sample_rate = loaded_model.model.generate_custom_voice(**kwargs)
            generation_seconds = time.perf_counter() - started_at
            with self._state_lock:
                if self._cancel_requested:
                    raise SidecarError(
                        "TASK_CANCELLED",
                        "合成任务已取消，推理结果已丢弃。",
                        details={"taskId": item.task_id},
                    )
            if not isinstance(waveforms, (list, tuple)) or len(waveforms) != 1:
                raise SidecarError("SYNTHESIS_FAILED", "CustomVoice 必须返回一个波形。")
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise SidecarError("SYNTHESIS_FAILED", "CustomVoice 返回了无效采样率。")
            metadata = self._audio_writer.write(output_path, waveforms[0], sample_rate)
            with self._state_lock:
                if self._cancel_requested:
                    _safe_unlink(output_path)
                    raise SidecarError(
                        "TASK_CANCELLED",
                        "合成任务已取消，WAV 已删除。",
                        details={"taskId": item.task_id},
                    )
        except SidecarError:
            _safe_unlink(output_path)
            raise
        except Exception as error:
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
            raise SidecarError(
                "INVALID_REQUEST",
                "缺少 ELECTRON_QWEN_SPEECH_TTS_OUTPUT_ROOT。",
            )
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
