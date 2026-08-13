from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

ALIGNER_MODEL_PATH_ENV = "ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH"
ASR_MODEL_PATH_ENV = "ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH"
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
class LoadedModels:
    model: Any
    asr_model_path: str
    aligner_model_path: str
    asr_device: str
    aligner_device: str
    asr_dtype: str
    aligner_dtype: str


class ModelLoader(Protocol):
    def load(self) -> LoadedModels: ...


class QwenModelLoader:
    def __init__(
        self,
        *,
        asr_model_path: Path | None = None,
        aligner_model_path: Path | None = None,
    ) -> None:
        self._configured_asr_model_path = asr_model_path
        self._configured_aligner_model_path = aligner_model_path

    def load(self) -> LoadedModels:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
            raise SidecarError(
                "MPS_UNAVAILABLE",
                "PYTORCH_ENABLE_MPS_FALLBACK 必须为 0，禁止 CPU fallback。",
            )

        asr_model_path = _resolve_local_model_path(
            self._configured_asr_model_path,
            ASR_MODEL_PATH_ENV,
            "Qwen ASR",
        )
        aligner_model_path = _resolve_local_model_path(
            self._configured_aligner_model_path,
            ALIGNER_MODEL_PATH_ENV,
            "Qwen Forced Aligner",
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
            from qwen_asr import Qwen3ASRModel

            model = Qwen3ASRModel.from_pretrained(
                str(asr_model_path),
                dtype=torch.bfloat16,
                device_map=MPS_DEVICE,
                forced_aligner=str(aligner_model_path),
                forced_aligner_kwargs={
                    "dtype": torch.bfloat16,
                    "device_map": MPS_DEVICE,
                },
                max_inference_batch_size=-1,
                max_new_tokens=256,
            )
        except SidecarError:
            raise
        except Exception as error:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "Qwen ASR 模型初始化失败。",
                details={"exceptionType": type(error).__name__},
            ) from error

        aligner = getattr(model, "forced_aligner", None)
        if aligner is None:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "Qwen ASR 未加载 Forced Aligner。",
            )

        asr_device = _model_device(getattr(model, "model", model))
        aligner_device = _model_device(getattr(aligner, "model", aligner))
        if asr_device != MPS_DEVICE or aligner_device != MPS_DEVICE:
            raise SidecarError(
                "MPS_DEVICE_MISMATCH",
                "ASR 与 Forced Aligner 必须全部位于 mps:0。",
                details={
                    "asrDevice": asr_device,
                    "alignerDevice": aligner_device,
                },
            )

        asr_dtype = _model_dtype(getattr(model, "model", model))
        aligner_dtype = _model_dtype(getattr(aligner, "model", aligner))
        if asr_dtype != MODEL_DTYPE or aligner_dtype != MODEL_DTYPE:
            raise SidecarError(
                "MPS_DEVICE_MISMATCH",
                "ASR 与 Forced Aligner 必须使用 bfloat16。",
                details={
                    "asrDtype": asr_dtype,
                    "alignerDtype": aligner_dtype,
                },
            )

        return LoadedModels(
            model=model,
            asr_model_path=str(asr_model_path),
            aligner_model_path=str(aligner_model_path),
            asr_device=asr_device,
            aligner_device=aligner_device,
            asr_dtype=asr_dtype,
            aligner_dtype=aligner_dtype,
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


def _model_device(model: Any) -> str:
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration) as error:
            raise SidecarError(
                "MPS_DEVICE_MISMATCH",
                "无法确认模型设备。",
            ) from error
    normalized = str(device)
    return MPS_DEVICE if normalized == "mps" else normalized


def _model_dtype(model: Any) -> str:
    dtype = getattr(model, "dtype", None)
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration) as error:
            raise SidecarError(
                "MPS_DEVICE_MISMATCH",
                "无法确认模型精度。",
            ) from error
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
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        require_python_312: bool = True,
    ) -> None:
        self._model_loader = model_loader or QwenModelLoader()
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._require_python_312 = require_python_312
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._queue: queue.Queue[WorkItem | object] = queue.Queue()
        self._queued_items: dict[str, WorkItem] = {}
        self._pending_items: list[WorkItem] = []
        self._models: LoadedModels | None = None
        self._load_count = 0
        self._state = "uninitialized"
        self._current_task_id: str | None = None
        self._cancel_requested = False
        self._stopping = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="qwen-asr-worker",
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
                candidate_request_id = request.get("id")
                if isinstance(candidate_request_id, str) and candidate_request_id.strip():
                    request_id = candidate_request_id
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

    def _validate_request(
        self,
        request: Any,
    ) -> tuple[str, str, dict[str, Any]]:
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
        if not isinstance(request_id, str) or not request_id.strip():
            raise SidecarError("INVALID_REQUEST", "请求 id 必须是非空字符串。")
        method = request.get("method")
        if not isinstance(method, str) or not method.strip():
            raise SidecarError("INVALID_REQUEST", "请求 method 必须是非空字符串。")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise SidecarError("INVALID_REQUEST", "请求 params 必须是 JSON 对象。")
        return request_id, method, params

    def _check_python_version(self) -> None:
        if self._require_python_312 and sys.version_info[:2] != REQUIRED_PYTHON_VERSION:
            raise SidecarError(
                "PYTHON_VERSION_UNSUPPORTED",
                "Qwen ASR Sidecar 必须使用 Python 3.12。",
                details={
                    "required": "3.12",
                    "actual": f"{sys.version_info.major}.{sys.version_info.minor}",
                },
            )

    def _dispatch(
        self,
        request_id: str,
        method: str,
        params: dict[str, Any],
    ) -> bool:
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
                if self._models is None:
                    self._state = "initializing"
            self._enqueue(WorkItem(request_id=request_id, method=method, params=params))
            return False
        if method == "transcribe":
            normalized_params = _normalize_transcribe_params(params)
            task_id = normalized_params["taskId"]
            item = WorkItem(
                request_id=request_id,
                method=method,
                params=normalized_params,
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
                    "排队中的转写任务已取消。",
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
            self._emit_error(
                item.request_id,
                SidecarError(code, "Sidecar 退出时取消了排队请求。"),
            )
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
                if item.method == "initialize":
                    result = self._initialize()
                else:
                    result = self._transcribe(item)
            except SidecarError as error:
                if error.__cause__ is not None:
                    traceback.print_exception(error, file=self._stderr)
                failure = error
            except Exception as error:
                traceback.print_exc(file=self._stderr)
                code = (
                    "INITIALIZATION_FAILED"
                    if item.method == "initialize"
                    else "TRANSCRIPTION_FAILED"
                )
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
                                "转写任务已取消，推理结果已丢弃。",
                                details={"taskId": item.task_id},
                            )
                        self._current_task_id = None
                        self._cancel_requested = False
                    if self._stopping:
                        self._state = "shutting_down"
                    elif self._models is not None:
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
            existing_models = self._models
        if existing_models is None:
            with redirect_stdout(self._stderr):
                loaded_models = self._model_loader.load()
            if (
                loaded_models.asr_device != MPS_DEVICE
                or loaded_models.aligner_device != MPS_DEVICE
                or loaded_models.asr_dtype != MODEL_DTYPE
                or loaded_models.aligner_dtype != MODEL_DTYPE
            ):
                raise SidecarError(
                    "MPS_DEVICE_MISMATCH",
                    "模型加载结果不满足 mps:0 与 bfloat16 约束。",
                )
            with self._state_lock:
                if self._models is None:
                    self._models = loaded_models
                    self._load_count += 1
                existing_models = self._models

        assert existing_models is not None
        with self._state_lock:
            self._state = "ready"
            load_count = self._load_count
        return {
            "state": "ready",
            "asrModelPath": existing_models.asr_model_path,
            "alignerModelPath": existing_models.aligner_model_path,
            "asrDevice": existing_models.asr_device,
            "alignerDevice": existing_models.aligner_device,
            "asrDtype": existing_models.asr_dtype,
            "alignerDtype": existing_models.aligner_dtype,
            "loadCount": load_count,
        }

    def _transcribe(self, item: WorkItem) -> dict[str, Any]:
        with self._state_lock:
            models = self._models
            cancelled_before_start = self._cancel_requested
        if models is None:
            raise SidecarError("NOT_INITIALIZED", "ASR Provider 尚未初始化。")
        if cancelled_before_start:
            raise SidecarError(
                "TASK_CANCELLED",
                "转写任务已取消。",
                details={"taskId": item.task_id},
            )

        batch_items = item.params.get("items")
        if batch_items is None:
            audio_path = _require_audio_file(item.params["audioPath"])
            transcription_kwargs: dict[str, Any] = {
                "audio": str(audio_path),
                "return_time_stamps": True,
            }
            if "language" in item.params:
                transcription_kwargs["language"] = item.params["language"]
        else:
            audio_paths = [_require_audio_file(batch_item["audioPath"]) for batch_item in batch_items]
            transcription_kwargs = {
                "audio": [str(audio_path) for audio_path in audio_paths],
                "language": [batch_item.get("language") for batch_item in batch_items],
                "return_time_stamps": True,
            }

        try:
            with redirect_stdout(self._stderr):
                raw_results = models.model.transcribe(**transcription_kwargs)
        except SidecarError:
            raise
        except Exception as error:
            raise SidecarError(
                "TRANSCRIPTION_FAILED",
                "Qwen ASR 转写失败。",
                details={"exceptionType": type(error).__name__},
            ) from error

        with self._state_lock:
            cancelled_after_inference = self._cancel_requested
        if cancelled_after_inference:
            raise SidecarError(
                "TASK_CANCELLED",
                "转写任务已取消，推理结果已丢弃。",
                details={"taskId": item.task_id},
            )

        if batch_items is None:
            return _normalize_transcription(item.task_id or "", raw_results)
        return _normalize_batch_transcription(item.task_id or "", batch_items, raw_results)

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "initialized": self._models is not None,
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


def _normalize_transcribe_params(params: dict[str, Any]) -> dict[str, Any]:
    task_id = _required_string(params, "taskId")
    has_audio_path = "audioPath" in params
    has_items = "items" in params
    if has_audio_path == has_items:
        raise SidecarError(
            "INVALID_REQUEST",
            "transcribe 必须且只能提供 audioPath 或 items。",
        )

    if has_audio_path:
        unknown_keys = set(params) - {"taskId", "audioPath", "language"}
        if unknown_keys:
            raise SidecarError(
                "INVALID_REQUEST",
                "transcribe 包含未知参数。",
                details={"keys": sorted(unknown_keys)},
            )
        normalized: dict[str, Any] = {
            "taskId": task_id,
            "audioPath": _required_string(params, "audioPath"),
        }
        language = params.get("language")
        if language is not None:
            if not isinstance(language, str) or not language.strip():
                raise SidecarError("INVALID_REQUEST", "language 必须是非空字符串。")
            normalized["language"] = language
        return normalized

    if set(params) != {"taskId", "items"}:
        raise SidecarError(
            "INVALID_REQUEST",
            "批量 transcribe 只接受 taskId 与 items。",
            details={"keys": sorted(set(params) - {"taskId", "items"})},
        )
    raw_items = params["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise SidecarError("INVALID_REQUEST", "items 必须是非空数组。")

    item_ids: set[str] = set()
    normalized_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise SidecarError(
                "INVALID_REQUEST",
                "items 中的项目必须是对象。",
                details={"index": index},
            )
        unknown_item_keys = set(raw_item) - {"itemId", "audioPath", "language"}
        if unknown_item_keys:
            raise SidecarError(
                "INVALID_REQUEST",
                "批量转写项目包含未知参数。",
                details={"index": index, "keys": sorted(unknown_item_keys)},
            )
        item_id = _required_string(raw_item, "itemId")
        if item_id in item_ids:
            raise SidecarError(
                "INVALID_REQUEST",
                "itemId 不能重复。",
                details={"itemId": item_id},
            )
        item_ids.add(item_id)
        normalized_item: dict[str, Any] = {
            "itemId": item_id,
            "audioPath": _required_string(raw_item, "audioPath"),
        }
        language = raw_item.get("language")
        if language is not None:
            if not isinstance(language, str) or not language.strip():
                raise SidecarError("INVALID_REQUEST", "language 必须是非空字符串。")
            normalized_item["language"] = language
        normalized_items.append(normalized_item)
    return {"taskId": task_id, "items": normalized_items}


def _require_audio_file(value: str) -> Path:
    audio_path = Path(value)
    if not audio_path.is_file():
        raise SidecarError(
            "AUDIO_NOT_FOUND",
            "音频文件不存在。",
            details={"path": str(audio_path)},
        )
    return audio_path


def _read_attribute(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _normalize_transcription(task_id: str, raw_results: Any) -> dict[str, Any]:
    if not isinstance(raw_results, (list, tuple)) or len(raw_results) != 1:
        raise SidecarError(
            "TRANSCRIPTION_FAILED",
            "Qwen ASR 必须为单个音频返回一个结果。",
        )
    result = _normalize_transcription_item(raw_results[0])
    return {"taskId": task_id, **result}


def _normalize_batch_transcription(
    task_id: str,
    batch_items: list[dict[str, Any]],
    raw_results: Any,
) -> dict[str, Any]:
    if not isinstance(raw_results, (list, tuple)) or len(raw_results) != len(batch_items):
        raise SidecarError(
            "TRANSCRIPTION_FAILED",
            "Qwen ASR 批量结果数量与输入数量不一致。",
        )
    return {
        "taskId": task_id,
        "items": [
            {
                "itemId": batch_item["itemId"],
                **_normalize_transcription_item(raw_result),
            }
            for batch_item, raw_result in zip(batch_items, raw_results, strict=True)
        ],
    }


def _normalize_transcription_item(raw_result: Any) -> dict[str, Any]:
    text = _read_attribute(raw_result, "text")
    if not isinstance(text, str):
        raise SidecarError("TRANSCRIPTION_FAILED", "Qwen ASR 结果缺少文本。")

    raw_timestamps = _read_attribute(raw_result, "time_stamps", "timestamps")
    raw_timestamp_items = _read_attribute(raw_timestamps, "items")
    if isinstance(raw_timestamps, (list, tuple)):
        timestamp_items = raw_timestamps
    elif isinstance(raw_timestamp_items, (list, tuple)):
        timestamp_items = raw_timestamp_items
    else:
        raise SidecarError("TRANSCRIPTION_FAILED", "Qwen ASR 结果缺少时间戳。")

    timestamps: list[dict[str, Any]] = []
    previous_start = -1.0
    previous_end = -1.0
    for raw_timestamp in timestamp_items:
        timestamp_text = _read_attribute(raw_timestamp, "text")
        start = _read_attribute(raw_timestamp, "start_time", "start")
        end = _read_attribute(raw_timestamp, "end_time", "end")
        if not isinstance(timestamp_text, str):
            raise SidecarError("TRANSCRIPTION_FAILED", "Qwen ASR 时间戳缺少文本。")
        if not isinstance(start, (int, float)) or not math.isfinite(start):
            raise SidecarError("TRANSCRIPTION_FAILED", "Qwen ASR 时间戳 start 非法。")
        if not isinstance(end, (int, float)) or not math.isfinite(end):
            raise SidecarError("TRANSCRIPTION_FAILED", "Qwen ASR 时间戳 end 非法。")
        start_seconds = float(start)
        end_seconds = float(end)
        if (
            start_seconds < 0
            or end_seconds < start_seconds
            or start_seconds < previous_start
            or end_seconds < previous_end
        ):
            raise SidecarError("TRANSCRIPTION_FAILED", "Qwen ASR 时间戳顺序非法。")
        previous_start = start_seconds
        previous_end = end_seconds
        timestamps.append(
            {
                "text": timestamp_text,
                "start": start_seconds,
                "end": end_seconds,
            }
        )

    result: dict[str, Any] = {"text": text, "timestamps": timestamps}
    language = _read_attribute(raw_result, "language")
    if isinstance(language, str):
        result["language"] = language
    return result
