from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any, Literal, Protocol, TextIO

SpeechRuntime = Literal["mlx"]

MLX_ASR_MODEL_REVISION = "e1f6c266914abc5a46e8756e02580f834a6cf8a7"
MLX_FORCED_ALIGNER_MODEL_REVISION = "53c8c0e46733eec430e4b53dd6471d0e5dee45f8"
MLX_ASR_MODEL_PATH_ENV = "ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH"
MLX_ALIGNER_MODEL_PATH_ENV = "ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH"
REQUIRED_PYTHON_VERSION = (3, 12)
MLX_DEVICE = "gpu:0"
MODEL_DTYPE = "bfloat16"


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
    asr_model: Any
    aligner_model: Any
    asr_model_path: str
    aligner_model_path: str
    asr_device: str
    aligner_device: str
    asr_dtype: str
    aligner_dtype: str
    runtime: SpeechRuntime
    accelerator: str
    device: str
    model_format: str
    asr_model_revision: str | None
    aligner_model_revision: str | None


class ModelLoader(Protocol):
    def load(self) -> LoadedModels: ...


class ModelIntegrityValidator(Protocol):
    def __call__(self, model_path: Path, expected_revision: str) -> None: ...


class MLXModelLoader:
    def __init__(
        self,
        *,
        asr_model_path: Path | None = None,
        aligner_model_path: Path | None = None,
        integrity_validator: ModelIntegrityValidator | None = None,
        mlx_core: Any = None,
        load_function: Callable[[str], Any] | None = None,
    ) -> None:
        self._asr_model_path = asr_model_path
        self._aligner_model_path = aligner_model_path
        self._integrity_validator = integrity_validator
        self._mlx_core = mlx_core
        self._load_function = load_function

    def load(self) -> LoadedModels:
        _enable_offline_mode()
        asr_model_path = _resolve_local_model_path(
            self._asr_model_path,
            MLX_ASR_MODEL_PATH_ENV,
        )
        aligner_model_path = _resolve_local_model_path(
            self._aligner_model_path,
            MLX_ALIGNER_MODEL_PATH_ENV,
        )
        self._validate_integrity("asr", asr_model_path, MLX_ASR_MODEL_REVISION)
        self._validate_integrity(
            "aligner",
            aligner_model_path,
            MLX_FORCED_ALIGNER_MODEL_REVISION,
        )

        try:
            mx = self._mlx_core
            if mx is None:
                import mlx.core as mx
        except Exception as error:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "无法导入 MLX。",
                details={"exceptionType": type(error).__name__},
            ) from error

        _require_mlx_gpu(mx)

        try:
            load = self._load_function
            if load is None:
                from mlx_audio.stt import load

            asr_model = load(str(asr_model_path))
            aligner_model = load(str(aligner_model_path))
        except SidecarError:
            raise
        except Exception as error:
            raise SidecarError(
                "INITIALIZATION_FAILED",
                "MLX ASR 模型初始化失败。",
                details={"exceptionType": type(error).__name__},
            ) from error

        asr_dtype = _mlx_model_dtype(asr_model)
        aligner_dtype = _mlx_model_dtype(aligner_model)
        if asr_dtype != MODEL_DTYPE or aligner_dtype != MODEL_DTYPE:
            raise SidecarError(
                "RUNTIME_DEVICE_MISMATCH",
                "ASR 与 Forced Aligner 必须使用 bfloat16。",
                details={
                    "asrDtype": asr_dtype,
                    "alignerDtype": aligner_dtype,
                },
            )

        return LoadedModels(
            asr_model=asr_model,
            aligner_model=aligner_model,
            asr_model_path=str(asr_model_path),
            aligner_model_path=str(aligner_model_path),
            asr_device=MLX_DEVICE,
            aligner_device=MLX_DEVICE,
            asr_dtype=asr_dtype,
            aligner_dtype=aligner_dtype,
            runtime="mlx",
            accelerator="metal",
            device=MLX_DEVICE,
            model_format="mlx",
            asr_model_revision=MLX_ASR_MODEL_REVISION,
            aligner_model_revision=MLX_FORCED_ALIGNER_MODEL_REVISION,
        )

    def _validate_integrity(
        self,
        model_key: str,
        model_path: Path,
        expected_revision: str,
    ) -> None:
        try:
            if self._integrity_validator is None:
                _validate_mlx_model(model_key, model_path, expected_revision)
            else:
                self._integrity_validator(model_path, expected_revision)
        except SidecarError:
            raise
        except Exception as error:
            raise SidecarError(
                "MODEL_INTEGRITY_FAILED",
                "本地 MLX 模型完整性校验失败。",
                details={
                    "path": str(model_path),
                    "revision": expected_revision,
                    "exceptionType": type(error).__name__,
                },
            ) from error


def _enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _validate_mlx_model(
    model_key: str,
    model_path: Path,
    expected_revision: str,
) -> None:
    validator_path = Path(__file__).resolve().parents[3] / "scripts" / (
        "validate_mlx_speech_models.py"
    )
    path_argument = f"--{model_key}-path"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(validator_path),
                "--model",
                model_key,
                path_argument,
                str(model_path),
                "--compact",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "无法执行 MLX 模型完整性校验。",
            details={"exceptionType": type(error).__name__},
        ) from error
    if completed.returncode != 0:
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "MLX 模型完整性校验失败。",
            details={"model": model_key, "path": str(model_path)},
        )
    try:
        report = json.loads(completed.stdout)
        model_reports = report["models"]
        model_report = model_reports[0]
        actual_path = Path(model_report["path"])
        actual_revision = model_report["revision"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "MLX 模型完整性校验结果无效。",
            details={"model": model_key},
        ) from error
    if (
        report.get("ok") is not True
        or len(model_reports) != 1
        or actual_path != model_path
        or actual_revision != expected_revision
    ):
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "MLX 模型完整性校验结果与固定模型不一致。",
            details={"model": model_key, "path": str(model_path)},
        )


def _resolve_local_model_path(
    explicit_path: Path | None,
    environment_name: str,
) -> Path:
    configured_path = explicit_path
    if configured_path is None:
        configured_value = os.environ.get(environment_name, "").strip()
        if not configured_value:
            raise SidecarError(
                "MODEL_INTEGRITY_FAILED",
                f"{environment_name} 未配置。",
                details={"environmentVariable": environment_name},
            )
        configured_path = Path(configured_value)
    if not configured_path.is_absolute() or not configured_path.is_dir():
        raise SidecarError(
            "MODEL_INTEGRITY_FAILED",
            "MLX 模型路径必须是存在的绝对本地目录。",
            details={"path": str(configured_path)},
        )
    return configured_path.resolve()


def _require_mlx_gpu(mx: Any) -> None:
    try:
        gpu_device = mx.Device(mx.gpu, 0)
        if (
            not mx.metal.is_available()
            or mx.device_count(mx.gpu) < 1
            or not mx.is_available(gpu_device)
        ):
            raise SidecarError(
                "ACCELERATOR_UNAVAILABLE",
                "当前 MLX 环境不可使用 Metal GPU 0。",
            )
        mx.set_default_device(gpu_device)
        if mx.default_device() != gpu_device:
            raise SidecarError(
                "RUNTIME_DEVICE_MISMATCH",
                "MLX 默认设备必须为 gpu:0。",
            )
    except SidecarError:
        raise
    except Exception as error:
        raise SidecarError(
            "ACCELERATOR_UNAVAILABLE",
            "无法确认 MLX Metal GPU 0。",
            details={"exceptionType": type(error).__name__},
        ) from error


def _mlx_model_dtype(model: Any) -> str:
    try:
        parameters = model.parameters()
    except Exception as error:
        raise SidecarError(
            "RUNTIME_DEVICE_MISMATCH",
            "无法读取 MLX 模型参数精度。",
        ) from error

    dtypes = {
        _normalize_dtype_name(str(value.dtype))
        for value in _iter_tree_values(parameters)
        if hasattr(value, "dtype")
    }
    if not dtypes:
        raise SidecarError(
            "RUNTIME_DEVICE_MISMATCH",
            "无法确认 MLX 模型精度。",
        )
    if len(dtypes) != 1:
        return ",".join(sorted(dtypes))
    return next(iter(dtypes))


def _iter_tree_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_tree_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_tree_values(child)
    else:
        yield value


def _normalize_dtype_name(value: str) -> str:
    return value.removeprefix("mlx.core.").lower()


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
        self._model_loader = model_loader or MLXModelLoader()
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
            _validate_loaded_models(loaded_models)
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
            "runtime": existing_models.runtime,
            "accelerator": existing_models.accelerator,
            "device": existing_models.device,
            "modelFormat": existing_models.model_format,
            "asrModelPath": existing_models.asr_model_path,
            "alignerModelPath": existing_models.aligner_model_path,
            "asrModelRevision": existing_models.asr_model_revision,
            "alignerModelRevision": existing_models.aligner_model_revision,
            "asrModelPrecision": existing_models.asr_dtype,
            "alignerModelPrecision": existing_models.aligner_dtype,
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
            result = self._transcribe_one(
                models,
                audio_path,
                item.params.get("language"),
                item.task_id,
            )
            return {"taskId": item.task_id or "", **result}

        prepared_items = [
            (batch_item, _require_audio_file(batch_item["audioPath"]))
            for batch_item in batch_items
        ]
        results: list[dict[str, Any]] = []
        for batch_item, audio_path in prepared_items:
            result = self._transcribe_one(
                models,
                audio_path,
                batch_item.get("language"),
                item.task_id,
            )
            results.append({"itemId": batch_item["itemId"], **result})
        return {"taskId": item.task_id or "", "items": results}

    def _transcribe_one(
        self,
        models: LoadedModels,
        audio_path: Path,
        requested_language: str | None,
        task_id: str | None,
    ) -> dict[str, Any]:
        self._raise_if_cancelled(task_id)
        raw_asr_result = self._run_asr(
            models,
            audio_path,
            requested_language,
        )
        self._raise_if_cancelled(task_id)
        text, detected_language = _normalize_asr_result(raw_asr_result)
        alignment_language = requested_language or detected_language
        alignment_status, timestamps = self._run_alignment(
            models,
            audio_path,
            text,
            alignment_language,
        )
        self._raise_if_cancelled(task_id)

        result: dict[str, Any] = {
            "text": text,
            "timestamps": timestamps,
            "alignmentStatus": alignment_status,
        }
        if alignment_language is not None:
            result["language"] = alignment_language
        return result

    def _run_asr(
        self,
        models: LoadedModels,
        audio_path: Path,
        requested_language: str | None,
    ) -> Any:
        try:
            generation_kwargs: dict[str, Any] = {}
            if requested_language is not None:
                generation_kwargs["language"] = requested_language
            with redirect_stdout(self._stderr):
                return models.asr_model.generate(
                    str(audio_path),
                    **generation_kwargs,
                )
        except Exception as error:
            raise SidecarError(
                "TRANSCRIPTION_FAILED",
                "Qwen ASR 转写失败。",
                details={"exceptionType": type(error).__name__},
            ) from error

    def _run_alignment(
        self,
        models: LoadedModels,
        audio_path: Path,
        text: str,
        language: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            supported_languages = models.aligner_model.get_supported_languages()
        except Exception as error:
            raise SidecarError(
                "ALIGNMENT_FAILED",
                "无法读取 Forced Aligner 支持的语言。",
                details={"exceptionType": type(error).__name__},
            ) from error

        if not _language_is_supported(language, supported_languages):
            return "unsupported_language", []

        assert language is not None
        try:
            with redirect_stdout(self._stderr):
                raw_alignment = models.aligner_model.generate(
                    audio=str(audio_path),
                    text=text,
                    language=language,
                )
        except Exception as error:
            raise SidecarError(
                "ALIGNMENT_FAILED",
                "Qwen Forced Aligner 对齐失败。",
                details={"exceptionType": type(error).__name__},
            ) from error
        return "aligned", _normalize_alignment(raw_alignment)

    def _raise_if_cancelled(self, task_id: str | None) -> None:
        with self._state_lock:
            cancelled = self._cancel_requested
        if cancelled:
            raise SidecarError(
                "TASK_CANCELLED",
                "转写任务已取消，推理结果已丢弃。",
                details={"taskId": task_id},
            )

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


def _validate_loaded_models(models: LoadedModels) -> None:
    if (
        models.runtime != "mlx"
        or models.accelerator != "metal"
        or models.device != MLX_DEVICE
        or models.model_format != "mlx"
        or models.asr_device != MLX_DEVICE
        or models.aligner_device != MLX_DEVICE
        or models.asr_dtype != MODEL_DTYPE
        or models.aligner_dtype != MODEL_DTYPE
        or models.asr_model_revision != MLX_ASR_MODEL_REVISION
        or models.aligner_model_revision != MLX_FORCED_ALIGNER_MODEL_REVISION
    ):
        raise SidecarError(
            "RUNTIME_DEVICE_MISMATCH",
            "模型加载结果不满足 MLX gpu:0 与 bfloat16 约束。",
            details={
                "runtime": models.runtime,
                "accelerator": models.accelerator,
                "device": models.device,
                "asrDevice": models.asr_device,
                "alignerDevice": models.aligner_device,
                "asrPrecision": models.asr_dtype,
                "alignerPrecision": models.aligner_dtype,
            },
        )


def _normalize_asr_result(raw_result: Any) -> tuple[str, str | None]:
    if isinstance(raw_result, (list, tuple)):
        if len(raw_result) != 1:
            raise SidecarError(
                "TRANSCRIPTION_FAILED",
                "Qwen ASR 必须为单个音频返回一个结果。",
            )
        raw_result = raw_result[0]
    text = _read_attribute(raw_result, "text")
    if not isinstance(text, str):
        raise SidecarError("TRANSCRIPTION_FAILED", "Qwen ASR 结果缺少文本。")
    language = _first_valid_language(_read_attribute(raw_result, "language"))
    if language is None:
        raw_segments = _read_attribute(raw_result, "segments")
        if isinstance(raw_segments, (list, tuple)):
            language = _first_valid_language(
                [_read_attribute(segment, "language") for segment in raw_segments]
            )
    return text, language


def _first_valid_language(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.casefold() in {"", "none", "null", "unknown", "auto"}:
            return None
        return normalized
    if isinstance(value, (list, tuple)):
        for candidate in value:
            language = _first_valid_language(candidate)
            if language is not None:
                return language
    return None


def _language_is_supported(
    language: str | None,
    supported_languages: Any,
) -> bool:
    if language is None or not language.strip():
        return False
    if supported_languages is None:
        raise SidecarError(
            "ALIGNMENT_FAILED",
            "Forced Aligner 未返回支持语言列表。",
        )
    if not isinstance(supported_languages, (list, tuple, set, frozenset)):
        raise SidecarError(
            "ALIGNMENT_FAILED",
            "Forced Aligner 支持语言列表格式无效。",
        )
    normalized_supported_languages = {
        candidate.strip().casefold()
        for candidate in supported_languages
        if isinstance(candidate, str) and candidate.strip()
    }
    return language.strip().casefold() in normalized_supported_languages


def _normalize_alignment(raw_alignment: Any) -> list[dict[str, Any]]:
    raw_items = _read_attribute(raw_alignment, "items")
    if isinstance(raw_items, (list, tuple)):
        timestamp_items = raw_items
    elif isinstance(raw_alignment, (list, tuple)):
        timestamp_items = raw_alignment
    else:
        raise SidecarError("ALIGNMENT_FAILED", "Qwen Forced Aligner 结果缺少时间戳。")
    if not timestamp_items:
        raise SidecarError("ALIGNMENT_FAILED", "Qwen Forced Aligner 返回了空时间戳。")

    timestamps: list[dict[str, Any]] = []
    previous_start = -1.0
    previous_end = -1.0
    for raw_timestamp in timestamp_items:
        timestamp_text = _read_attribute(raw_timestamp, "text")
        start = _read_attribute(raw_timestamp, "start_time", "start")
        end = _read_attribute(raw_timestamp, "end_time", "end")
        if not isinstance(timestamp_text, str):
            raise SidecarError("ALIGNMENT_FAILED", "Qwen Forced Aligner 时间戳缺少文本。")
        if not isinstance(start, (int, float)) or not math.isfinite(start):
            raise SidecarError("ALIGNMENT_FAILED", "Qwen Forced Aligner 时间戳 start 非法。")
        if not isinstance(end, (int, float)) or not math.isfinite(end):
            raise SidecarError("ALIGNMENT_FAILED", "Qwen Forced Aligner 时间戳 end 非法。")
        start_seconds = float(start)
        end_seconds = float(end)
        if (
            start_seconds < 0
            or end_seconds < start_seconds
            or start_seconds < previous_start
            or end_seconds < previous_end
        ):
            raise SidecarError("ALIGNMENT_FAILED", "Qwen Forced Aligner 时间戳顺序非法。")
        previous_start = start_seconds
        previous_end = end_seconds
        timestamps.append(
            {
                "text": timestamp_text,
                "start": start_seconds,
                "end": end_seconds,
            }
        )

    return timestamps
