#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import tempfile
import time
import traceback
import wave
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASR_FIXTURE_PATH = (
    REPOSITORY_ROOT
    / 'services'
    / 'asr-sidecar'
    / 'tests'
    / 'fixtures'
    / 'Complex-1-Live-stream-Sales.wav'
)
TTS_INPUT = {
    'text': '银行行长走过人行道。',
    'language': 'Chinese',
    'speaker': 'Vivian',
    'instruct': '自然、清晰地朗读',
}
REQUIRED_PYTHON_VERSION = (3, 12)
RUNTIME = 'mlx'
Capability = Literal['tts', 'asr']
MemoryMeasurement = Literal['high_water_mark', 'sampled']


class BenchmarkArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def target_python_path(
    capability: Capability,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    relative_paths = {
        'tts': Path('services/tts-sidecar/.venv-mlx/bin/python'),
        'asr': Path('services/asr-sidecar/.venv-mlx/bin/python'),
    }
    return repository_root / relative_paths[capability]


def calculate_tts_rates(
    generation_seconds: float,
    audio_seconds: float,
) -> dict[str, float]:
    if (
        not math.isfinite(generation_seconds)
        or generation_seconds <= 0
        or not math.isfinite(audio_seconds)
        or audio_seconds <= 0
    ):
        raise ValueError('TTS 计时与音频时长必须是有限正数。')
    return {
        'rtf': generation_seconds / audio_seconds,
        'realtimeMultiple': audio_seconds / generation_seconds,
    }


def calculate_character_throughput(
    text: str,
    inference_seconds: float,
) -> dict[str, float | int]:
    if not text:
        raise ValueError('吞吐计算的文本不能为空。')
    if not math.isfinite(inference_seconds) or inference_seconds <= 0:
        raise ValueError('推理时间必须是有限正数。')
    character_count = len(text)
    return {
        'characterCount': character_count,
        'charactersPerSecond': character_count / inference_seconds,
    }


def memory_metric(
    value: int,
    measurement: MemoryMeasurement,
) -> dict[str, str | int]:
    if value < 0:
        raise ValueError('内存计数不能为负数。')
    return {
        'value': value,
        'unit': 'bytes',
        'measurement': measurement,
    }


def summarize_memory_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError('内存采样不能为空。')

    os_max_rss = max(sample['os']['maxRssBytes']['value'] for sample in samples)
    backend_metric_names = set(samples[0]['backend']['metrics'])
    for sample in samples[1:]:
        if set(sample['backend']['metrics']) != backend_metric_names:
            raise ValueError('各阶段后端内存 metric 不一致。')

    backend_summary: dict[str, Any] = {}
    for name in sorted(backend_metric_names):
        metrics = [sample['backend']['metrics'][name] for sample in samples]
        measurements = {metric['measurement'] for metric in metrics}
        if len(measurements) != 1:
            raise ValueError(f'后端内存 metric 语义不一致：{name}')
        measurement = measurements.pop()
        summary_measurement = (
            'high_water_mark' if measurement == 'high_water_mark' else 'max_of_samples'
        )
        backend_summary[name] = {
            'value': max(metric['value'] for metric in metrics),
            'unit': 'bytes',
            'measurement': summary_measurement,
        }

    return {
        'os': {
            'maxRssBytes': memory_metric(os_max_rss, 'high_water_mark'),
            'source': 'resource.getrusage(RUSAGE_SELF).ru_maxrss',
        },
        'backend': {
            'source': samples[0]['backend']['source'],
            'metrics': backend_summary,
        },
    }


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = BenchmarkArgumentParser(
        description='Benchmark ElectronQwenSpeech 本地 MLX 语音运行时。',
    )
    parser.add_argument('--capability', choices=('tts', 'asr'), required=True)
    return parser.parse_args(argv)


def _run_in_target_venv(arguments: argparse.Namespace) -> None:
    python_path = target_python_path(arguments.capability)
    expected_prefix = python_path.parent.parent.resolve(strict=False)
    if Path(sys.prefix).resolve(strict=False) == expected_prefix:
        return
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError(f'目标 Python 不可执行：{python_path}')
    os.execve(
        str(python_path),
        [
            str(python_path),
            str(Path(__file__).resolve()),
            '--capability',
            arguments.capability,
        ],
        dict(os.environ),
    )


def _configure_offline_mode() -> None:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def _require_python_312() -> None:
    if sys.version_info[:2] != REQUIRED_PYTHON_VERSION:
        actual = f'{sys.version_info.major}.{sys.version_info.minor}'
        raise RuntimeError(f'benchmark 必须使用 Python 3.12，当前为 {actual}。')


def _add_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _validate_mlx_models(
    requests: list[tuple[str, Path]],
) -> dict[str, Any]:
    _add_import_path(REPOSITORY_ROOT)
    from scripts.validate_mlx_speech_models import validate_model

    model_reports: list[dict[str, Any]] = []
    total_started_at = time.perf_counter()
    for model_key, model_path in requests:
        started_at = time.perf_counter()
        report = validate_model(model_key, model_path)
        elapsed_seconds = time.perf_counter() - started_at
        model_reports.append({
            'model': report['model'],
            'path': report['path'],
            'revision': report['revision'],
            'validatedFileCount': len(report['files']),
            'validatedSafetensorsCount': len(report['safetensors']),
            'seconds': elapsed_seconds,
        })
    return {
        'status': 'passed',
        'seconds': time.perf_counter() - total_started_at,
        'models': model_reports,
    }


def _synchronize_backend() -> None:
    import mlx.core as mx

    mx.synchronize()


def _sample_memory(stage: str) -> dict[str, Any]:
    import mlx.core as mx

    return {
        'stage': stage,
        'os': {
            'maxRssBytes': memory_metric(
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                'high_water_mark',
            ),
            'source': 'resource.getrusage(RUSAGE_SELF).ru_maxrss',
        },
        'backend': {
            'source': 'mlx.core',
            'metrics': {
                'activeMemoryBytes': memory_metric(
                    int(mx.get_active_memory()),
                    'sampled',
                ),
                'cacheMemoryBytes': memory_metric(
                    int(mx.get_cache_memory()),
                    'sampled',
                ),
                'peakMemoryBytes': memory_metric(
                    int(mx.get_peak_memory()),
                    'high_water_mark',
                ),
            },
        },
    }


def _audio_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), 'rb') as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if frame_count <= 0 or sample_rate <= 0:
        raise RuntimeError(f'音频 fixture 时长无效：{audio_path}')
    return frame_count / sample_rate


def _runtime_metadata(capability: Capability) -> dict[str, Any]:
    return {
        'runtime': RUNTIME,
        'capability': capability,
        'pythonExecutable': sys.executable,
        'pythonVersion': platform.python_version(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'pid': os.getpid(),
        'offlineEnvironment': {
            key: os.environ[key]
            for key in (
                'HF_HUB_OFFLINE',
                'TRANSFORMERS_OFFLINE',
                'HF_DATASETS_OFFLINE',
                'HF_HUB_DISABLE_TELEMETRY',
            )
        },
    }


def _run_tts_benchmark() -> dict[str, Any]:
    _add_import_path(REPOSITORY_ROOT / 'services' / 'tts-sidecar')
    from tts_sidecar import service

    configured_model_path = os.environ.get(service.MLX_MODEL_PATH_ENV, '').strip()
    if not configured_model_path:
        raise RuntimeError(f'{service.MLX_MODEL_PATH_ENV} 未配置。')
    model_path = Path(configured_model_path).resolve(strict=False)
    integrity_validation = _validate_mlx_models([('tts', model_path)])
    loader = service.MLXQwenTTSModelLoader(
        model_path=model_path,
        integrity_validator=lambda _path, _revision: None,
    )

    load_started_at = time.perf_counter()
    loaded_model = loader.load()
    service._validate_loaded_model(loaded_model)
    _synchronize_backend()
    load_seconds = time.perf_counter() - load_started_at
    memory_samples = [_sample_memory('after_model_load')]

    iterations: list[dict[str, Any]] = []
    writer = service.PCM16WaveAudioWriter()
    with tempfile.TemporaryDirectory(
        prefix='electron-qwen-speech-tts-benchmark-',
    ) as directory:
        output_root = Path(directory)
        for phase in ('cold', 'hot'):
            _synchronize_backend()
            started_at = time.perf_counter()
            waveform, sample_rate = service._generate_custom_voice(
                loaded_model,
                dict(TTS_INPUT),
            )
            _synchronize_backend()
            generation_seconds = time.perf_counter() - started_at
            metadata = writer.write(
                output_root / f'{phase}.wav',
                waveform,
                sample_rate,
            )
            rates = calculate_tts_rates(
                generation_seconds,
                metadata.duration_seconds,
            )
            iterations.append({
                'phase': phase,
                'generationSeconds': generation_seconds,
                'audioSeconds': metadata.duration_seconds,
                **rates,
                'sampleRate': metadata.sample_rate,
                'channels': metadata.channels,
                'frameCount': metadata.frame_count,
            })
            memory_samples.append(_sample_memory(f'after_{phase}_inference'))

    return {
        'schemaVersion': 1,
        'ok': True,
        **_runtime_metadata('tts'),
        'input': dict(TTS_INPUT),
        'integrityValidation': integrity_validation,
        'modelLoad': {
            'seconds': load_seconds,
            'modelPath': loaded_model.model_path,
            'modelRevision': loaded_model.model_revision,
            'modelFormat': loaded_model.model_format,
            'modelPrecision': loaded_model.model_precision,
            'speechTokenizerPrecision': loaded_model.speech_tokenizer_precision,
            'accelerator': loaded_model.accelerator,
            'device': loaded_model.device,
        },
        'metricDefinitions': {
            'rtf': 'generationSeconds / audioSeconds',
            'realtimeMultiple': 'audioSeconds / generationSeconds',
        },
        'iterations': iterations,
        'memorySamples': memory_samples,
        'memorySummary': summarize_memory_samples(memory_samples),
    }


def _run_asr_benchmark() -> dict[str, Any]:
    _add_import_path(REPOSITORY_ROOT / 'services' / 'asr-sidecar')
    from asr_sidecar import service

    if not ASR_FIXTURE_PATH.is_file():
        raise FileNotFoundError(f'ASR fixture 不存在：{ASR_FIXTURE_PATH}')
    audio_seconds = _audio_duration_seconds(ASR_FIXTURE_PATH)

    configured_asr_model_path = os.environ.get(service.MLX_ASR_MODEL_PATH_ENV, '').strip()
    configured_aligner_model_path = os.environ.get(
        service.MLX_ALIGNER_MODEL_PATH_ENV,
        '',
    ).strip()
    if not configured_asr_model_path:
        raise RuntimeError(f'{service.MLX_ASR_MODEL_PATH_ENV} 未配置。')
    if not configured_aligner_model_path:
        raise RuntimeError(f'{service.MLX_ALIGNER_MODEL_PATH_ENV} 未配置。')
    asr_model_path = Path(configured_asr_model_path).resolve(strict=False)
    aligner_model_path = Path(configured_aligner_model_path).resolve(strict=False)
    integrity_validation = _validate_mlx_models([
        ('asr', asr_model_path),
        ('aligner', aligner_model_path),
    ])
    loader = service.MLXModelLoader(
        asr_model_path=asr_model_path,
        aligner_model_path=aligner_model_path,
        integrity_validator=lambda _path, _revision: None,
    )

    load_started_at = time.perf_counter()
    loaded_models = loader.load()
    service._validate_loaded_models(loaded_models)
    _synchronize_backend()
    load_seconds = time.perf_counter() - load_started_at
    memory_samples = [_sample_memory('after_model_load')]

    runner = SimpleNamespace(_stderr=sys.stderr)
    iterations: list[dict[str, Any]] = []
    for phase in ('cold', 'hot'):
        _synchronize_backend()
        asr_started_at = time.perf_counter()
        raw_asr_result = service.SidecarService._run_asr(
            runner,
            loaded_models,
            ASR_FIXTURE_PATH,
            None,
        )
        _synchronize_backend()
        asr_seconds = time.perf_counter() - asr_started_at
        text, detected_language = service._normalize_asr_result(raw_asr_result)
        asr_throughput = calculate_character_throughput(text, asr_seconds)
        memory_samples.append(_sample_memory(f'after_{phase}_asr'))

        _synchronize_backend()
        aligner_started_at = time.perf_counter()
        alignment_status, timestamps = service.SidecarService._run_alignment(
            runner,
            loaded_models,
            ASR_FIXTURE_PATH,
            text,
            detected_language,
        )
        _synchronize_backend()
        aligner_seconds = time.perf_counter() - aligner_started_at
        if alignment_status != 'aligned' or not timestamps:
            raise RuntimeError('benchmark fixture 必须完成真实 Forced Aligner 推理。')
        aligner_throughput = calculate_character_throughput(text, aligner_seconds)
        memory_samples.append(_sample_memory(f'after_{phase}_aligner'))

        iterations.append({
            'phase': phase,
            'text': text,
            'detectedLanguage': detected_language,
            'audioSeconds': audio_seconds,
            'asrSeconds': asr_seconds,
            'asrCharacterCount': asr_throughput['characterCount'],
            'asrCharactersPerSecond': asr_throughput['charactersPerSecond'],
            'alignerSeconds': aligner_seconds,
            'inferenceSeconds': asr_seconds + aligner_seconds,
            'alignerCharacterCount': aligner_throughput['characterCount'],
            'alignerCharactersPerSecond': aligner_throughput['charactersPerSecond'],
            'alignmentStatus': alignment_status,
            'timestampCount': len(timestamps),
        })

    return {
        'schemaVersion': 1,
        'ok': True,
        **_runtime_metadata('asr'),
        'input': {
            'audioPath': str(ASR_FIXTURE_PATH),
            'audioSeconds': audio_seconds,
        },
        'integrityValidation': integrity_validation,
        'modelLoad': {
            'seconds': load_seconds,
            'asrModelPath': loaded_models.asr_model_path,
            'alignerModelPath': loaded_models.aligner_model_path,
            'asrModelRevision': loaded_models.asr_model_revision,
            'alignerModelRevision': loaded_models.aligner_model_revision,
            'modelFormat': loaded_models.model_format,
            'asrModelPrecision': loaded_models.asr_dtype,
            'alignerModelPrecision': loaded_models.aligner_dtype,
            'accelerator': loaded_models.accelerator,
            'device': loaded_models.device,
            'asrDevice': loaded_models.asr_device,
            'alignerDevice': loaded_models.aligner_device,
        },
        'metricDefinitions': {
            'characterCount': 'len(text)',
            'asrCharactersPerSecond': 'asrCharacterCount / asrSeconds',
            'alignerCharactersPerSecond': 'alignerCharacterCount / alignerSeconds',
            'inferenceSeconds': 'asrSeconds + alignerSeconds',
        },
        'iterations': iterations,
        'memorySamples': memory_samples,
        'memorySummary': summarize_memory_samples(memory_samples),
    }


def run_benchmark(capability: Capability) -> dict[str, Any]:
    if capability == 'tts':
        return _run_tts_benchmark()
    return _run_asr_benchmark()


def _error_payload(
    error: BaseException,
    arguments: argparse.Namespace | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'schemaVersion': 1,
        'ok': False,
        'runtime': RUNTIME,
        'error': {
            'type': type(error).__name__,
            'message': str(error),
        },
    }
    if arguments is not None:
        payload['capability'] = arguments.capability
    return payload


def main(argv: list[str] | None = None) -> int:
    arguments: argparse.Namespace | None = None
    try:
        arguments = _parse_arguments(argv)
        _run_in_target_venv(arguments)
        _require_python_312()
        _configure_offline_mode()
        with redirect_stdout(sys.stderr):
            report = run_benchmark(arguments.capability)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(
            _error_payload(error, arguments),
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=True,
        ))
        return 1

    print(json.dumps(
        report,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
