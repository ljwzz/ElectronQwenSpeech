#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).with_name('mlx_speech_model_manifest.json')
MODEL_KEYS = ('tts', 'asr', 'aligner')
MODEL_PATH_ENV = {
    'tts': 'ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH',
    'asr': 'ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH',
    'aligner': 'ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH',
}
CHUNK_BYTES = 8 * 1024 * 1024


class ModelIntegrityError(RuntimeError):
    pass


def _load_manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if value.get('formatVersion') != 1 or not isinstance(value.get('models'), dict):
        raise ModelIntegrityError(f'校验 manifest 格式无效：{MANIFEST_PATH}')
    return value


def _hash_file(path: Path, *, include_git_blob: bool) -> tuple[str, str | None]:
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1() if include_git_blob else None
    if git_blob is not None:
        git_blob.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        while chunk := stream.read(CHUNK_BYTES):
            sha256.update(chunk)
            if git_blob is not None:
                git_blob.update(chunk)
    return sha256.hexdigest(), None if git_blob is None else git_blob.hexdigest()


def _read_safetensors(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open('rb') as stream:
        raw_header_size = stream.read(8)
        if len(raw_header_size) != 8:
            raise ModelIntegrityError(f'Safetensors header 长度不足：{path}')
        header_size = struct.unpack('<Q', raw_header_size)[0]
        if header_size <= 0 or header_size > file_size - 8:
            raise ModelIntegrityError(f'Safetensors header 边界无效：{path}')
        try:
            header = json.loads(stream.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelIntegrityError(f'Safetensors header JSON 无效：{path}') from error

    if not isinstance(header, dict):
        raise ModelIntegrityError(f'Safetensors header 必须是对象：{path}')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict):
        raise ModelIntegrityError(f'Safetensors metadata 必须是对象：{path}')

    offsets: list[tuple[int, int]] = []
    dtypes: set[str] = set()
    for name, tensor in header.items():
        if not isinstance(name, str) or not isinstance(tensor, dict):
            raise ModelIntegrityError(f'Safetensors tensor 条目无效：{path}')
        dtype = tensor.get('dtype')
        shape = tensor.get('shape')
        data_offsets = tensor.get('data_offsets')
        if not isinstance(dtype, str):
            raise ModelIntegrityError(f'Safetensors dtype 无效：{path}:{name}')
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or value < 0 for value in shape
        ):
            raise ModelIntegrityError(f'Safetensors shape 无效：{path}:{name}')
        if (
            not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or any(not isinstance(value, int) or value < 0 for value in data_offsets)
            or data_offsets[1] < data_offsets[0]
        ):
            raise ModelIntegrityError(f'Safetensors data_offsets 无效：{path}:{name}')
        dtypes.add(dtype)
        offsets.append((data_offsets[0], data_offsets[1]))

    offsets.sort()
    expected_offset = 0
    for start, end in offsets:
        if start != expected_offset:
            raise ModelIntegrityError(f'Safetensors tensor 数据不连续：{path}')
        expected_offset = end
    if 8 + header_size + expected_offset != file_size:
        raise ModelIntegrityError(f'Safetensors tensor 数据区大小不匹配：{path}')
    return {
        'format': metadata.get('format'),
        'dtypes': sorted(dtypes),
        'tensorCount': len(offsets),
    }


def validate_model(
    model_key: str,
    model_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest()
    if model_key not in MODEL_KEYS:
        raise ModelIntegrityError(f'未知模型键：{model_key}')
    spec = manifest['models'][model_key]
    environment_name = MODEL_PATH_ENV[model_key]
    configured_path = model_path or os.environ.get(environment_name, '').strip()
    if not configured_path:
        raise ModelIntegrityError(f'未配置模型路径环境变量：{environment_name}')
    root = Path(configured_path)
    if not root.is_absolute():
        raise ModelIntegrityError(f'模型路径必须是绝对路径：{root}')
    if not root.is_dir():
        raise ModelIntegrityError(f'模型目录不存在：{root}')
    root = root.resolve()

    revision = spec['revision']
    revision_manifest_path = root / '.cache' / 'huggingface' / 'trees' / f'{revision}.json'
    try:
        revision_manifest = json.loads(revision_manifest_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelIntegrityError(f'固定 revision manifest 不可用：{revision_manifest_path}') from error
    tree_files = revision_manifest.get('files')
    if revision_manifest.get('format_version') != 1 or not isinstance(tree_files, dict):
        raise ModelIntegrityError(f'固定 revision manifest 格式无效：{revision_manifest_path}')

    expected_files: dict[str, list[Any]] = spec['files']
    if set(tree_files) != set(expected_files):
        raise ModelIntegrityError(f'固定 revision 文件清单不匹配：{root}')
    download_directory = root / '.cache' / 'huggingface' / 'download'
    metadata_by_relative_path = {
        path.relative_to(download_directory).as_posix().removesuffix('.metadata'): path
        for path in download_directory.rglob('*.metadata')
        if path.is_file()
    }
    if set(metadata_by_relative_path) != set(expected_files):
        raise ModelIntegrityError(f'固定 revision 下载 metadata 清单不匹配：{root}')
    for relative_path, metadata_path in metadata_by_relative_path.items():
        try:
            metadata_lines = metadata_path.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ModelIntegrityError(f'固定 revision metadata 不可读：{metadata_path}') from error
        tree_entry = tree_files[relative_path]
        expected_object_id = tree_entry.get('lfs_sha256', tree_entry.get('blob_id'))
        if (
            len(metadata_lines) < 2
            or metadata_lines[0] != revision
            or metadata_lines[1] != expected_object_id
        ):
            raise ModelIntegrityError(f'固定 revision metadata 内容不匹配：{metadata_path}')
    actual_root_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and '.cache' not in path.relative_to(root).parts
    }
    if actual_root_files != set(expected_files):
        raise ModelIntegrityError(f'模型根目录文件清单不匹配：{root}')

    file_reports: list[dict[str, Any]] = []
    for relative_path, expected in expected_files.items():
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ModelIntegrityError(f'模型文件必须是普通文件且不能是符号链接：{path}')
        expected_size, expected_sha256 = expected
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ModelIntegrityError(
                f'模型文件大小不匹配：{path}，expected={expected_size}，actual={actual_size}',
            )
        tree_entry = tree_files.get(relative_path)
        if not isinstance(tree_entry, dict):
            raise ModelIntegrityError(f'revision manifest 缺少文件：{relative_path}')
        tree_size = tree_entry.get('lfs_size', tree_entry.get('size'))
        if tree_size != expected_size:
            raise ModelIntegrityError(f'revision manifest 文件大小不匹配：{relative_path}')
        is_lfs = 'lfs_sha256' in tree_entry
        actual_sha256, actual_git_blob = _hash_file(path, include_git_blob=not is_lfs)
        if actual_sha256 != expected_sha256:
            raise ModelIntegrityError(f'模型文件 SHA-256 不匹配：{path}')
        if is_lfs:
            if tree_entry.get('lfs_sha256') != expected_sha256:
                raise ModelIntegrityError(f'revision manifest LFS SHA-256 不匹配：{relative_path}')
        elif tree_entry.get('blob_id') != actual_git_blob:
            raise ModelIntegrityError(f'revision manifest Git blob 不匹配：{relative_path}')
        file_reports.append({
            'path': relative_path,
            'size': actual_size,
            'sha256': actual_sha256,
        })

    safetensors_reports: dict[str, Any] = {}
    for relative_path, expectation in spec['safetensors'].items():
        actual = _read_safetensors(root / relative_path)
        expected_format = expectation['format']
        if actual['format'] != expected_format:
            raise ModelIntegrityError(
                f'Safetensors format 不匹配：{relative_path}，'
                f'expected={expected_format}，actual={actual["format"]}',
            )
        if actual['dtypes'] != [expectation['dtype']]:
            raise ModelIntegrityError(
                f'Safetensors dtype 不匹配：{relative_path}，actual={actual["dtypes"]}',
            )
        if actual['tensorCount'] != expectation['tensorCount']:
            raise ModelIntegrityError(f'Safetensors tensor 数量不匹配：{relative_path}')
        safetensors_reports[relative_path] = actual

    return {
        'model': model_key,
        'path': str(root),
        'revision': revision,
        'files': file_reports,
        'safetensors': safetensors_reports,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='离线校验 ElectronQwenSpeech 固定 MLX 语音模型。',
    )
    parser.add_argument('--model', action='append', choices=MODEL_KEYS, dest='models')
    parser.add_argument('--tts-path')
    parser.add_argument('--asr-path')
    parser.add_argument('--aligner-path')
    parser.add_argument('--compact', action='store_true')
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    requested_models = tuple(dict.fromkeys(arguments.models or MODEL_KEYS))
    paths = {
        'tts': arguments.tts_path,
        'asr': arguments.asr_path,
        'aligner': arguments.aligner_path,
    }
    try:
        reports = [validate_model(model_key, paths[model_key]) for model_key in requested_models]
    except ModelIntegrityError as error:
        print(json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(
        {'ok': True, 'models': reports},
        ensure_ascii=False,
        indent=None if arguments.compact else 2,
        sort_keys=True,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
