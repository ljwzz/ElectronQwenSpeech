from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_mlx_speech_models import (
    MANIFEST_PATH,
    ModelIntegrityError,
    _read_safetensors,
    validate_model,
)


def _write_safetensors(path: Path, header: dict[str, object], data: bytes) -> None:
    encoded_header = json.dumps(header, separators=(',', ':')).encode()
    path.write_bytes(struct.pack('<Q', len(encoded_header)) + encoded_header + data)


class SafetensorsValidationTest(unittest.TestCase):
    def test_manifest_does_not_contain_machine_specific_paths(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        for spec in manifest['models'].values():
            self.assertNotIn('defaultPath', spec)

    def test_reads_format_dtype_tensor_count_and_contiguous_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            _write_safetensors(path, {
                '__metadata__': {'format': 'mlx'},
                'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
            }, b'\x00\x00\x00\x00')

            self.assertEqual(_read_safetensors(path), {
                'format': 'mlx',
                'dtypes': ['F32'],
                'tensorCount': 1,
            })

    def test_rejects_non_contiguous_tensor_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            _write_safetensors(path, {
                'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [1, 5]},
            }, b'\x00\x00\x00\x00\x00')

            with self.assertRaisesRegex(ModelIntegrityError, '数据不连续'):
                _read_safetensors(path)

    def test_rejects_relative_model_path(self) -> None:
        with self.assertRaisesRegex(ModelIntegrityError, '绝对路径'):
            validate_model('tts', 'relative/model')

    def test_model_path_environment_supplies_model_location(self) -> None:
        custom_path = '/missing/custom-tts-model'
        with patch.dict(
            os.environ,
            {'ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH': custom_path},
        ):
            with self.assertRaisesRegex(ModelIntegrityError, custom_path):
                validate_model('tts')

    def test_requires_model_path_environment_variable(self) -> None:
        with patch.dict(
            os.environ,
            {'ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH': ''},
        ):
            with self.assertRaisesRegex(
                ModelIntegrityError,
                'ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH',
            ):
                validate_model('tts')


if __name__ == '__main__':
    unittest.main()
