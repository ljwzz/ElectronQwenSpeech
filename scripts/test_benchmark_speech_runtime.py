from __future__ import annotations

import unittest
from pathlib import Path

from scripts.benchmark_speech_runtime import (
    _parse_arguments,
    calculate_character_throughput,
    calculate_tts_rates,
    memory_metric,
    summarize_memory_samples,
    target_python_path,
)


class BenchmarkSpeechRuntimeTest(unittest.TestCase):
    def test_selects_capability_specific_mlx_virtual_environment(self) -> None:
        root = Path('/repo')
        self.assertEqual(
            target_python_path('tts', root),
            root / 'services/tts-sidecar/.venv-mlx/bin/python',
        )
        self.assertEqual(
            target_python_path('asr', root),
            root / 'services/asr-sidecar/.venv-mlx/bin/python',
        )

    def test_accepts_capability_argument(self) -> None:
        self.assertEqual(_parse_arguments(['--capability', 'tts']).capability, 'tts')

    def test_calculates_tts_rate_metrics(self) -> None:
        self.assertEqual(
            calculate_tts_rates(2.0, 4.0),
            {'rtf': 0.5, 'realtimeMultiple': 2.0},
        )

    def test_rejects_invalid_tts_rate_inputs(self) -> None:
        for generation_seconds, audio_seconds in (
            (0.0, 1.0),
            (1.0, 0.0),
            (float('inf'), 1.0),
            (1.0, float('nan')),
        ):
            with self.subTest(
                generation_seconds=generation_seconds,
                audio_seconds=audio_seconds,
            ):
                with self.assertRaises(ValueError):
                    calculate_tts_rates(generation_seconds, audio_seconds)

    def test_calculates_unicode_character_throughput(self) -> None:
        self.assertEqual(
            calculate_character_throughput('银行A', 1.5),
            {'characterCount': 3, 'charactersPerSecond': 2.0},
        )

    def test_rejects_invalid_character_throughput_inputs(self) -> None:
        with self.assertRaises(ValueError):
            calculate_character_throughput('', 1.0)
        with self.assertRaises(ValueError):
            calculate_character_throughput('文本', 0.0)

    def test_summarizes_sampled_memory_without_calling_it_a_peak(self) -> None:
        samples = [
            {
                'stage': 'load',
                'os': {
                    'maxRssBytes': memory_metric(100, 'high_water_mark'),
                },
                'backend': {
                    'source': 'mlx.core',
                    'metrics': {
                        'activeMemoryBytes': memory_metric(50, 'sampled'),
                        'cacheMemoryBytes': memory_metric(70, 'sampled'),
                    },
                },
            },
            {
                'stage': 'hot',
                'os': {
                    'maxRssBytes': memory_metric(120, 'high_water_mark'),
                },
                'backend': {
                    'source': 'mlx.core',
                    'metrics': {
                        'activeMemoryBytes': memory_metric(60, 'sampled'),
                        'cacheMemoryBytes': memory_metric(65, 'sampled'),
                    },
                },
            },
        ]

        summary = summarize_memory_samples(samples)

        self.assertEqual(summary['os']['maxRssBytes']['value'], 120)
        active = summary['backend']['metrics']['activeMemoryBytes']
        cache = summary['backend']['metrics']['cacheMemoryBytes']
        self.assertEqual(active['value'], 60)
        self.assertEqual(active['measurement'], 'max_of_samples')
        self.assertEqual(cache['value'], 70)
        self.assertEqual(cache['measurement'], 'max_of_samples')

    def test_preserves_backend_high_water_mark_semantics(self) -> None:
        samples = [
            {
                'stage': 'load',
                'os': {
                    'maxRssBytes': memory_metric(100, 'high_water_mark'),
                },
                'backend': {
                    'source': 'mlx.core',
                    'metrics': {
                        'peakMemoryBytes': memory_metric(80, 'high_water_mark'),
                    },
                },
            },
            {
                'stage': 'hot',
                'os': {
                    'maxRssBytes': memory_metric(120, 'high_water_mark'),
                },
                'backend': {
                    'source': 'mlx.core',
                    'metrics': {
                        'peakMemoryBytes': memory_metric(90, 'high_water_mark'),
                    },
                },
            },
        ]

        summary = summarize_memory_samples(samples)

        peak = summary['backend']['metrics']['peakMemoryBytes']
        self.assertEqual(peak['value'], 90)
        self.assertEqual(peak['measurement'], 'high_water_mark')


if __name__ == '__main__':
    unittest.main()
