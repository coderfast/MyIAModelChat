"""
Benchmark for thinking quality metrics.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_preparer.thinking_engine import ThinkingEngine
from dataset_preparer.thinking_quality import validate_thinking, BatchQualityReport
from dataset_preparer.csv.thinking import CSVThinkingGenerator
from dataset_preparer.aiml.thinking import AIMLThinkingGenerator
from dataset_preparer.pdf.thinking import PDFThinkingGenerator
from dataset_preparer.epub.thinking import EPUBThinkingGenerator
from dataset_preparer.hf.thinking import HFThinkingGenerator
from dataset_preparer.web.thinking import WebThinkingGenerator


class ThinkingBenchmark:
    """Benchmark for thinking quality and performance."""

    def __init__(self):
        self.engine = ThinkingEngine(depth='adaptive')
        self.results = {}

    def run_benchmark(self):
        """Run complete benchmark."""
        print("=" * 60)
        print("THINKING ENGINE BENCHMARK")
        print("=" * 60)

        # Test samples
        samples = self._get_test_samples()

        # Benchmark each source
        for source_name, generator, sample in samples:
            print(f"\n--- {source_name} ---")
            self._benchmark_source(source_name, generator, sample)

        # Overall metrics
        print("\n" + "=" * 60)
        print("OVERALL METRICS")
        print("=" * 60)
        self._print_overall_metrics()

    def _get_test_samples(self):
        """Get test samples for benchmarking."""
        return [
            ('CSV', CSVThinkingGenerator(self.engine, depth='adaptive'),
             {'input': 'What is machine learning?', 'output': 'Machine learning is a subset of AI.'}),
            ('AIML', AIMLThinkingGenerator(self.engine, depth='adaptive'),
             {'input': 'hello', 'output': 'Hello! How can I help you?'}),
            ('PDF', PDFThinkingGenerator(self.engine, depth='adaptive'),
             {'input_ids': 'Python es un lenguaje de programación interpretado de alto nivel.'}),
            ('EPUB', EPUBThinkingGenerator(self.engine, depth='adaptive'),
             {'input_ids': 'El capítulo describe la historia del héroe en su viaje.'}),
            ('HF', HFThinkingGenerator(self.engine, depth='adaptive'),
             {'question': 'What is neural network?', 'answer': 'A neural network is a computing system.'}),
            ('Web', WebThinkingGenerator(self.engine, depth='adaptive'),
             {'input_ids': 'This documentation explains how to install the package.'}),
        ]

    def _benchmark_source(self, name, generator, sample):
        """Benchmark a single source."""
        # Generate thinking
        start_time = time.time()
        result = generator.generate(sample)
        generation_time = time.time() - start_time

        thinking = result.get('thinking', '')

        # Validate
        validation = validate_thinking(thinking, sample.get('output', ''))

        # Metrics
        word_count = len(thinking.split())
        char_count = len(thinking)

        print(f"  Generated: {word_count} words, {char_count} chars")
        print(f"  Time: {generation_time*1000:.1f}ms")
        print(f"  Valid: {validation.valid}, Score: {validation.score:.2f}")
        if validation.issues:
            print(f"  Issues: {', '.join(validation.issues)}")

        self.results[name] = {
            'word_count': word_count,
            'char_count': char_count,
            'generation_time': generation_time,
            'valid': validation.valid,
            'score': validation.score,
            'issues': validation.issues,
        }

    def _print_overall_metrics(self):
        """Print overall benchmark metrics."""
        if not self.results:
            return

        total_samples = len(self.results)
        valid_samples = sum(1 for r in self.results.values() if r['valid'])
        avg_score = sum(r['score'] for r in self.results.values()) / total_samples
        avg_time = sum(r['generation_time'] for r in self.results.values()) / total_samples
        avg_words = sum(r['word_count'] for r in self.results.values()) / total_samples

        print(f"  Samples: {total_samples}")
        print(f"  Valid: {valid_samples}/{total_samples} ({valid_samples/total_samples*100:.1f}%)")
        print(f"  Avg Score: {avg_score:.2f}")
        print(f"  Avg Time: {avg_time*1000:.1f}ms")
        print(f"  Avg Words: {avg_words:.1f}")

        # Target metrics
        print("\n  TARGETS:")
        print(f"  Valid Rate: {'PASS' if valid_samples/total_samples >= 0.9 else 'FAIL'} ({valid_samples/total_samples*100:.1f}% >= 90%)")
        print(f"  Avg Score: {'PASS' if avg_score >= 0.7 else 'FAIL'} ({avg_score:.2f} >= 0.7)")


def main():
    """Run benchmark."""
    benchmark = ThinkingBenchmark()
    benchmark.run_benchmark()


if __name__ == '__main__':
    main()
