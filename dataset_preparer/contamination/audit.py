"""
Audit reporting system for contamination filtering.

Generates structured JSON reports with per-source statistics,
filter results, and pipeline metadata.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = os.path.join(os.path.dirname(__file__), 'reports')


@dataclass
class SourceReport:
    """Report for a single data source."""
    original_count: int = 0
    after_noise: int = 0
    after_quality: int = 0
    after_dedup: int = 0
    after_balance: int = 0
    after_language: int = 0
    after_leakage: int = 0
    discarded: Dict[str, int] = field(default_factory=dict)

    @property
    def total_discarded(self) -> int:
        return sum(self.discarded.values())


@dataclass
class AuditReport:
    """Complete audit report for a filtering pipeline run."""
    timestamp: str = ''
    pipeline_version: str = '1.0'
    filters_applied: List[str] = field(default_factory=list)
    per_source: Dict[str, SourceReport] = field(default_factory=dict)
    cross_source_duplicates: int = 0
    balance_report: Dict[str, float] = field(default_factory=dict)
    thinking_stats: Dict[str, int] = field(default_factory=dict)
    final_count: int = 0
    total_discarded: int = 0
    retention_rate: float = 0.0
    processing_time_seconds: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'pipeline_version': self.pipeline_version,
            'filters_applied': self.filters_applied,
            'per_source': {
                name: asdict(report) for name, report in self.per_source.items()
            },
            'cross_source_duplicates': self.cross_source_duplicates,
            'balance_report': self.balance_report,
            'thinking_stats': self.thinking_stats,
            'final_count': self.final_count,
            'total_discarded': self.total_discarded,
            'retention_rate': self.retention_rate,
            'processing_time_seconds': self.processing_time_seconds,
        }

    def summary(self) -> str:
        lines = [
            f"=== Audit Report ({self.timestamp}) ===",
            f"Filters applied: {', '.join(self.filters_applied) or 'none'}",
            f"Final count: {self.final_count}",
            f"Total discarded: {self.total_discarded}",
            f"Retention rate: {self.retention_rate:.1%}",
            f"Processing time: {self.processing_time_seconds:.1f}s",
            "",
            "Per-source breakdown:",
        ]
        for name, report in self.per_source.items():
            lines.append(
                f"  {name}: {report.original_count} -> "
                f"{report.after_leakage or report.after_language or report.after_dedup or report.original_count} "
                f"({report.total_discarded} discarded)"
            )
        if self.thinking_stats:
            lines.append("")
            lines.append(f"Thinking: {self.thinking_stats.get('with_thinking', 0)}/{self.thinking_stats.get('total_samples', 0)} enriched")
        return "\n".join(lines)


class AuditReporter:
    """Generates and saves audit reports."""

    def __init__(self, report_dir: str = None, enabled: bool = True):
        """
        Initialize audit reporter.

        Args:
            report_dir: Directory to save reports
            enabled: Whether reporting is enabled
        """
        self.report_dir = report_dir or DEFAULT_REPORT_DIR
        self.enabled = enabled
        self._start_time = None
        self.report = AuditReport()

        if self.enabled:
            os.makedirs(self.report_dir, exist_ok=True)

    def start_timer(self):
        """Start processing timer."""
        self._start_time = datetime.now()

    def stop_timer(self):
        """Stop processing timer and update report."""
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.report.processing_time_seconds = round(elapsed, 2)

    def set_filter(self, filter_name: str):
        """Record that a filter was applied."""
        if filter_name not in self.report.filters_applied:
            self.report.filters_applied.append(filter_name)

    def update_source(
        self,
        source_name: str,
        original_count: int = None,
        after_noise: int = None,
        after_quality: int = None,
        after_dedup: int = None,
        after_balance: int = None,
        after_language: int = None,
        after_leakage: int = None,
        discarded: Dict[str, int] = None
    ):
        """Update report for a specific source."""
        if source_name not in self.report.per_source:
            self.report.per_source[source_name] = SourceReport()

        sr = self.report.per_source[source_name]
        if original_count is not None:
            sr.original_count = original_count
        if after_noise is not None:
            sr.after_noise = after_noise
        if after_quality is not None:
            sr.after_quality = after_quality
        if after_dedup is not None:
            sr.after_dedup = after_dedup
        if after_balance is not None:
            sr.after_balance = after_balance
        if after_language is not None:
            sr.after_language = after_language
        if after_leakage is not None:
            sr.after_leakage = after_leakage
        if discarded:
            sr.discarded.update(discarded)

    def set_cross_source_duplicates(self, count: int):
        """Record cross-source duplicate count."""
        self.report.cross_source_duplicates = count

    def set_balance(self, ratios: Dict[str, float]):
        """Record source balance ratios."""
        self.report.balance_report = ratios

    def set_thinking_stats(self, stats: Dict[str, int]):
        """Record thinking generation statistics."""
        self.report.thinking_stats = stats

    def set_final(self, count: int, discarded: int):
        """Set final counts."""
        self.report.final_count = count
        self.report.total_discarded = discarded
        total = count + discarded
        self.report.retention_rate = count / total if total > 0 else 0.0

    def save(self, filename: str = None) -> Optional[str]:
        """
        Save report to JSON file.

        Args:
            filename: Optional filename (default: auto-generated)

        Returns:
            Path to saved file, or None if disabled
        """
        if not self.enabled:
            return None

        self.stop_timer()

        if filename is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'filter_report_{ts}.json'

        filepath = os.path.join(self.report_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.report.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"  Audit report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"  Could not save audit report: {e}")
            return None

    def print_summary(self):
        """Print report summary to logger."""
        if self.report.filters_applied:
            logger.info(self.report.summary())
