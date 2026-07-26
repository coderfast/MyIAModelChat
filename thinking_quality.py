"""
Thinking quality validation module.
Validates that generated thinking is real reasoning, not meta-commentary.
"""
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class QualityResult:
    """Result of quality validation for a single sample."""
    valid: bool
    score: float
    issues: List[str] = field(default_factory=list)


@dataclass
class BatchQualityReport:
    """Report of quality validation for a batch of samples."""
    total: int = 0
    valid: int = 0
    too_short: int = 0
    meta_commentary: int = 0
    no_derivation: int = 0
    low_quality: int = 0

    @property
    def valid_rate(self) -> float:
        return self.valid / max(1, self.total)

    def summary(self) -> str:
        return (
            f"Quality: {self.valid}/{self.total} valid ({self.valid_rate:.1%}) | "
            f"Too short: {self.too_short} | Meta: {self.meta_commentary} | "
            f"No derivation: {self.no_derivation} | Low quality: {self.low_quality}"
        )


META_PATTERNS = [
    re.compile(r'^(el usuario|the user|el humano|the human)', re.IGNORECASE),
    re.compile(r'^(saludo|greeting|despedida|farewell)', re.IGNORECASE),
    re.compile(r'^(respondo|i respond|contestando|answering)', re.IGNORECASE),
    re.compile(r'^(el bot|the bot|asistente|assistant)', re.IGNORECASE),
]

STEP_INDICATORS = [
    'porque', 'por lo tanto', 'primero', 'paso', 'análisis',
    'entonces', 'sin embargo', 'además', 'en cambio', 'consiste',
    'because', 'therefore', 'first', 'step', 'analysis',
    'however', 'additionally', 'furthermore', 'consists',
]


def validate_thinking(thinking: str, answer: str = '', question: str = '') -> QualityResult:
    """Validate that thinking is real reasoning, not meta-commentary."""
    issues = []
    score = 0.0

    if not thinking or len(thinking.strip()) < 15:
        return QualityResult(valid=False, score=0.0, issues=['too_short'])

    if len(thinking) < 30:
        issues.append('short')
        score += 0.2
    else:
        score += 0.4

    is_meta = False
    for pattern in META_PATTERNS:
        if pattern.match(thinking.strip()):
            is_meta = True
            break
    if is_meta:
        issues.append('meta_commentary')
    else:
        score += 0.3

    if answer:
        answer_words = set(answer.lower().split()[:5])
        thinking_words = set(thinking.lower().split())
        if answer_words and answer_words.intersection(thinking_words):
            score += 0.3
        else:
            issues.append('no_derivation')
    else:
        score += 0.2

    has_steps = any(ind in thinking.lower() for ind in STEP_INDICATORS)
    has_length = len(thinking.split()) >= 8
    if has_steps or has_length:
        score += 0.2
    else:
        issues.append('low_reasoning')

    valid = score >= 0.5 and not is_meta
    return QualityResult(valid=valid, score=score, issues=issues)


def validate_thinking_batch(samples: List[Dict[str, Any]]) -> BatchQualityReport:
    """Validate a batch of samples with thinking."""
    report = BatchQualityReport()

    for sample in samples:
        thinking = sample.get('thinking', '')
        answer = sample.get('output', sample.get('answer', ''))
        question = sample.get('input', sample.get('question', ''))

        report.total += 1
        result = validate_thinking(thinking, answer, question)

        if result.valid:
            report.valid += 1
        else:
            if 'too_short' in result.issues:
                report.too_short += 1
            if 'meta_commentary' in result.issues:
                report.meta_commentary += 1
            if 'no_derivation' in result.issues:
                report.no_derivation += 1
            if 'low_reasoning' in result.issues or 'low_quality' in result.issues:
                report.low_quality += 1

    return report


def filter_low_quality(samples: List[Dict[str, Any]], min_score: float = 0.4) -> List[Dict[str, Any]]:
    """Filter out low-quality thinking samples."""
    filtered = []
    for sample in samples:
        thinking = sample.get('thinking', '')
        answer = sample.get('output', sample.get('answer', ''))
        result = validate_thinking(thinking, answer)
        if result.score >= min_score:
            filtered.append(sample)
    return filtered
