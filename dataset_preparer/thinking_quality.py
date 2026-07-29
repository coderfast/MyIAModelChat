"""
Thinking quality validation module - V2 with advanced detection.
Validates that generated thinking is real reasoning, not meta-commentary or re-declaration.
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
    re_declaration: int = 0
    placeholder: int = 0
    no_derivation: int = 0
    low_quality: int = 0

    @property
    def valid_rate(self) -> float:
        return self.valid / max(1, self.total)

    def summary(self) -> str:
        return (
            f"Quality: {self.valid}/{self.total} valid ({self.valid_rate:.1%}) | "
            f"Too short: {self.too_short} | Meta: {self.meta_commentary} | "
            f"Re-declaration: {self.re_declaration} | Placeholder: {self.placeholder} | "
            f"No derivation: {self.no_derivation} | Low quality: {self.low_quality}"
        )


# Original meta-commentary patterns
META_PATTERNS = [
    re.compile(r'^(el usuario|the user|el humano|the human)\s*(me\s+)?(saluda|despide|pregunta|pide)', re.IGNORECASE),
    re.compile(r'^(saludo|greeting|despedida|farewell)$', re.IGNORECASE),
    re.compile(r'^(respondo|i respond|contestando|answering)\s*(con|with)', re.IGNORECASE),
    re.compile(r'^(el bot|the bot|asistente|assistant)\s*(responde|answer)', re.IGNORECASE),
]

# NEW: Re-declaration patterns (bypass detection)
RE_DECLARATION_PATTERNS = [
    # Q&A re-declaration (the main bypass)
    re.compile(r'la pregunta es:.*la respuesta', re.IGNORECASE),
    re.compile(r'la pregunta:.*respuesta correcta', re.IGNORECASE),
    re.compile(r'pregunta del usuario:.*respuesta del bot', re.IGNORECASE),
    # Also match without colon but with specific structure
    re.compile(r'la respuesta correcta es', re.IGNORECASE),
    # Information re-statement
    re.compile(r'se presenta información sobre:.*la respuesta contiene', re.IGNORECASE),
    re.compile(r'esto indica que la información.*puede responderse', re.IGNORECASE),
    re.compile(r'se presenta información sobre.*la respuesta contiene', re.IGNORECASE),
]

# NEW: Placeholder patterns (generic filler)
PLACEHOLDER_PATTERNS = [
    # Word count placeholders
    re.compile(r'contiene \d+ palabras', re.IGNORECASE),
    re.compile(r'con \d+ palabras de contenido', re.IGNORECASE),
    re.compile(r'texto con \d+ palabras', re.IGNORECASE),
    # Generic relevance (only when combined with other filler)
    re.compile(r'información relevante.*puede ser utilizada', re.IGNORECASE),
    re.compile(r'contiene información que debe ser procesada', re.IGNORECASE),
    re.compile(r'información que puede ser procesada', re.IGNORECASE),
    # Only reject if it's JUST a placeholder without real analysis
    re.compile(r'^.*presenta información (técnica|académica)\.?$', re.IGNORECASE),
    re.compile(r'^.*contiene (datos|información) sobre el tema\.?$', re.IGNORECASE),
]

# Step indicators (reasoning markers)
STEP_INDICATORS = [
    'porque', 'por lo tanto', 'primero', 'paso', 'análisis',
    'entonces', 'sin embargo', 'además', 'en cambio', 'consiste',
    'because', 'therefore', 'first', 'step', 'analysis',
    'however', 'additionally', 'furthermore', 'consists',
    # NEW: Reasoning verbs
    'analizando', 'identificando', 'observando', 'detectando',
    'analyzing', 'identifying', 'observing', 'detecting',
    # NEW: Content markers
    'conceptos', 'entidades', 'términos', 'secciones',
    'concepts', 'entities', 'terms', 'sections',
]


def validate_thinking(thinking: str, answer: str = '', question: str = '') -> QualityResult:
    """
    Validate that thinking is real reasoning, not meta-commentary or re-declaration.
    V2 with advanced detection of bypass patterns.
    """
    issues = []
    score = 0.0

    # Check minimum length
    if not thinking or len(thinking.strip()) < 15:
        return QualityResult(valid=False, score=0.0, issues=['too_short'])

    # Length scoring
    if len(thinking) < 30:
        issues.append('short')
        score += 0.2
    else:
        score += 0.4

    # Check for meta-commentary (original patterns)
    is_meta = False
    for pattern in META_PATTERNS:
        if pattern.match(thinking.strip()):
            is_meta = True
            issues.append('meta_commentary')
            break

    # NEW: Check for re-declaration (bypass detection)
    is_re_declaration = False
    if not is_meta:
        for pattern in RE_DECLARATION_PATTERNS:
            if pattern.search(thinking):
                is_re_declaration = True
                issues.append('re_declaration')
                break

    # NEW: Check for placeholder content
    is_placeholder = False
    if not is_meta and not is_re_declaration:
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(thinking):
                is_placeholder = True
                issues.append('placeholder')
                break

    # Check for reasoning indicators
    has_steps = any(ind in thinking.lower() for ind in STEP_INDICATORS)

    # Check for vocabulary diversity (not just re-declaration)
    has_vocabulary_diversity = False
    if answer:
        answer_words = set(answer.lower().split())
        thinking_words = set(thinking.lower().split())
        # If thinking has many unique words not in answer, it's diverse
        unique_thinking = thinking_words - answer_words
        has_vocabulary_diversity = len(unique_thinking) > len(thinking_words) * 0.3

    # Check for answer derivation
    has_answer_derivation = False
    if answer:
        answer_words = set(answer.lower().split()[:5])
        thinking_words = set(thinking.lower().split())
        has_answer_derivation = bool(answer_words and answer_words.intersection(thinking_words))

    # NEW: Check for logical connectors
    has_logical_connectors = False
    logical_connectors = ['en primer lugar', 'además', 'por otro lado', 'finalmente',
                         'first', 'additionally', 'furthermore', 'finally',
                         'por lo tanto', 'therefore', 'sin embargo', 'however']
    for connector in logical_connectors:
        if connector in thinking.lower():
            has_logical_connectors = True
            break

    # Scoring based on checks
    if is_meta and not has_steps and not has_answer_derivation:
        score -= 0.3  # Penalty for pure meta-commentary
    elif is_re_declaration:
        score -= 0.2  # Penalty for re-declaration
    elif is_placeholder:
        score -= 0.1  # Penalty for placeholder
    else:
        score += 0.3  # Bonus for non-meta content

    if has_answer_derivation:
        score += 0.2
    elif answer:
        issues.append('no_derivation')

    if has_vocabulary_diversity:
        score += 0.2

    if has_logical_connectors:
        score += 0.1

    if has_steps:
        score += 0.2

    # Final validation
    valid = (
        score >= 0.5 and
        not (is_meta and not has_steps and not has_answer_derivation) and
        not is_re_declaration and
        not is_placeholder
    )

    return QualityResult(valid=valid, score=max(0.0, score), issues=issues)


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
            if 're_declaration' in result.issues:
                report.re_declaration += 1
            if 'placeholder' in result.issues:
                report.placeholder += 1
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
