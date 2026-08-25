"""
Source validators for data quality assurance.
Each data source has its own validator that understands source-specific issues.
Classification: good / fixable / discardable
"""
import re
import html
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Report of data quality for a specific source."""
    source: str
    total_samples: int = 0
    good: int = 0
    fixable: int = 0
    discardable: int = 0
    fixes_applied: Dict[str, int] = field(default_factory=dict)
    discarded_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def retention_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.good + self.fixable) / self.total_samples

    def summary(self) -> str:
        return (
            f"[{self.source}] Total: {self.total_samples} | "
            f"Good: {self.good} | Fixable: {self.fixable} | "
            f"Discardable: {self.discardable} | "
            f"Retention: {self.retention_rate:.1%}"
        )


class SourceValidator:
    """Base class for all source validators."""

    def classify(self, sample: dict) -> Literal['good', 'fixable', 'discardable']:
        """Classify a sample's quality."""
        raise NotImplementedError

    def fix(self, sample: dict) -> dict:
        """Fix a fixable sample. Returns fixed sample."""
        raise NotImplementedError

    def report(self) -> QualityReport:
        """Return the quality report."""
        raise NotImplementedError

    def validate_batch(self, samples: list) -> tuple:
        """Validate a batch of samples. Returns (cleaned_samples, report)."""
        raise NotImplementedError


class GenericValidator(SourceValidator):
    """Generic validator for simple sources (AIML, CSV)."""

    MIN_LENGTH = 5
    MAX_LENGTH = 2000

    URL_PATTERN = re.compile(
        r'https?://[^\s<>\"\'\)]+|'
        r'www\.[^\s<>\"\'\)]+'
    )
    EMAIL_PATTERN = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
    PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}')
    SYSTEM_PATH_PATTERN = re.compile(r'([A-Z]:\\|/home/|/usr/|/var/|/etc/|~/)')
    CODE_PATTERN = re.compile(
        r'("""|\'\'\'|```|def\s+\w+\s*\(|class\s+\w+|import\s+\w+|'
        r'function\s*\(|var\s+\w+\s*=|document\.|window\.)',
        re.IGNORECASE
    )
    BOILERPLATE_PATTERN = re.compile(
        r'(cookie|privacy|terms of use|all rights reserved|copyright|'
        r'subscribe|newsletter|sign up|log in|sign in|'
        r'advertisement|sponsored|click here|read more)',
        re.IGNORECASE
    )
    NUMBERS_ONLY_PATTERN = re.compile(r'^[\d\s\.\,\-\+\(\)]+$')

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.report_data = QualityReport(source=source_name)
        self._fix_counts = {}
        self._discard_counts = {}

    def classify(self, sample: dict) -> Literal['good', 'fixable', 'discardable']:
        text = self._get_main_text(sample)
        if not text:
            self._discard_counts['empty_text'] = self._discard_counts.get('empty_text', 0) + 1
            return 'discardable'

        if len(text) < self.MIN_LENGTH:
            self._discard_counts['too_short'] = self._discard_counts.get('too_short', 0) + 1
            return 'discardable'

        if len(text) > self.MAX_LENGTH:
            self._fix_counts['truncated'] = self._fix_counts.get('truncated', 0) + 1
            return 'fixable'

        if self._has_encoding_issues(text):
            self._fix_counts['encoding'] = self._fix_counts.get('encoding', 0) + 1
            return 'fixable'

        if self._has_html_tags(text):
            self._fix_counts['html_tags'] = self._fix_counts.get('html_tags', 0) + 1
            return 'fixable'

        if self._has_noise_content(text):
            return 'discardable'

        self.report_data.good += 1
        return 'good'

    def fix(self, sample: dict) -> dict:
        result = dict(sample)
        for key in result:
            if isinstance(result[key], str):
                result[key] = self._clean_text(result[key])
                # Truncate to MAX_LENGTH
                if len(result[key]) > self.MAX_LENGTH:
                    result[key] = result[key][:self.MAX_LENGTH]
        return result

    def report(self) -> QualityReport:
        self.report_data.fixes_applied = dict(self._fix_counts)
        self.report_data.discarded_reasons = dict(self._discard_counts)
        return self.report_data

    def validate_batch(self, samples: list) -> tuple:
        cleaned = []
        self.report_data = QualityReport(source=self.source_name)
        self._fix_counts = {}
        self._discard_counts = {}

        for sample in samples:
            classification = self.classify(sample)
            self.report_data.total_samples += 1

            if classification == 'good':
                cleaned.append(sample)
            elif classification == 'fixable':
                fixed = self.fix(sample)
                cleaned.append(fixed)
                self.report_data.fixable += 1
            else:
                self.report_data.discardable += 1

        self.report_data.good = len(cleaned) - self.report_data.fixable
        logger.info(self.report_data.summary())
        return cleaned, self.report_data

    def _get_main_text(self, sample: dict) -> str:
        for key in ('input', 'input_ids', 'text', 'question'):
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return ''

    def _has_encoding_issues(self, text: str) -> bool:
        replacement_count = text.count('\ufffd')
        if replacement_count > 0:
            return True
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            return True
        return False

    def _has_html_tags(self, text: str) -> bool:
        return bool(re.search(r'<[a-zA-Z][^>]*>', text))

    def _has_noise_content(self, text: str) -> bool:
        """Detect noise: URLs, emails, phone numbers, paths, code, boilerplate, numbers-only."""
        url_matches = self.URL_PATTERN.findall(text)
        if len(url_matches) > 2 or (len(url_matches) >= 1 and len(text) < 50):
            self._discard_counts['contains_urls'] = self._discard_counts.get('contains_urls', 0) + 1
            return True

        if self.EMAIL_PATTERN.search(text):
            self._discard_counts['contains_emails'] = self._discard_counts.get('contains_emails', 0) + 1
            return True

        phone_matches = self.PHONE_PATTERN.findall(text)
        if len(phone_matches) > 1:
            self._discard_counts['contains_phones'] = self._discard_counts.get('contains_phones', 0) + 1
            return True

        path_matches = self.SYSTEM_PATH_PATTERN.findall(text)
        if len(path_matches) > 1 or (len(path_matches) >= 1 and len(text) < 80):
            self._discard_counts['contains_paths'] = self._discard_counts.get('contains_paths', 0) + 1
            return True

        code_matches = self.CODE_PATTERN.findall(text)
        if len(code_matches) > 2:
            self._discard_counts['contains_code'] = self._discard_counts.get('contains_code', 0) + 1
            return True

        boilerplate_count = len(self.BOILERPLATE_PATTERN.findall(text))
        if boilerplate_count >= 2:
            self._discard_counts['boilerplate'] = self._discard_counts.get('boilerplate', 0) + 1
            return True

        if self.NUMBERS_ONLY_PATTERN.match(text.strip()):
            self._discard_counts['numbers_only'] = self._discard_counts.get('numbers_only', 0) + 1
            return True

        return False

    def _clean_text(self, text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = unicodedata.normalize('NFKC', text)
        return text


class AIMLValidator(GenericValidator):
    """Validator for AIML source files."""

    AIML_TAG_PATTERN = re.compile(
        r'<(random|srai|condition|set|get|star|bot|topic|that|person|li|learn|eval|index)[^>]*>.*?</\1>',
        re.DOTALL
    )
    AIML_SELFCLOSING = re.compile(r'<(star|bot|person|br|hr)\s*/?\s*>')
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    WILDCARD_PATTERN = re.compile(r'^[_*\s]+$')

    def __init__(self):
        super().__init__('aiml')

    def classify(self, sample: dict) -> Literal['good', 'fixable', 'discardable']:
        pattern = sample.get('input', '')
        template = sample.get('output', '')

        if not pattern or not template:
            self._discard_counts['empty_pattern_or_template'] = self._discard_counts.get('empty_pattern_or_template', 0) + 1
            return 'discardable'

        if self.WILDCARD_PATTERN.match(pattern.strip()):
            self._discard_counts['wildcard_only'] = self._discard_counts.get('wildcard_only', 0) + 1
            return 'discardable'

        if self.AIML_TAG_PATTERN.search(template) or self.AIML_SELFCLOSING.search(template):
            self._fix_counts['aiml_tags'] = self._fix_counts.get('aiml_tags', 0) + 1
            return 'fixable'

        if self.HTML_TAG_PATTERN.search(template):
            self._fix_counts['html_tags'] = self._fix_counts.get('html_tags', 0) + 1
            return 'fixable'

        if self._has_encoding_issues(pattern) or self._has_encoding_issues(template):
            self._fix_counts['encoding'] = self._fix_counts.get('encoding', 0) + 1
            return 'fixable'

        if len(template.strip()) < 2:
            self._discard_counts['empty_template_after_check'] = self._discard_counts.get('empty_template_after_check', 0) + 1
            return 'discardable'

        self.report_data.good += 1
        return 'good'

    def fix(self, sample: dict) -> dict:
        result = dict(sample)
        if 'output' in result and isinstance(result['output'], str):
            result['output'] = self._clean_aiml_template(result['output'])
        if 'input' in result and isinstance(result['input'], str):
            result['input'] = self._clean_text(result['input'])
        return result

    def _clean_aiml_template(self, text: str) -> str:
        text = self.AIML_TAG_PATTERN.sub('', text)
        text = self.AIML_SELFCLOSING.sub('', text)
        text = self.HTML_TAG_PATTERN.sub('', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = unicodedata.normalize('NFKC', text)
        return text


class PDFValidator(SourceValidator):
    """Validator for PDF source files."""

    PAGE_NUMBER_PATTERN = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
    HYPHEN_LINE_BREAK = re.compile(r'(\w)-\s*\n\s*(\w)')
    FORM_FEED = re.compile(r'\x0c')
    REPEATED_HEADER = re.compile(r'^(.{1,30})\n\1\n', re.MULTILINE)
    BOOKMARK_PATTERN = re.compile(
        r'(table of contents|contents|chapter \d|section \d|'
        r'appendix|bibliography|index|glossary|acknowledgments)',
        re.IGNORECASE
    )
    PAGE_REF_PATTERN = re.compile(r'(page|pag|p\.)\s*\d+', re.IGNORECASE)
    URL_IN_PDF = re.compile(r'https?://[^\s<>\"\'\)]+|www\.[^\s<>\"\'\)]+')

    def __init__(self):
        self.source_name = 'pdf'
        self.report_data = QualityReport(source='pdf')
        self._fix_counts = {}
        self._discard_counts = {}

    def classify(self, sample: dict) -> Literal['good', 'fixable', 'discardable']:
        text = self._get_text(sample)
        if not text:
            self._discard_counts['empty_text'] = self._discard_counts.get('empty_text', 0) + 1
            return 'discardable'

        if len(text.strip()) < 20:
            self._discard_counts['too_short'] = self._discard_counts.get('too_short', 0) + 1
            return 'discardable'

        replacement_ratio = text.count('\ufffd') / max(len(text), 1)
        if replacement_ratio > 0.05:
            self._discard_counts['garbled_text'] = self._discard_counts.get('garbled_text', 0) + 1
            return 'discardable'

        needs_fix = False
        if self.HYPHEN_LINE_BREAK.search(text):
            self._fix_counts['hyphen_breaks'] = self._fix_counts.get('hyphen_breaks', 0) + 1
            needs_fix = True
        if self.FORM_FEED.search(text):
            self._fix_counts['form_feeds'] = self._fix_counts.get('form_feeds', 0) + 1
            needs_fix = True
        if self.PAGE_NUMBER_PATTERN.search(text):
            self._fix_counts['page_numbers'] = self._fix_counts.get('page_numbers', 0) + 1
            needs_fix = True
        if self.REPEATED_HEADER.search(text):
            self._fix_counts['repeated_headers'] = self._fix_counts.get('repeated_headers', 0) + 1
            needs_fix = True

        if needs_fix:
            return 'fixable'

        if self.BOOKMARK_PATTERN.search(text) and len(text.strip()) < 200:
            self._discard_counts['bookmark_or_toc'] = self._discard_counts.get('bookmark_or_toc', 0) + 1
            return 'discardable'

        if self.PAGE_REF_PATTERN.search(text) and len(text.strip()) < 50:
            self._discard_counts['page_reference'] = self._discard_counts.get('page_reference', 0) + 1
            return 'discardable'

        url_count = len(self.URL_IN_PDF.findall(text))
        if url_count > 2 or (url_count >= 1 and len(text) < 60):
            self._discard_counts['contains_urls'] = self._discard_counts.get('contains_urls', 0) + 1
            return 'discardable'

        self.report_data.good += 1
        return 'good'

    def fix(self, sample: dict) -> dict:
        result = dict(sample)
        for key in result:
            if isinstance(result[key], str):
                result[key] = self._clean_pdf_text(result[key])
        return result

    def report(self) -> QualityReport:
        self.report_data.fixes_applied = dict(self._fix_counts)
        self.report_data.discarded_reasons = dict(self._discard_counts)
        return self.report_data

    def validate_batch(self, samples: list) -> tuple:
        cleaned = []
        self.report_data = QualityReport(source='pdf')
        self._fix_counts = {}
        self._discard_counts = {}

        for sample in samples:
            classification = self.classify(sample)
            self.report_data.total_samples += 1

            if classification == 'good':
                cleaned.append(sample)
            elif classification == 'fixable':
                fixed = self.fix(sample)
                cleaned.append(fixed)
                self.report_data.fixable += 1
            else:
                self.report_data.discardable += 1

        self.report_data.good = len(cleaned) - self.report_data.fixable
        logger.info(self.report_data.summary())
        return cleaned, self.report_data

    def _get_text(self, sample: dict) -> str:
        for key in ('input_ids', 'text', 'input'):
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return ''

    def _clean_pdf_text(self, text: str) -> str:
        text = self.HYPHEN_LINE_BREAK.sub(r'\1\2', text)
        text = self.FORM_FEED.sub(' ', text)
        text = self.PAGE_NUMBER_PATTERN.sub('', text)
        text = self.REPEATED_HEADER.sub('', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()


class EPUBValidator(SourceValidator):
    """Validator for EPUB source files."""

    HTML_TAG = re.compile(r'<[^>]+>')
    STYLE_SCRIPT = re.compile(r'<(style|script)[^>]*>.*?</\1>', re.DOTALL)
    NAVIGATION_KEYWORDS = re.compile(
        r'(table of contents|contents|chapter \d|section \d|navigation|menu)',
        re.IGNORECASE
    )

    def __init__(self):
        self.source_name = 'epub'
        self.report_data = QualityReport(source='epub')
        self._fix_counts = {}
        self._discard_counts = {}

    def classify(self, sample: dict) -> Literal['good', 'fixable', 'discardable']:
        text = self._get_text(sample)
        if not text or len(text.strip()) < 10:
            self._discard_counts['empty_or_short'] = self._discard_counts.get('empty_or_short', 0) + 1
            return 'discardable'

        has_html = bool(self.HTML_TAG.search(text))
        has_entities = bool(re.search(r'&\w+;|&#\d+;', text))
        has_style = bool(self.STYLE_SCRIPT.search(text))

        if has_html or has_entities or has_style:
            self._fix_counts['html_entities'] = self._fix_counts.get('html_entities', 0) + 1
            return 'fixable'

        if self.NAVIGATION_KEYWORDS.search(text) and len(text) < 200:
            self._discard_counts['navigation'] = self._discard_counts.get('navigation', 0) + 1
            return 'discardable'

        self.report_data.good += 1
        return 'good'

    def fix(self, sample: dict) -> dict:
        result = dict(sample)
        for key in result:
            if isinstance(result[key], str):
                result[key] = self._clean_epub_text(result[key])
        return result

    def report(self) -> QualityReport:
        self.report_data.fixes_applied = dict(self._fix_counts)
        self.report_data.discarded_reasons = dict(self._discard_counts)
        return self.report_data

    def validate_batch(self, samples: list) -> tuple:
        cleaned = []
        self.report_data = QualityReport(source='epub')
        self._fix_counts = {}
        self._discard_counts = {}

        for sample in samples:
            classification = self.classify(sample)
            self.report_data.total_samples += 1

            if classification == 'good':
                cleaned.append(sample)
            elif classification == 'fixable':
                fixed = self.fix(sample)
                cleaned.append(fixed)
                self.report_data.fixable += 1
            else:
                self.report_data.discardable += 1

        self.report_data.good = len(cleaned) - self.report_data.fixable
        logger.info(self.report_data.summary())
        return cleaned, self.report_data

    def _get_text(self, sample: dict) -> str:
        for key in ('input_ids', 'text', 'input'):
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return ''

    def _clean_epub_text(self, text: str) -> str:
        text = self.STYLE_SCRIPT.sub('', text)
        text = html.unescape(text)
        text = self.HTML_TAG.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class WebValidator(SourceValidator):
    """Validator for web-scraped source files."""

    BOILERPLATE_KEYWORDS = re.compile(
        r'(cookie|privacy|terms of use|all rights reserved|copyright|'
        r'subscribe|newsletter|sign up|log in|sign in|search|'
        r'advertisement|sponsored|click here|read more|'
        r'facebook|twitter|instagram|linkedin|youtube)',
        re.IGNORECASE
    )
    NAVIGATION_PATTERNS = re.compile(
        r'(skip to (main|content|navigation)|back to (top|home)|'
        r'previous|next|page \d+ of \d+|showing \d+ of \d+|'
        r'loading\.\.\.|please wait|page not found)',
        re.IGNORECASE
    )
    SOCIAL_MEDIA_BOILERPLATE = re.compile(
        r'(follow us on|share this|tweet|pin it|'
        r'share on facebook|share on twitter|'
        r'like us|subscribe to our|join our)',
        re.IGNORECASE
    )
    HTML_TAG = re.compile(r'<[^>]+>')
    JS_CONTENT = re.compile(r'(function\s*\(|var\s+\w+\s*=|document\.|window\.|addEventListener)')

    def __init__(self):
        self.source_name = 'web'
        self.report_data = QualityReport(source='web')
        self._fix_counts = {}
        self._discard_counts = {}

    def classify(self, sample: dict) -> Literal['good', 'fixable', 'discardable']:
        text = self._get_text(sample)
        if not text or len(text.strip()) < 10:
            self._discard_counts['empty_content'] = self._discard_counts.get('empty_content', 0) + 1
            return 'discardable'

        has_html = bool(self.HTML_TAG.search(text))
        has_js = bool(self.JS_CONTENT.search(text))

        if has_html or has_js:
            self._fix_counts['html_js'] = self._fix_counts.get('html_js', 0) + 1
            return 'fixable'

        boilerplate_ratio = len(self.BOILERPLATE_KEYWORDS.findall(text)) / max(len(text.split()), 1)
        if boilerplate_ratio > 0.1:
            self._discard_counts['too_much_boilerplate'] = self._discard_counts.get('too_much_boilerplate', 0) + 1
            return 'discardable'

        if self.NAVIGATION_PATTERNS.search(text) and len(text) < 200:
            self._discard_counts['navigation_content'] = self._discard_counts.get('navigation_content', 0) + 1
            return 'discardable'

        social_count = len(self.SOCIAL_MEDIA_BOILERPLATE.findall(text))
        if social_count >= 2:
            self._discard_counts['social_media_boilerplate'] = self._discard_counts.get('social_media_boilerplate', 0) + 1
            return 'discardable'

        if len(text.strip()) < 30:
            self._discard_counts['content_too_short'] = self._discard_counts.get('content_too_short', 0) + 1
            return 'discardable'

        self.report_data.good += 1
        return 'good'

    def fix(self, sample: dict) -> dict:
        result = dict(sample)
        for key in result:
            if isinstance(result[key], str):
                result[key] = self._clean_web_text(result[key])
        return result

    def report(self) -> QualityReport:
        self.report_data.fixes_applied = dict(self._fix_counts)
        self.report_data.discarded_reasons = dict(self._discard_counts)
        return self.report_data

    def validate_batch(self, samples: list) -> tuple:
        cleaned = []
        self.report_data = QualityReport(source='web')
        self._fix_counts = {}
        self._discard_counts = {}

        for sample in samples:
            classification = self.classify(sample)
            self.report_data.total_samples += 1

            if classification == 'good':
                cleaned.append(sample)
            elif classification == 'fixable':
                fixed = self.fix(sample)
                cleaned.append(fixed)
                self.report_data.fixable += 1
            else:
                self.report_data.discardable += 1

        self.report_data.good = len(cleaned) - self.report_data.fixable
        logger.info(self.report_data.summary())
        return cleaned, self.report_data

    def _get_text(self, sample: dict) -> str:
        for key in ('input_ids', 'text', 'input'):
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return ''

    def _clean_web_text(self, text: str) -> str:
        text = html.unescape(text)
        text = self.HTML_TAG.sub('', text)
        text = self.JS_CONTENT.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class CSVValidator(GenericValidator):
    """Validator for CSV source files."""

    NUMBERS_ONLY = re.compile(r'^[\d\s\.\,\-\+\(\)]+$')

    def __init__(self):
        super().__init__('csv')

    def classify(self, sample: dict) -> Literal['good', 'fixable', 'discardable']:
        result = super().classify(sample)
        if result != 'discardable':
            input_text = sample.get('input', '')
            output_text = sample.get('output', '')
            if not input_text or not output_text:
                self._discard_counts['missing_fields'] = self._discard_counts.get('missing_fields', 0) + 1
                return 'discardable'
            if self.NUMBERS_ONLY.match(input_text.strip()) and self.NUMBERS_ONLY.match(output_text.strip()):
                self._discard_counts['numbers_only'] = self._discard_counts.get('numbers_only', 0) + 1
                return 'discardable'
            combined = f"{input_text} {output_text}"
            if len(combined.strip()) < 10:
                self._discard_counts['too_short_combined'] = self._discard_counts.get('too_short_combined', 0) + 1
                return 'discardable'
        return result


def get_validator(source_name: str) -> SourceValidator:
    """Factory function to get the appropriate validator for a source."""
    validators = {
        'aiml': AIMLValidator,
        'pdf': PDFValidator,
        'epub': EPUBValidator,
        'web': WebValidator,
        'csv': CSVValidator,
    }
    cls = validators.get(source_name, GenericValidator)
    if cls == GenericValidator:
        return GenericValidator(source_name)
    return cls()
